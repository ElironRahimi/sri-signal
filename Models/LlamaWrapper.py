# llama_wrapper.py

import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import os

from Models.ModelWrapper import BaseWrapper
from utils import GenerationProfile, get_prompt_tensor, refusalJudge


class LlamaModelWrapper(BaseWrapper):
    """
    Wrapper for Meta Llama 3.1 autoregressive (AR) causal-LM models.

    What this wrapper does:
    - Loads a causal LM (next-token prediction)
    - Runs explicit token-by-token decoding for a fixed number of steps
      (one decoding step ~= one generated token)
    - Collects step-aligned hidden-state summaries (one per decoding step)
    - Produces one refusal label per prompt, based on the final decoded text

    Output of _collectActivationsAndRefusals:
    - activations_list: list of tensors, one per prompt, each shaped [steps, D]
      where D is the hidden size from the final layer.
    - refusal_results: list of booleans, one per prompt.
    """

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(
        self,
        generation_profile: GenerationProfile = GenerationProfile(),
        device: str = "cuda",
    ):
        super().__init__(generation_profile=generation_profile, device=device)

    # ============================================================
    # Model loading
    # ============================================================

    def _load_model(self):
        """
        Load tokenizer + causal LM, move model to device, and set eval mode.
        """
        print(f"[Llama-3.1] Loading model {self.MODEL_NAME}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast=True)

        # Some Llama tokenizers may not define a pad token.
        # We set pad_token_id to eos_token_id for compatibility with downstream utilities.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16,
        )

        self.model.to(self.device)
        self.model.eval()

        # Used for saving results under Results/{FILE_NAME}/
        self.FILE_NAME = "llama_3"

        print("[Llama-3.1] Model loaded.")

    # ============================================================
    # Token sampling helpers
    # ============================================================

    @torch.no_grad()
    def _sample_next_token(self, logits: torch.Tensor) -> int:
        """
        Sample a next-token id from a single-step logits vector.

        Args:
            logits: Tensor of shape [vocab] (or [1, vocab] depending on caller)
                    containing next-token logits for one decoding step.

        Returns:
            next_token_id: Python int token id.
        """
        temp = float(self.generation_profile.temperature)

        # temperature <= 0 => greedy decoding
        if temp is None or temp <= 0.0:
            return int(torch.argmax(logits).item())

        probs = torch.softmax(logits / temp, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        return int(next_id.item())

    # ============================================================
    # Collect activations & refusals
    # ============================================================

    @torch.no_grad()
    def _collectActivationsAndRefusals(self, prompts: List[str]):
        """
        For each prompt:
        - Encode prompt
        - Iteratively decode up to `generation_profile.steps` tokens
        - At each step, record the last-token hidden state from *all layers*
        - Convert per-step hidden states into a running average over steps
        - Keep only the final-layer representation per step -> [steps, D]
        - Decode generated tokens into text and run refusalJudge()

        Notes:
        - We stop early if EOS is generated, and pad remaining steps by repeating
          the last hidden state so every sample has exactly `steps` entries.
        - This implementation currently uses greedy decoding (argmax) inside the loop.
        """
        EOS_TOKEN_ID = self.tokenizer.eos_token_id
        activations_list = []
        refusal_results = []

        for idx, prompt in tqdm(
            enumerate(prompts),
            total=len(prompts),                 # ensures a full progress bar
            desc="Collecting acts/refusals",
            dynamic_ncols=True,
            leave=True,
        ):
            # Tokenize prompt
            prompt_tensor, _ = get_prompt_tensor(prompt, self.tokenizer, self.device)

            # generated_ids grows by 1 token each decoding step
            generated_ids = prompt_tensor.clone()

            # Will store per-step hidden states: each entry is [layers, D]
            all_hidden_states = []

            # Store generated token ids (excluding the prompt)
            generated_answer = []

            for gen_idx in range(self.generation_profile.steps):
                outputs = self.model(
                    input_ids=generated_ids,
                    output_hidden_states=True,
                )

                # outputs.hidden_states is a tuple of length (layers+1)= L
                # each element is [B, seq_len, D]. Here B=1.
                hidden_stack = torch.stack(outputs.hidden_states).squeeze(1)  # [L, seq_len, D]

                # Take the last position (the current next-token context) for each layer/state
                last_hidden = hidden_stack[:, -1, :]  # [L, D]
                all_hidden_states.append(last_hidden)

                # Next-token logits for the last position
                next_token_logits = outputs.logits[:, -1, :]  # [1, vocab]

                # Greedy decoding (argmax). If you want sampling, call _sample_next_token().
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]

                generated_answer.append(next_token.squeeze(0))  
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Early stop: if EOS, pad remaining steps with the last hidden state
                if next_token.item() == EOS_TOKEN_ID:
                    remaining = self.generation_profile.steps - (gen_idx + 1)
                    all_hidden_states.extend([last_hidden] * remaining)
                    break

            # Stack -> [steps, L, D]
            hs = torch.stack(all_hidden_states, dim=0)  # [steps, L, D]

            # Running (cumulative) mean over steps:
            # cumsum[t] / (t+1) gives mean hidden state up to step t
            cumsum = hs.cumsum(dim=0)
            counts = torch.arange(1, cumsum.size(0) + 1, device=cumsum.device).view(-1, 1, 1)
            activation = cumsum / counts  # [steps, L, D]

            # Keep ONLY the final layer/state per step -> [steps, D]
            activation = activation[:, -1, :]
            activations_list.append(activation.cpu())

            # Decode generated tokens (not including the prompt)
            generated_answer = torch.stack(generated_answer).squeeze(1)  # [T]
            response = self.tokenizer.batch_decode(
                generated_answer.unsqueeze(0),
                skip_special_tokens=True
            )[0]

            refusal_results.append(refusalJudge(response))

        return activations_list, refusal_results
