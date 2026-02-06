# llada_wrapper.py

import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

from Models.ModelWrapper import BaseWrapper
from utils import GenerationProfile, get_prompt_tensor, refusalJudge
from submodules.llada.generate import generate


class LLADAModelWrapper(BaseWrapper):
    """
    Wrapper for LLaDA diffusion language model.

    Responsibilities:
    - Load the LLaDA model and tokenizer
    - Register forward hooks to capture hidden activations
    - Run diffusion-style generation
    - Aggregate per-step activations and refusal labels
    """

    MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

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
        Load the LLaDA tokenizer and model, move the model to the
        target device, and switch it to evaluation mode.
        """
        print(f"[LLaDA] Loading model {self.MODEL_NAME}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.model.to(self.device)
        self.model.eval()

        # Used for saving results under Results/{FILE_NAME}/
        self.FILE_NAME = "llada"

        print("[LLaDA] Model loaded.")

    # ============================================================
    # Forward-hook setup
    # ============================================================

    def generateHook(self):
        """
        Register a forward hook on the model's final layer normalization
        module to capture hidden states.

        The captured activations correspond to the final transformer
        hidden states (pre-output projection) for each forward pass.

        Returns:
            hook_fn:
                The hook function itself.
            handle:
                The hook handle (must be removed after use).
            activations:
                A list populated with tensors of shape [B, L, D],
                one entry per forward call.
        """
        assert self.model is not None, "Model must be initialized"

        activations = []

        # Prefer the inner model if wrapped (common in HF models)
        core = getattr(self.model, "model", self.model)

        # Final layer norm before output head
        module_to_hook = core.transformer.ln_f

        # ---------------------------
        # Hook function
        # ---------------------------
        def hook_fn(module, inputs, output):
            # Some modules return tuples; extract the tensor if needed
            out = output[0] if isinstance(output, (tuple, list)) else output
            activations.append(out.detach().clone())

        # ---------------------------
        # Register hook
        # ---------------------------
        handle = module_to_hook.register_forward_hook(hook_fn)

        return hook_fn, handle, activations

    # ============================================================
    # Activation + refusal collection
    # ============================================================

    def _collectActivationsAndRefusals(self, prompts: List[str]):
        """
        For each prompt:
        - Run diffusion generation
        - Collect per-forward-call activations via hooks
        - Aggregate activations into a per-step representation
        - Record whether the final response is a refusal

        Returns:
            activations_list:
                List of tensors, one per prompt.
                Each tensor has shape [steps, D].
            refusal_results:
                List of booleans indicating refusal for each prompt.
        """
        activations_list = []
        refusal_results = []

        # Install hook once for all prompts
        hook_fn, handle, step_activations = self.generateHook()
        assert handle is not None, "Hook handle is None"

        for prompt in tqdm(
            prompts,
            desc="Collecting activations",
            ncols=100,
            leave=False,
            dynamic_ncols=True,
        ):
            # Clear activations collected from previous prompt
            step_activations.clear()

            # Tokenize prompt
            prompt_tensor, _ = get_prompt_tensor(
                prompt,
                self.tokenizer,
                self.device
            )
            prompt_len = prompt_tensor.shape[1]

            # Run LLaDA diffusion generation
            output = generate(
                self.model,
                prompt_tensor,
                steps=self.generation_profile.steps,
                gen_length=self.generation_profile.gen_length,
                block_length=self.generation_profile.block_length,
                temperature=self.generation_profile.temperature,
                cfg_scale=self.generation_profile.cfg_scale,
                remasking="low_confidence",
            )

            # Aggregate activations per forward call
            acts = []
            for act in step_activations:
                act = act.detach().cpu()

                # Keep only generated tokens (exclude prompt tokens)
                response_acts = act[-1, prompt_len:, :]

                # Average over token dimension → [D]
                avg_act = response_acts.mean(dim=0)
                acts.append(avg_act)

            # Stack into [steps, D]
            prompt_acts = torch.stack(acts, dim=0)
            activations_list.append(prompt_acts)

            # Decode final response for refusal detection
            response = self.tokenizer.batch_decode(
                output.squeeze(1)[:, prompt_len:],
                skip_special_tokens=True,
            )[-1]

            refusal_results.append(refusalJudge(response))
            torch.cuda.empty_cache()

        # Always remove hook
        handle.remove()

        return activations_list, refusal_results
