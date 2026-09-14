# SRI Playground (Colab notebook)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrieleSerussi/sri-project-page/blob/main/colab/sri_playground.ipynb)


`sri_playground.ipynb` accompanies the paper *Step-Wise Refusal Dynamics in Autoregressive and Diffusion Language Models* ([arXiv:2602.02600](https://arxiv.org/abs/2602.02600); official code at [ElironRahimi/sri-signal](https://github.com/ElironRahimi/sri-signal)). A reader types any prompt, the notebook generates a response with a language model of their choice, and an animated widget shows the tokens appearing step by step next to the Step-Wise Refusal Internal Dynamics (SRI) signal sigma_t of that generation, so the internal turn toward or away from refusal can be watched as it happens.

## What the notebook does

1. Loads a model (default `Qwen/Qwen2.5-1.5B-Instruct`, autoregressive, ungated, fits a free T4).
2. Builds the two SRI anchors for that model: for 32 harmless prompts (Alpaca) and 32 harmful prompts (AdvBench, or a built-in list when the gated dataset is not accessible) it generates T = 32 steps, mean-pools the last-layer hidden states of the generated tokens at every step, and averages them per step into mu_t^harmless and mu_t^harmful. The anchors are cached with `torch.save`.
3. Generates the reader's prompt, records phi_t at each step, and computes sigma_t = sigmoid((log(d_t^harmful + eps) - log(d_t^harmless + eps)) / tau) with tau = 0.1, where d_t are cosine distances to the anchors. It also records the decoded text after every step and whether an entry of the paper's refusal dictionary (Table 12) appears in it, which is the text-level refusal signal.
4. Renders an HTML/JavaScript animation (token grid, state tag with the paper's thresholds 0.5 and 0.3, SVG sparkline of sigma_t, dashed text-level signal, play/pause and slider), a static matplotlib figure, and optionally a GIF.
5. Optionally compares two models on the same prompt and trains a miniature SRI Guard autoencoder on the harmless anchor signals.

## Hardware and expected runtimes

| Model | Family | Hardware | Notes | Anchor stage (32 + 32 prompts) |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-1.5B-Instruct` (default) | autoregressive | free Colab T4 (fp16) | ungated, about 3 GB download | under 1 minute of generation; the whole notebook, including installs and download, in roughly 5 to 10 minutes |
| `meta-llama/Llama-3.1-8B-Instruct` | autoregressive | T4 in 8-bit, or A100 in bf16 | gated: accept the licence on Hugging Face and provide a token (form field or Colab secret `HF_TOKEN`); 16 GB download | a few minutes on a T4 in 8-bit |
| `GSAI-ML/LLaDA-8B-Instruct` | diffusion (masked, low-confidence remasking) | T4 in 8-bit, or A100 in bf16 (about 16 GB) | one prompt at a time, 32 full-sequence forward passes over a 128-token response per prompt | tens of minutes on a T4; use 16 + 16 anchor prompts there |

Runtimes are estimates; the notebook prints the measured anchor time. On CPU the default model is too slow to be pleasant; the notebook was smoke-tested on CPU only with a 0.5B model and reduced settings (see below). The T4 does not support bf16 natively, so the notebook uses fp16 there and bf16 on Ampere or newer GPUs; `precision` can force fp32. Dream is not offered because the official repository has no wrapper for it.

## Cells

| Section | Cell | Content |
| --- | --- | --- |
| Title | markdown | What SRI is, links to the paper, the code and the project page (TODO placeholder), content warning, note on the reduced anchor sets, how to run. |
| 1. Setup | code | GPU check, pinned installs (`transformers==4.57.6`, `accelerate==1.10.1`, `datasets==4.5.0`, `huggingface_hub==0.36.2`, plus `bitsandbytes==0.50.2` on GPU), recursive clone of `sri-signal`, imports of the official code. |
| 2a. SRI core | code | `ARWrapper` and `LLaDAWrapper` (subclasses of the official `BaseWrapper`), `Trace` dataclass, `load_wrapper`, dtype handling, chat formatting, `sri_from_activations`, text-level refusal signal. |
| 2b. Model settings | code | Form fields: model id (dropdown with free input), `steps`, `precision`, `load_in_8bit`, automatic 8-bit for large models on small GPUs, `batch_size`, `hf_token`, `gen_length` and `block_length` for LLaDA. Loads the model. |
| 3. Anchors | code | Form fields `n_harmless`, `n_harmful`, `force_recompute`. Loads prompts (official `read_alpaca` and `read_advbench` with built-in fallbacks), generates, averages with the official `compute_centers` (fallback to plain means), caches to `sri_anchors/`, prints refusal rates, computes reference trajectories. |
| 4. Your prompt | code | Form fields `prompt` and `apply_refusal_suppression`. Generates T steps, computes sigma_t and the text-level signal, prints a step table and the sigma array. |
| 5a. Animation | code | `render_sri_animation(trace, refs)` returns the self-contained HTML widget. |
| 5b. Static figure | code | `plot_sri` with the compliance, transition and refusal zones, anchor references and the dashed text-level signal. |
| 5c. GIF export | code | Optional (`make_gif`), `matplotlib.animation` with `PillowWriter`. |
| 6. Compare | code | Optional (`run_comparison`): second model, its own anchors, overlay of the two curves and a second animation. |
| 7. SRI Guard | code | Optional (`run_guard`, on by default because it takes seconds): autoencoder T-32-8-32-T, L2 loss, 99th percentile threshold on held-out harmless signals, score of the reader's prompt. |
| Cite | markdown | BibTeX and links. |

Each section is preceded by a markdown cell that explains what it does.

## What comes from the official repository and what is notebook-local

Reused from `ElironRahimi/sri-signal` (cloned at runtime, imported from the clone):

- `SRI_alg.create_signal` and `SRI_alg.cosine_dist`: the SRI signal itself (cosine distances, smoothed log-ratio with `sig_T = 0.1`, sigmoid).
- `SRI_alg.compute_centers`: per-step anchors with the official filtering (harmless anchor from non-refused harmless generations, harmful anchor from refused harmful generations).
- `utils.GenerationProfile`: the generation and signal configuration (`steps = 32`, `gen_length = 128`, `block_length = 128`, `temperature = 0`, `remasking = "low_confidence"`, `sig_T = 0.1`).
- `utils.refusalJudge` and `utils.REFUSAL_KEYWORDS`: the refusal dictionary of Table 12 and its matcher, used both for the anchor filtering and for the text-level refusal signal.
- `utils.read_alpaca` and `utils.read_advbench`: the anchor prompt sources.
- `Models.ModelWrapper.BaseWrapper`: the abstract wrapper that both notebook wrappers subclass, implementing its two abstract methods `_load_model` and `_collectActivationsAndRefusals`.
- `submodules/llada/generate.py` (`add_gumbel_noise`, `get_num_transfer_tokens`): imported when the LLaDA submodule was cloned; otherwise the notebook uses verbatim copies.

Notebook-local (not in the repository):

- `ARWrapper`: a model-agnostic autoregressive wrapper (any Hugging Face causal LM, batched generation with left padding, fp16/bf16/fp32 or 8-bit, EOS handling) with per-step token and text capture for the animation. The official `LlamaWrapper` is hard-coded to LLaMA-3.1-8B in fp16 without quantisation and has no per-step trace.
- `LLaDAWrapper._generate_with_trace`: a port of the official LLaDA sampler (fixed `gen_length`, `steps`, `block_length`, temperature 0, low-confidence remasking) that records, at every step, the committed tokens, the current prediction for every masked position and phi_t. The official `LLaDAWrapper` calls the submodule's `generate` and hooks `transformer.ln_f`; this notebook hooks the same module.
- Anchor caching keyed by model, T and anchor counts (the official `generateCenter` caches under `Results/<model>/centers.pt` regardless of the anchor count), the reference trajectories, the prompt fallbacks, the refusal-suppression wrapper, the animation, the figures, the comparison cell and the SRI Guard cell.

Two deliberate differences from the official wrappers, documented in the notebook: following Section 3.1 of the paper, phi_t is the mean of the last-layer hidden states of the generated tokens only. The official `LlamaWrapper` averages the hidden state at the position that produced each token (one position earlier, so it includes the last prompt token), and the official `LLaDAWrapper` averages over every response position including the still-masked ones. The paper's Appendix D, Algorithm 1 also writes the log-ratio with the opposite sign to Section 3.1; the notebook follows Section 3.1 and the official `create_signal`, under which 1 means compliance-aligned.

The repository's `requirements.txt` is not installed because it is unpinned and lists torch, torchvision and torchaudio, which would reinstall Colab's torch.

### Licence

The `sri-signal` repository contains no LICENSE file and GitHub reports no licence for it (checked on 2 September 2026), so there is no licence text to quote. Its code is used here by cloning it at runtime rather than by copying it, except for the two small sampler helpers copied from the LLaDA repository as a fallback. Ask the authors before redistributing their code.

## Reduced anchor sets

The paper uses 400 harmless and 400 harmful anchor prompts per model, a separate harmless test set, and trains SRI Guard on 1,200 harmless prompts with 200 more for calibration. This notebook defaults to 32 and 32 anchors and trains the miniature SRI Guard on the anchor signals themselves. Absolute sigma values and the guard's threshold therefore differ from the paper and are noisier; the qualitative behaviour (compliance-aligned trajectories for harmless prompts, refusal-aligned ones for refused prompts, unstable ones for jailbreaks) is what the notebook shows. AdvBench is a gated dataset; without a Hugging Face token the notebook uses a built-in list of 32 generic harmful requests of the same kind instead, and says so.

## How this notebook was tested

The machine used to write the notebook has no GPU and no system torch, so testing used an isolated virtual environment (Python 3.9, CPU wheels: torch 2.8.0, transformers 4.57.6, accelerate 1.10.1, datasets 4.5.0, huggingface_hub 0.36.2, nbformat, nbclient, ipykernel, matplotlib) and reduced settings. What was executed:

1. **Full notebook run on CPU with nbclient**, after a driver set the form fields to `install_dependencies = False`, `model_name = "Qwen/Qwen2.5-0.5B-Instruct"`, `steps = 12`, `batch_size = 4`, `n_harmless = 4`, `n_harmful = 4`, `make_gif = True`. All ten code cells ran without an exception in 17 to 19 seconds (after the one-off model download): the repository was cloned, the official modules imported, 4 + 4 anchors built with the official `compute_centers` (0/4 harmless and 4/4 harmful prompts refused), the prompt "Write a short poem about the sea." produced a 12-value sigma array (0.84 to 0.97, compliant throughout, no text-level refusal), the animation cell emitted its HTML widget, the static figure produced a PNG, the GIF was written (71 KB), and the miniature SRI Guard trained and scored the prompt. AdvBench was not accessible without a token, so the built-in harmful list was used, which exercised that fallback.
2. **The same run with the comparison enabled** (`run_comparison = True`, `second_model = "HuggingFaceTB/SmolLM2-135M-Instruct"`, 8-bit off): the second model loaded, built its own anchors (the 135M model refused nothing, which exercised the fallback to unfiltered means), traced the same prompt, and the overlay figure and second animation were produced. Zero errors.
3. **Animation JavaScript in a real browser**: the HTML emitted by the animation cell (autoregressive run) and by the diffusion wrapper (see item 4) was opened in Chromium; no console errors, autoplay advanced the step counter, filled the token chips, updated the state tag, the sigma value, the text panel, the sparkline and the slider.
4. **Diffusion path with a tiny randomly initialised LLaDA**: the official `configuration_llada.py` and `modeling_llada.py` were loaded through `AutoModel.from_config(..., trust_remote_code=True)` under transformers 4.57.6 with `d_model = 64`, two layers and the real LLaDA tokenizer and mask id. `LLaDAWrapper` produced `[steps, D]` activations for anchor prompts, the committed-token count grew monotonically to `gen_length` for both `block_length = gen_length` and two blocks, `trace_prompt` returned committed and predicted tokens for every position at every step, and the diffusion animation rendered. The weights were random, so this checks shapes and control flow, not the behaviour of the real model.
5. `nbformat.validate` passes on the shipped notebook, which is stored without outputs.

Not tested here, because it needs a GPU or gated access: the pinned `pip install` line inside Colab, 8-bit loading with bitsandbytes, `meta-llama/Llama-3.1-8B-Instruct`, and the real `GSAI-ML/LLaDA-8B-Instruct` weights (memory and speed on a T4 in 8-bit are estimates). The default model and settings were also not timed on a T4; the five to ten minute figure is an estimate based on the model size and the number of forward passes (64 prompts, 33 decoding steps each, in batches of 8).
