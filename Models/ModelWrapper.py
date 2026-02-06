# base_wrapper.py

import gc
import torch
from abc import ABC, abstractmethod
from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import GenerationProfile
from SRI_alg import compute_centers, create_signal


class BaseWrapper(ABC):
    """
    Abstract base class for signal-based language model wrappers.

    This class implements model-agnostic logic for:
    - computing activation centers for harmless vs. harmful behavior
    - generating stepwise SRI signals from activations
    - visualizing signals and decision regions

    Subclasses are responsible for:
    - loading a concrete model and tokenizer
    - defining what a “step” corresponds to (e.g., AR token or diffusion step)
    - collecting step-aligned activations and refusal labels
    """

    def __init__(
        self,
        generation_profile: GenerationProfile = GenerationProfile(),
        device: str = "cuda",
    ):
        """
        Args:
            generation_profile:
                Configuration controlling generation and signal extraction
                (e.g., number of steps, temperature, signal temperature).
            device:
                Target computation device ("cuda" or "cpu").
        """
        self.generation_profile = generation_profile
        self.device = device

        # These are initialized by subclasses in _load_model()
        self.model = None
        self.tokenizer = None
        self.FILE_NAME = None

    # ============================================================
    # Abstract model-specific methods
    # ============================================================

    @abstractmethod
    def _load_model(self):
        """
        Load the underlying model and tokenizer and move the model to `self.device`.
        """
        raise NotImplementedError

    @abstractmethod
    def _collectActivationsAndRefusals(self, prompts: List[str]):
        """
        Run prompts through the model and collect activations and refusal labels.

        Must return:
            activations_list:
                List of tensors, one per prompt.
                Each tensor contains step-aligned activation summaries
                Shape of tensor must be [steps, D]
            refusals_list:
                List of booleans, one per prompt, indicating whether the
                final generated response is classified as a refusal.

        Subclasses define:
        - what constitutes a single “step”
        - how activations are collected and summarized
        - how refusals are detected
        """
        raise NotImplementedError

    # ============================================================
    # Model-independent methods
    # ============================================================

    def generateCenter(self, harmless_prompts: List[str], harmful_prompts: List[str]):
        """
        Compute and cache activation centers for harmless and harmful prompts.

        For each group, activations are collected using the subclass implementation
        and aggregated into class-wise center representations.

        Returns:
            centers:
                A tuple (harmless_center, harmful_center), as produced by
                `compute_centers`.

        Side effects:
            Saves centers to:
                Results/{FILE_NAME}/centers.pt
        """
        assert self.model is not None, "Model must be initialized"

        base_dir = os.path.join("Results", self.FILE_NAME)
        center_path = os.path.join(base_dir, "centers.pt")
        os.makedirs(base_dir, exist_ok=True)

        # Load cached centers if they already exist
        if os.path.isfile(center_path):
            centers = torch.load(center_path, map_location="cpu")
            print("Centers found on disk, skipping regeneration.")
            return centers

        # Collect activations and refusal labels
        harmless_acts, harmless_ref = self._collectActivationsAndRefusals(harmless_prompts)
        harmful_acts, harmful_ref = self._collectActivationsAndRefusals(harmful_prompts)

        # Compute class-wise centers
        centers = compute_centers(
            harmless_acts,
            harmless_ref,
            harmful_acts,
            harmful_ref,
        )

        # Persist centers to disk
        torch.save(centers, center_path)

        return centers

    def generateSignal(self, prompts: List[str]):
        """
        Generate an SRI signal for each prompt.

        Each signal quantifies, at every generation step, how closely the
        prompt's activations align with the harmless vs. harmful centers.

        Returns:
            signal_list:
                List of tensors, one per prompt.
                Each tensor has shape [steps].
        """
        assert self.model is not None, "Model must be initialized"

        activations_list, _ = self._collectActivationsAndRefusals(prompts)

        base_dir = os.path.join("Results", self.FILE_NAME)
        center_path = os.path.join(base_dir, "centers.pt")

        if not os.path.isfile(center_path):
            raise FileNotFoundError(f"Missing centers file: {center_path}")

        centers = torch.load(center_path, map_location="cpu")

        # Compute stepwise SRI signal for each prompt
        signal_list = create_signal(
            centers=centers,
            activations=activations_list,
            sig_T=self.generation_profile.sig_T,
        )

        return signal_list

    def generateImage(
        self,
        signals: list,
        fig_name: str = "signal_example.png",
    ):
        """
        Visualize a test signal together with harmless and harmful reference signals.

        Args:
            signals:
                List (or tuple) of exactly three tensors in the following order:
                    [test_signal, harmless_signal, harmful_signal]
                Each tensor must have shape [steps].
            fig_name:
                Filename for the saved figure (relative to Results/{FILE_NAME}/).
        """
        assert isinstance(signals, (list, tuple)), "signals must be a list or tuple"
        assert len(signals) == 3, "signals must contain exactly three tensors"

        # Required order: test, harmless, harmful
        signal, harmless, harmful = signals

        signal_np = signal.detach().cpu().numpy()
        harmless_np = harmless.detach().cpu().numpy()
        harmful_np = harmful.detach().cpu().numpy()

        plt.figure(figsize=(8, 4))

        # --- Background regions ---
        plt.axhspan(0.5, 1.05, color="#C8E6C9", alpha=0.9, label="Compliance Region")
        plt.axhspan(-0.05, 0.5, color="#FFCDD2", alpha=0.9, label="Refusal Region")

        # --- Signals ---
        plt.plot(harmless_np, color="#1B5E20", lw=2.8, label="Harmless Center (SRI Signal)")
        plt.plot(harmful_np,  color="#B71C1C", lw=2.8, label="Harmful Center (SRI Signal)")
        plt.plot(signal_np,   color="#0D47A1", lw=2.8, label="Test Signal (SRI Signal)")

        # --- Decision threshold ---
        plt.axhline(0.5, color="gray", lw=1.8, ls=":")

        # --- Axes ---
        plt.xlabel("Generation Step", fontsize=20, fontweight="bold")
        plt.ylabel("Refusal Score", fontsize=20, fontweight="bold")
        plt.ylim(-0.05, 1.05)
        plt.xlim(0, len(signal_np) - 1)

        # --- Tick styling ---
        plt.tick_params(axis="both", labelsize=13)
        for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            tick.set_fontweight("bold")

        # --- Legend ---
        plt.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 0.10),
            frameon=True,
            framealpha=0.95,
            prop={"size": 12, "weight": "bold"},
        )

        plt.tight_layout()

        model_dir = os.path.join("Results", self.FILE_NAME)
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, fig_name)
        plt.savefig(save_path, dpi=300)

        plt.show()
        plt.close()

        print(f"Signal plot saved to {save_path}")
