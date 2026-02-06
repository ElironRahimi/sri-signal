# ============================================================
# Signal generation
# ============================================================

import torch


def create_signal(centers, activations, sig_T=0.1, eps=1e-8):
    """
    Compute a stepwise refusal signal from activations and class centers.

    For each generation step t, the signal compares the cosine distance
    between the activation at that step and:
      - the harmless (non-refusal) center
      - the harmful (refusal) center

    The relative distance is mapped to a scalar in [0, 1] using a
    temperature-scaled logistic function.

    Args:
        centers:
            Tuple (harmless_center, harmful_center),
            each a tensor of shape [steps, dim].
        activations:
            List of activation tensors, one per prompt.
            Each tensor has shape [steps, dim].
        sig_T:
            Temperature controlling the sharpness of the signal
            (lower => sharper separation).
        eps:
            Small constant for numerical stability.

    Returns:
        signals:
            List of tensors, one per prompt.
            Each tensor has shape [steps] and values in [0, 1].
    """
    center_good, center_bad = centers
    signals = []

    for act in activations:
        signal_l = []

        for t in range(act.shape[0]):
            x_t = act[t]          # activation at step t
            g_t = center_good[t]  # harmless center at step t
            b_t = center_bad[t]   # harmful center at step t

            d_good = cosine_dist(x_t, g_t, eps)
            d_bad  = cosine_dist(x_t, b_t, eps)

            # Log-distance ratio mapped through a sigmoid
            logit = (torch.log(d_bad + eps) - torch.log(d_good + eps)) / sig_T
            score = torch.sigmoid(logit)

            signal_l.append(score.item())

        signals.append(torch.tensor(signal_l))

    return signals


def cosine_dist(x, y, eps=1e-8):
    """
    Compute cosine distance between two vectors.

    Cosine distance is defined as:
        1 - cosine_similarity(x, y)

    Args:
        x, y:
            1D tensors of the same shape.
        eps:
            Small constant to avoid division by zero.

    Returns:
        Scalar cosine distance.
    """
    x = x / (x.norm() + eps)
    y = y / (y.norm() + eps)
    return 1.0 - torch.dot(x, y)


# ============================================================
# Center computation
# ============================================================

def compute_centers(harmless_act_list, harmless_ref, harmful_act_list, harmful_ref):
    """
    Compute stepwise activation centers for harmless and harmful behavior.

    Filtering convention:
        - Harmless center:
            computed from *non-refusal* samples in the harmless set
        - Harmful center:
            computed from *refusal* samples in the harmful set

    This ensures that each center reflects the characteristic behavior
    actually expressed by the model in each category.

    Args:
        harmless_act_list:
            List of activation tensors [steps, dim] for harmless prompts.
        harmless_ref:
            List of booleans indicating refusals for harmless prompts
            (True = refusal).
        harmful_act_list:
            List of activation tensors [steps, dim] for harmful prompts.
        harmful_ref:
            List of booleans indicating refusals for harmful prompts
            (True = refusal).

    Returns:
        harmless_center:
            Tensor of shape [steps, dim] representing the harmless center.
        harmful_center:
            Tensor of shape [steps, dim] representing the harmful center.
    """
    if len(harmless_act_list) != len(harmless_ref):
        raise ValueError("harmless_ref length does not match harmless_act_list")
    if len(harmful_act_list) != len(harmful_ref):
        raise ValueError("harmful_ref length does not match harmful_act_list")

    # Select valid samples for each class
    filtered_harmless = [
        act for act, r in zip(harmless_act_list, harmless_ref) if not r
    ]
    filtered_harmful = [
        act for act, r in zip(harmful_act_list, harmful_ref) if r
    ]

    if len(filtered_harmless) == 0:
        raise ValueError("No non-refusal samples found in harmless set.")
    if len(filtered_harmful) == 0:
        raise ValueError("No refusal samples found in harmful set.")

    # Mean over samples to obtain stepwise centers
    harmless_center = torch.stack(filtered_harmless, dim=0).mean(dim=0)
    harmful_center  = torch.stack(filtered_harmful,  dim=0).mean(dim=0)

    return harmless_center, harmful_center
