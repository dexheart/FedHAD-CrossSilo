"""Publication accounting for the historical 10-round FEMNIST campaign.

The raw log value is retained separately.  For the fixed-epoch baselines, the
publication value is the scheduled nominal workload, because all four methods
use the same 10 writers, 5 local epochs and 10 communication rounds.
"""
from __future__ import annotations

FIXED_EPOCH_METHODS = frozenset({"FedAVG", "FedAvgM", "FedProx", "FedNova"})
FLOPS_PER_SAMPLE_FULL = 116_602_368
TRAIN_EXAMPLES = 2_126
LOCAL_EPOCHS = 5
ROUNDS = 10
ROUND_TFLOPS = FLOPS_PER_SAMPLE_FULL * TRAIN_EXAMPLES * LOCAL_EPOCHS / 1e12
TOTAL_TFLOPS = ROUND_TFLOPS * ROUNDS


def is_historical_femnist_fixed(
    source_campaign: str,
    experiment_tag: str,
    dataset: str,
    method: str,
    rounds: float,
) -> bool:
    """Return whether the row uses the shared fixed-epoch nominal budget."""
    return (
        source_campaign == "base_alpha_0.5"
        and experiment_tag == "test8_femnist"
        and dataset == "FEMNIST"
        and method in FIXED_EPOCH_METHODS
        and rounds == ROUNDS
    )


def publication_tflops(
    reported_tflops: float,
    *,
    source_campaign: str,
    experiment_tag: str,
    dataset: str,
    method: str,
    rounds: float,
) -> float:
    """Return nominal publication TFLOPs without erasing the reported value."""
    if is_historical_femnist_fixed(
        source_campaign, experiment_tag, dataset, method, rounds
    ):
        return TOTAL_TFLOPS
    return reported_tflops
