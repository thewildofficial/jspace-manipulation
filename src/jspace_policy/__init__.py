"""Utilities for the When Words Override Consequences study."""

from jspace_policy.dataset import generate_dataset

# sprint_analysis (pandas/numpy) is lazy: the Modal GPU image ships only
# torch/transformers, and importing this package remotely must not pull the
# local-CPU analysis stack. Direct `jspace_policy.sprint_analysis` imports
# are unaffected.
_LAZY = {
    "paired_base_effects": "jspace_policy.sprint_analysis",
    "grouped_paired_analysis": "jspace_policy.sprint_analysis",
    "primary_gate": "jspace_policy.sprint_analysis",
    "simulate_primary_power": "jspace_policy.sprint_analysis",
    "incident_gate": "jspace_policy.sprint_analysis",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY])


__all__ = [
    "generate_dataset",
    "paired_base_effects",
    "grouped_paired_analysis",
    "primary_gate",
    "simulate_primary_power",
    "incident_gate",
]
