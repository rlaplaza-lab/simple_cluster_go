"""SCGO: cluster global optimization tools. See README.md and runners/ for usage."""

from __future__ import annotations

# Configure PyTorch allocator so reserved memory segments can expand instead of
# fragmenting; recommended for long-running TS campaigns. Set unconditionally so
# callers do not need to export it in the shell.
import os
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Algorithms
from scgo.algorithms import (
    bh_go,
    ga_go,
)

# Cluster + adsorbate (composable local relax)
from scgo.cluster_adsorbate import (
    ClusterAdsorbateConfig,
    ClusterOHConfig,
    attach_fix_bond_lengths,
    attach_oh_bond_constraint,
    combine_core_adsorbate,
    place_fragment_on_cluster,
    place_oh_on_cluster,
    relax_metal_cluster_with_adsorbate,
    relax_metal_cluster_with_oh,
    validate_combined_cluster_structure,
)

# Database
from scgo.database import (
    SCGODatabaseManager,
    load_previous_run_results,
    load_reference_structures,
    setup_database,
)

# Initialization
from scgo.initialization import (
    create_initial_cluster,
    generate_template_structure,
    validate_cluster_structure,
)

# Parameter presets
from scgo.param_presets import (
    AVAILABLE_MACE_MODELS,
    AVAILABLE_UMA_MODELS,
    get_default_params,
    get_default_uma_params,
    get_diversity_params,
    get_high_energy_params,
    get_minimal_ga_params,
    get_testing_params,
    get_ts_run_kwargs,
    get_ts_search_params_uma,
)

# Main run functions
from scgo.run_minima import (
    parse_composition_arg,
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_campaign_one_element,
    run_scgo_campaign_two_elements,
    run_scgo_trials,
)

# Transition state search
from scgo.run_ts import (
    run_transition_state_campaign,
    run_transition_state_search,
)

# Surface / adsorption
from scgo.surface import (
    SurfaceSystemConfig,
    adsorption_energy,
)

# Utilities
from scgo.utils.helpers import (
    get_cluster_formula,
    is_true_minimum,
    perform_local_relaxation,
)
from scgo.utils.logging import (
    configure_logging,
    get_logger,
)

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "ga_go_torchsim":
        try:
            from scgo.algorithms.geneticalgorithm_go_torchsim import ga_go_torchsim
        except ImportError as e:
            raise ImportError(
                "TorchSim GA requires the MACE stack. Install with: pip install 'scgo[mace]'"
            ) from e
        return ga_go_torchsim
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # Version and capabilities
    "__version__",
    "AVAILABLE_MACE_MODELS",
    "AVAILABLE_UMA_MODELS",
    # Algorithms (for advanced users)
    "bh_go",
    "ga_go",
    "ga_go_torchsim",
    # Database
    "load_previous_run_results",
    "load_reference_structures",
    "SCGODatabaseManager",
    "setup_database",
    # Initialization
    "create_initial_cluster",
    "generate_template_structure",
    "validate_cluster_structure",
    # Surface
    "SurfaceSystemConfig",
    "adsorption_energy",
    # Cluster + adsorbate
    "ClusterAdsorbateConfig",
    "ClusterOHConfig",
    "attach_fix_bond_lengths",
    "attach_oh_bond_constraint",
    "combine_core_adsorbate",
    "place_fragment_on_cluster",
    "place_oh_on_cluster",
    "relax_metal_cluster_with_adsorbate",
    "relax_metal_cluster_with_oh",
    "validate_combined_cluster_structure",
    # Logging
    "configure_logging",
    "get_logger",
    # Parameter presets
    "get_default_params",
    "get_diversity_params",
    "get_high_energy_params",
    "get_minimal_ga_params",
    "get_testing_params",
    "get_ts_run_kwargs",
    "get_default_uma_params",
    "get_ts_search_params_uma",
    # Main run API
    "run_scgo_campaign_arbitrary_compositions",
    "run_scgo_campaign_one_element",
    "run_scgo_campaign_two_elements",
    "run_scgo_trials",
    "parse_composition_arg",
    # Transition states
    "run_transition_state_campaign",
    "run_transition_state_search",
    # Utilities
    "get_cluster_formula",
    "is_true_minimum",
    "perform_local_relaxation",
]
