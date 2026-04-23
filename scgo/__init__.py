"""SCGO: cluster global optimization tools. See README.md and runners/ for usage."""

from __future__ import annotations

# Configure PyTorch allocator so reserved memory segments can expand instead of
# fragmenting; recommended for long-running TS campaigns. Set unconditionally so
# callers do not need to export it in the shell.
import os
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# SCGO sets torch.load behavior explicitly for trusted model loads; avoid the
# global env override that triggers third-party import warnings.
os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)

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
    get_torchsim_ga_params,
    get_ts_search_params,
    get_ts_search_params_uma,
    get_uma_ga_benchmark_params,
)
from scgo.runner_api import (
    CompositionInput,
    log_go_ts_summary,
    parse_composition_arg,
    resolve_workflow_seed,
    run_go,
    run_go_campaign,
    run_go_ts,
    run_go_ts_campaign,
    run_ts_campaign,
    run_ts_search,
)

# Surface / adsorption
from scgo.surface import (
    SurfaceSystemConfig,
    adsorption_energy,
    make_graphite_surface_config,
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
                "TorchSim GA requires TorchSim. Install with: pip install 'scgo[mace]' "
                "(MACE) or 'scgo[uma]' (UMA) depending on the model family."
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
    "make_graphite_surface_config",
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
    "get_ts_search_params",
    "get_default_uma_params",
    "get_ts_search_params_uma",
    "get_torchsim_ga_params",
    "get_uma_ga_benchmark_params",
    # Main run API (see scgo.runner_api)
    "CompositionInput",
    "resolve_workflow_seed",
    "run_go",
    "run_go_campaign",
    "run_go_ts",
    "run_go_ts_campaign",
    "log_go_ts_summary",
    "parse_composition_arg",
    "run_ts_campaign",
    "run_ts_search",
    # Utilities
    "get_cluster_formula",
    "is_true_minimum",
    "perform_local_relaxation",
]
