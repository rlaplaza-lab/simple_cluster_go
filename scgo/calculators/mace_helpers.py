"""MACE machine learning potential wrapper for cluster optimization.

This module provides a simplified wrapper around the MACE-MP pretrained models,
handling device selection and initialization for seamless integration with
global optimization workflows.
"""

from __future__ import annotations

from enum import StrEnum

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import mace_mp

from scgo.utils.logging import get_logger
from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed


class MaceUrls(StrEnum):
    """Checkpoint download URLs for MACE models."""

    mace_mp_small = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model"
    mace_mpa_medium = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    mace_off_small = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true"
    mace_matpes_0 = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model"


class MACE(Calculator):
    """A wrapper for the MACE-MP-0 calculator for global optimization.

    This class simplifies the initialization of a MACE calculator, handling
    automatic device selection (CUDA/MPS/CPU) and model loading. It serves as a
    standard ASE-compliant calculator, making it easy to integrate MACE into
    global optimization workflows.

    Attributes:
        implemented_properties (list): A list of properties that this calculator
                                     can compute, which are "energy" and "forces".

    """

    implemented_properties: list[str] = ["energy", "forces"]

    def __init__(
        self,
        model_name: str = "mace_matpes_0",
        device: str | None = None,
        default_dtype: str = "float64",
        **kwargs,
    ):
        """Initializes the MACE calculator.

        Args:
            model_name: The name of the pretrained MACE model to use. Can be:
                - A MaceUrls enum member name (e.g., "mace_matpes_0")
                - A direct URL to a model file
                - A standard mace_mp model name (e.g., "small", "medium", "large")
                Defaults to "mace_matpes_0" (r2scan variant).
            device: The computing device to run the model on. If None, it will
                auto-detect CUDA or MPS (for Apple Silicon) and fall back to CPU
                if neither is available. Defaults to None.
            default_dtype: The default floating-point precision for calculations.
                "float64" is recommended for stable optimizations.
                Defaults to "float64".
            **kwargs: Additional keyword arguments passed to the base ASE
                Calculator class.
        """
        ensure_mace_uma_not_both_installed()
        if device is None:
            if torch.cuda.is_available():
                selected_device = "cuda"
            elif torch.backends.mps.is_available():
                selected_device = "mps"
            else:
                selected_device = "cpu"
        else:
            selected_device = device

        # Resolve model name to URL if it's a MaceUrls enum member
        if hasattr(MaceUrls, model_name):
            model_selector = getattr(MaceUrls, model_name)
        else:
            model_selector = model_name

        name = f"MACE-{model_name}"
        # Pass the constructed name to the parent class initializer.
        super().__init__(name=name, **kwargs)

        logger = get_logger(__name__)
        logger.info(
            f'Initializing MACE calculator ("{model_name}" model) on device: "{selected_device}"',
        )

        # The mace_mp function from mace.calculators automatically handles
        # downloading and loading the specified pretrained MACE model.
        # It returns a fully functional ASE calculator instance.
        self._mace_calc = mace_mp(
            model=model_selector,
            device=selected_device,
            default_dtype=default_dtype,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):
        """Performs the MACE calculation by delegating to the wrapped calculator.

        This method is called by ASE algorithms. It sets up the calculation,
        calls the underlying MACE calculator, and stores the results.

        Args:
            atoms: The Atoms object to perform the calculation on. If None, the
                calculator's internal atoms object is used.
            properties: A list of properties to calculate (e.g., ["energy", "forces"]).
                If None, defaults to `self.implemented_properties`.
            system_changes: A list of strings specifying what has changed since
                the last calculation. See ASE documentation for details.
        """
        if properties is None:
            properties = self.implemented_properties

        # Call the base class's calculate method to handle setup
        super().calculate(atoms, properties, system_changes)

        # Delegate the actual computation to the wrapped MACE calculator instance
        self._mace_calc.calculate(
            atoms=self.atoms,
            properties=properties,
            system_changes=system_changes,
        )
        self.results = self._mace_calc.results
