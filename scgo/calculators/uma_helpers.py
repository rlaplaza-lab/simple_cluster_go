"""UMA (Universal Material Approximation) calculator via fairchem-core."""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from scgo.utils.logging import get_logger
from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed

_MISSING_FAIRCHEM_MSG = (
    "fairchem-core is not installed. Install with: pip install 'scgo[uma]' "
    "(do not combine with the [mace] extra in the same environment)."
)


class UMA(Calculator):
    """ASE calculator wrapping FAIRChem UMA checkpoints (fairchem-core).

    Parameters mirror common SCGO ``calculator_kwargs`` patterns: ``model_name``
    is a fairchem pretrained name or path; ``task_name`` selects the UMA task
    (e.g. ``\"omat\"``, ``\"oc20\"``). Device defaults to CUDA when available,
    else CPU (fairchem expects ``\"cuda\"`` or ``\"cpu\"``).
    """

    def __init__(
        self,
        model_name: str = "uma-s-1",
        task_name: str | None = "omat",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        ensure_mace_uma_not_both_installed()
        try:
            import torch
            from fairchem.core import FAIRChemCalculator
        except ImportError as e:
            raise ImportError(_MISSING_FAIRCHEM_MSG) from e

        if device is None:
            dev: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            d = str(device).lower()
            dev = "cuda" if "cuda" in d else "cpu"

        name = f"UMA-{model_name}"
        super().__init__(name=name, **kwargs)

        logger = get_logger(__name__)
        logger.info(
            'Initializing UMA calculator ("%s") on device: "%s"', model_name, dev
        )

        self._inner = FAIRChemCalculator.from_model_checkpoint(
            model_name,
            task_name=task_name,
            device=dev,
        )
        self.implemented_properties = list(self._inner.implemented_properties)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        self._inner.calculate(
            atoms=self.atoms,
            properties=properties,
            system_changes=system_changes,
        )
        self.results = self._inner.results
