"""Contract tests for ``coerce_ts_params_to_runner_kwargs`` (WP3: P1 + P2).

Strict-key rejection mirrors GO allowlist behavior; ``tag_ts_in_db`` must flow
as a boolean with a ``True`` default instead of being silently dropped.
"""

from __future__ import annotations

import pytest

from scgo.exceptions import SCGOValidationError
from scgo.param_presets import get_ts_search_params
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs


def _gas_ts_params() -> dict:
    ts = get_ts_search_params(calculator="EMT", system_type="gas_cluster")
    # EMT has no TorchSim backend; presets default use_torchsim=True.
    ts["use_torchsim"] = False
    return ts


def test_coerce_full_preset_output_passes_strict_allowlist():
    """The canonical preset dict never trips the strict-key rejection."""
    kwargs = coerce_ts_params_to_runner_kwargs(
        _gas_ts_params(), system_type="gas_cluster"
    )
    assert kwargs["params"]["calculator"] == "EMT"


@pytest.mark.parametrize(
    "typo_key", ["neb_fmx", "tag_ts_in_db_typo", "spring_constants"]
)
def test_coerce_rejects_unknown_keys_listing_them(typo_key):
    ts = _gas_ts_params()
    ts[typo_key] = 1.0
    with pytest.raises(SCGOValidationError) as excinfo:
        coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    message = str(excinfo.value)
    assert typo_key in message
    assert "Unexpected ts_params keys" in message


def test_coerce_error_lists_expected_subset():
    ts = _gas_ts_params()
    ts["not_a_real_knob"] = True
    with pytest.raises(SCGOValidationError, match="neb_fmax"):
        coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")


def test_tag_ts_in_db_defaults_true_and_flows_false():
    kwargs = coerce_ts_params_to_runner_kwargs(
        _gas_ts_params(), system_type="gas_cluster"
    )
    assert kwargs["tag_ts_in_db"] is True

    ts = _gas_ts_params()
    ts["tag_ts_in_db"] = False
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["tag_ts_in_db"] is False


def test_binding_penetration_tolerance_still_forwarded_to_gate():
    """The kwarg feeds the post-NEB gate even though the cfg field is gone."""
    ts = _gas_ts_params()
    ts["binding_penetration_tolerance_a"] = 0.3
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["binding_penetration_tolerance_a"] == pytest.approx(0.3)
