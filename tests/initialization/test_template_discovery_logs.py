import logging

from ase.cluster import Icosahedron

from scgo.initialization import templates


def test_grow_template_logs_discovery_failure(monkeypatch, caplog, rng):
    """When template growth cannot generate facet positions, a concise "discovery"
    debug message should be logged (not a per-structure fallback message).
    """

    # Force _generate_batch_positions_on_convex_hull to return no candidates
    def fake_generate_batch_positions_on_convex_hull(*args, **kwargs):
        return []

    monkeypatch.setattr(
        templates,
        "_generate_batch_positions_on_convex_hull",
        fake_generate_batch_positions_on_convex_hull,
    )

    base = Icosahedron("Pt", 2)
    base.center()
    base.set_cell([30.0, 30.0, 30.0])
    comp = ["Pt"] * 20

    caplog.set_level(logging.DEBUG)
    result = templates.grow_template_via_facets(
        base,
        comp,
        placement_radius_scaling=templates.PLACEMENT_RADIUS_SCALING_DEFAULT,
        cell_side=30.0,
        rng=rng,
        min_distance_factor=templates.MIN_DISTANCE_FACTOR_DEFAULT,
        connectivity_factor=templates.CONNECTIVITY_FACTOR,
    )

    assert result is None
    # Ensure the concise discovery clarification is present in logs
    assert any("discovery failure" in r.getMessage() for r in caplog.records)
