"""Energy metrics for adsorption (optional objectives)."""


def adsorption_energy(
    e_adsorbate_slab: float,
    e_slab: float,
    e_isolated_cluster: float,
) -> float:
    """Classical adsorption energy: E(ads+slab) - E(slab) - E(cluster).

    Negative values usually indicate binding.

    Args:
        e_adsorbate_slab: Total energy of the relaxed adsorbate+slab system.
        e_slab: Total energy of the bare relaxed slab (same cell/supercell).
        e_isolated_cluster: Total energy of the isolated cluster.

    Returns:
        Adsorption energy in the same units as inputs (typically eV).
    """
    return e_adsorbate_slab - e_slab - e_isolated_cluster
