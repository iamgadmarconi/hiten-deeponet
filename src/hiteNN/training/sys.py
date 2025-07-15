from hiten import CenterManifold, LibrationPoint, System
from hiten.system.body import Body

# Internal cache keyed by (mu, point, degree)
_CACHE: dict[tuple[float | None, int, int], tuple[System, LibrationPoint, CenterManifold]] = {}


def _create_system(
    point: int,
    degree: int,
    mu: float | None = None,
) -> tuple[System, LibrationPoint, CenterManifold]:
    """Create or retrieve a CR3BP *System* object for given parameters.

    Parameters
    ----------
    point : 1 | 2 | 3 | 4 | 5
        Libration point index.
    degree : int
        Centre-manifold polynomial degree.
    mu : float | None, optional
        Mass ratio μ = m2 / (m1 + m2). If *None* defaults to Earth-Moon value
        via *System.from_bodies('earth','moon')*.
    """
    key = (mu, point, degree)
    if key not in _CACHE:
        if mu is None:
            sys_obj = System.from_bodies('earth', 'moon')
        else:
            # ------------------------------------------------------------------
            # Construct an artificial CR3BP system with the requested mu
            # ------------------------------------------------------------------
            # Choose canonical units: total mass = 1, distance = 1 [unit distance]
            m2 = float(mu)
            m1 = 1.0 - m2

            primary = Body("P1", mass=m1, radius=1.0)
            secondary = Body("P2", mass=m2, radius=1.0, _parent_input=primary)

            sys_obj = System(primary, secondary, distance=1.0)

        point_obj = sys_obj.get_libration_point(point)
        cm = point_obj.get_center_manifold(degree)

        _CACHE[key] = (sys_obj, point_obj, cm)

    return _CACHE[key]
