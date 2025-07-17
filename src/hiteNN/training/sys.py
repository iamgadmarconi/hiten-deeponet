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
            sys_obj = System.from_mu(mu)

        point_obj = sys_obj.get_libration_point(point)
        cm = point_obj.get_center_manifold(degree)

        _CACHE[key] = (sys_obj, point_obj, cm)

    return _CACHE[key]
