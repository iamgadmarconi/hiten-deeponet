from typing import List, Sequence, Tuple

import numpy as np
from hiten.algorithms.poincare.config import _get_section_config
from hiten.algorithms.poincare.map import _solve_missing_coord

from .sys import _create_system


def _create_dataset(point: int, degree: int, n_samples: int,
                   energy_range: Tuple[float, float] = (0.0, 1.0),
                   section_coord: str = 'q3',
                   amplitude: float = 1.0,
                   seed: int = 42,
                   mu: float | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data for deep operator network mapping centre-manifold
    states (6-D with appropriate zeros) and energy → full synodic ICs.

    Returns
    -------
    cm_states : np.ndarray, shape (n_samples, 6)
        Centre-manifold states in synodic coordinates. Hyperbolic directions
        (q1,p1) are identically zero, and the chosen section coordinate is
        zero by construction.
    energies : np.ndarray, shape (n_samples,)
        Hamiltonian energy levels used for each sample (measured above the
        libration-point energy).
    synodic_states : np.ndarray, shape (n_samples, 6)
        Target synodic initial conditions (identical to *cm_states* but kept
        separate for clarity / potential future transforms).
    """
    np.random.seed(seed)

    # System objects (optionally for custom mass ratio)
    _, _, center_manifold = _create_system(point, degree, mu=mu)

    # Configuration for the chosen section
    cfg = _get_section_config(section_coord)

    # Random sampling in the 2-D plane of free coordinates
    plane_vals_1 = np.random.uniform(-amplitude, amplitude, n_samples)
    plane_vals_2 = np.random.uniform(-amplitude, amplitude, n_samples)

    # Sample energies uniformly within the requested range
    energies_full = np.random.uniform(energy_range[0], energy_range[1], n_samples)

    # Output arrays
    cm_states = np.zeros((n_samples, 6))
    synodic_states = np.zeros((n_samples, 6))
    energies_out = np.zeros(n_samples)

    valid = 0
    idx = 0
    max_attempts = n_samples * 3  # safety guard
    while valid < n_samples and idx < max_attempts:
        # Generate extra samples if we exhausted pre-sampled ones
        if idx >= len(plane_vals_1):
            add = max_attempts - len(plane_vals_1)
            plane_vals_1 = np.concatenate([plane_vals_1, np.random.uniform(-amplitude, amplitude, add)])
            plane_vals_2 = np.concatenate([plane_vals_2, np.random.uniform(-amplitude, amplitude, add)])
            energies_full = np.concatenate([energies_full, np.random.uniform(energy_range[0], energy_range[1], add)])

        plane = np.array([plane_vals_1[idx], plane_vals_2[idx]], dtype=float)
        energy = float(energies_full[idx])

        try:
            # 1. Solve for the missing CM coordinate (q/p2/3) on the centre manifold.
            poly_cm_real = center_manifold.compute()
            var_to_solve = cfg.missing_coord  # e.g., "p3" for q3 section

            known = {cfg.section_coord: 0.0,
                     cfg.plane_coords[0]: float(plane[0]),
                     cfg.plane_coords[1]: float(plane[1])}

            missing_val = _solve_missing_coord(var_to_solve, known, energy, poly_cm_real, center_manifold._clmo)

            if missing_val is None:
                raise RuntimeError("root-finding failed")

            # Assemble 6-D centre-manifold state in synodic variables but *before* Lie back-transform
            # In real CM coordinates: [q1, q2, q3, p1, p2, p3] with q1=p1=0 and section_coord = 0
            cm_state = np.zeros(6, dtype=float)
            cm_state[0] = 0.0  # q1 hyperbolic
            cm_state[3] = 0.0  # p1 hyperbolic

            # Fill plane coordinates
            var_indices = {"q1":0,"q2":1,"q3":2,"p1":3,"p2":4,"p3":5}
            cm_state[var_indices[cfg.plane_coords[0]]] = plane[0]
            cm_state[var_indices[cfg.plane_coords[1]]] = plane[1]
            # Section coordinate already zero by initialization
            cm_state[var_indices[var_to_solve]] = missing_val

            # 2. Compute corresponding full synodic initial conditions
            synodic_ic = center_manifold.ic(plane, energy, section_coord)
            syn_real = np.real_if_close(np.asarray(synodic_ic, dtype=float))

            # Store
            cm_states[valid] = cm_state
            synodic_states[valid] = syn_real
            energies_out[valid] = energy
            valid += 1
        except RuntimeError:
            pass  # skip invalid sample
        idx += 1

    if valid < n_samples:
        print(f"Warning: only generated {valid} valid samples out of {n_samples}")
        cm_states = cm_states[:valid]
        synodic_states = synodic_states[:valid]
        energies_out = energies_out[:valid]

    return cm_states, energies_out, synodic_states


def build_parameter_sweep_dataset(
    mus: Sequence[float],
    l_points: Sequence[int],
    degree: int,
    n_samples: int,
    energy_range: Tuple[float, float] = (0.0, 1.0),
    section_coord: str = "q3",
    amplitude: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single combined dataset spanning multiple mass ratios and L-points.

    Returns
    -------
    cm_states : (N,6)
    energies  : (N,)
    mus       : (N,)
    lag_idxs  : (N,)
    syn_states: (N,6)
    """
    rng = np.random.default_rng(seed)

    cm_list: List[np.ndarray] = []
    e_list: List[np.ndarray] = []
    mu_list: List[np.ndarray] = []
    lag_list: List[np.ndarray] = []
    syn_list: List[np.ndarray] = []

    for mu_val in mus:
        for point in l_points:
            # keep deterministic but different seeds per combo
            local_seed = int(rng.integers(0, 1e9))

            cm, en, syn = _create_dataset(
                point=point,
                degree=degree,
                n_samples=n_samples,
                energy_range=energy_range,
                section_coord=section_coord,
                amplitude=amplitude,
                seed=local_seed,
                mu=mu_val,
            )

            cm_list.append(cm)
            e_list.append(en)
            syn_list.append(syn)
            mu_list.append(np.full_like(en, fill_value=mu_val, dtype=float))
            lag_list.append(np.full_like(en, fill_value=point, dtype=float))

    return (
        np.concatenate(cm_list, axis=0),
        np.concatenate(e_list, axis=0),
        np.concatenate(mu_list, axis=0),
        np.concatenate(lag_list, axis=0),
        np.concatenate(syn_list, axis=0),
    )


def create_training_validation_split(cm_states: np.ndarray,
                                   energies: np.ndarray,
                                   synodic_states: np.ndarray,
                                   validation_ratio: float = 0.2,
                                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into training and validation sets while keeping all arrays aligned."""
    np.random.seed(seed)
    n_samples = len(cm_states)
    n_val = int(n_samples * validation_ratio)
    idx = np.random.permutation(n_samples)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return (
        cm_states[train_idx], energies[train_idx], synodic_states[train_idx],
        cm_states[val_idx], energies[val_idx], synodic_states[val_idx]
    )


def create_multi_energy_dataset(point: int, degree: int, n_samples_per_energy: int,
                               energy_levels: List[float],
                               section_coord: str = 'q3',
                               amplitude: float = 0.05,
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dataset for specific discrete energy levels."""
    cm_list, e_list, syn_list = [], [], []
    for energy in energy_levels:
        cm, en, syn = _create_dataset(point, degree, n_samples_per_energy,
                                      (energy, energy), section_coord,
                                      amplitude, seed)
        cm_list.append(cm)
        e_list.append(en)
        syn_list.append(syn)
        seed += 1  # update seed so each energy level differs
    return (
        np.concatenate(cm_list, axis=0),
        np.concatenate(e_list, axis=0),
        np.concatenate(syn_list, axis=0),
    )


def save_dataset(
    cm_states: np.ndarray,
    energies: np.ndarray,
    mus: np.ndarray,
    lag_idxs: np.ndarray,
    synodic_states: np.ndarray,
    path: str = "training_data.npz",
) -> None:
    """Save extended dataset including parameters to a compressed .npz file."""
    np.savez(
        path,
        center_manifold_states=cm_states,
        energies=energies,
        mus=mus,
        lag_idxs=lag_idxs,
        synodic_states=synodic_states,
    )
    print(f"Dataset saved to {path}")


def load_dataset(
    path: str = "training_data.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset with parameters (cm_states, energies, mus, lag_idxs, synodic_states)."""
    data = np.load(path)
    return (
        data['center_manifold_states'],
        data['energies'],
        data['mus'],
        data['lag_idxs'],
        data['synodic_states'],
    )


def main():
    # Configuration
    degree = 8
    n_samples = 2000  # per (mu, point) combination
    energy_range = (0.0, 1.0)
    amplitude = 0.8
    section_coord = 'q3'
    seed = 42
    
    # Parameters to sweep
    mus = [0.01215, 0.0009537]  # Earth-Moon, Sun-Jupiter for example
    l_points = [1, 2]

    print("Generating dataset with:")
    print(f"  mass ratios      : {mus}")
    print(f"  Libration points : {l_points}")
    print(f"  Samples per combo: {n_samples}")

    cm_states, energies, mus_arr, lag_idxs, synodic_states = build_parameter_sweep_dataset(
        mus=mus,
        l_points=l_points,
        degree=degree,
        n_samples=n_samples,
        energy_range=energy_range,
        section_coord=section_coord,
        amplitude=amplitude,
        seed=seed,
    )

    print(f"Total samples generated: {len(cm_states)}")
    print(f"CM state shape: {cm_states.shape}")
    print(f"Energy array shape: {energies.shape}")
    print(f"Mu array shape    : {mus_arr.shape}")
    print(f"L-point array shape: {lag_idxs.shape}")
    print(f"Synodic state shape: {synodic_states.shape}")
    
    # Save data
    path = r"src\hiteNN\training\_data\training_data.npz"
    save_dataset(cm_states, energies, mus_arr, lag_idxs, synodic_states, path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
