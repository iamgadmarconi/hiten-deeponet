import sys

import numpy as np

sys.path.append('src')
from hiteNN.training.data import build_parameter_sweep_dataset, save_dataset


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
    save_dataset(cm_states, energies, mus_arr, lag_idxs, synodic_states, "training_data.npz")
    print("Saved to training_data.npz")

if __name__ == "__main__":
    main()
