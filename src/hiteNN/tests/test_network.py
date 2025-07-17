import os
import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from hiten import GenericOrbit

from hiteNN.training.data import _create_dataset
from hiteNN.training.sys import _create_system

MODEL_PATH = os.path.join("outputs", "network", "deepo.0.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_deeponet() -> DeepONetArch:
    """Re-create the architecture defined in src/hiteNN/network/network.py."""
    branch_net = FullyConnectedArch(
        input_keys=[Key("cm", 6)],
        output_keys=[Key("branch", 128)],
    )
    trunk_net = FullyConnectedArch(
        input_keys=[
            Key("x", 1),
            Key("energy", 1),
            Key("mu", 1),
            Key("lag_idx", 1),
        ],
        output_keys=[Key("trunk", 128)],
    )
    deeponet = DeepONetArch(
        output_keys=[Key("u")],
        branch_net=branch_net,
        trunk_net=trunk_net,
    )
    return deeponet


def load_trained_model(path: str = MODEL_PATH) -> torch.nn.Module:
    model = build_deeponet()
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def deeponet_predict(model: torch.nn.Module, cm_state: np.ndarray, energy: float, mu: float = 0.01215, lag_idx: int = 1) -> np.ndarray:
    """Predict full 6-D synodic state using DeepONet.

    Parameters
    ----------
    model : loaded Deeponet model
    cm_state : (6,) centre-manifold state
    energy : scalar energy value

    Returns
    -------
    np.ndarray
        (6,) predicted synodic state values corresponding to q1,q2,q3,p1,p2,p3 order.
    """
    # Branch input = cm_state (6,)
    cm = cm_state.astype(np.float32)
    cm_rep = np.repeat(cm[None, :], repeats=6, axis=0)  # (6,6)

    # trunk input x: indices 0..5
    x = np.arange(6, dtype=np.float32)[:, None]              # (6,1)
    e_arr = np.full((6, 1), energy, dtype=np.float32)
    mu_arr = np.full((6, 1), mu, dtype=np.float32)
    lag_arr = np.full((6, 1), lag_idx, dtype=np.float32)

    # Convert to torch
    cm_tensor   = torch.tensor(cm_rep, device=DEVICE)
    x_tensor    = torch.tensor(x, device=DEVICE)
    e_tensor    = torch.tensor(e_arr, device=DEVICE)
    mu_tensor   = torch.tensor(mu_arr, device=DEVICE)
    lag_tensor  = torch.tensor(lag_arr, device=DEVICE)

    with torch.no_grad():
        out = model({
            "cm": cm_tensor,
            "x": x_tensor,
            "energy": e_tensor,
            "mu": mu_tensor,
            "lag_idx": lag_tensor,
        })
    # Handle different output key variants
    if isinstance(out, dict):
        if Key("u", out.shape[1] if hasattr(out, 'shape') else 1) in out:
            u = out[Key("u", out.shape[1] if hasattr(out, 'shape') else 1)]
        elif Key("u") in out:
            u = out[Key("u")]
        elif "u" in out:
            u = out["u"]
        else:
            # fallback to first value
            u = next(iter(out.values()))
    else:
        u = out
    return u.cpu().numpy().flatten()


def sample_initial_condition(point: int | None = None, degree: int = 8, mu: float = 0.01215, n_samples: int = 100) -> Tuple[np.ndarray, float, float, int, np.ndarray]:
    """Generate one random IC and return cm_state, energy, mu, lag_idx, synodic_truth.
    
    If point is None, randomly selects from L1, L2, or L3 libration points.
    Creates n_samples initial conditions and randomly selects one.
    """
    # If no point specified, randomly choose from L1, L2 (points 1, 2)
    if point is None:
        point = np.random.randint(1, 3)  # Random choice of 1, 2
    
    # Use a random seed to ensure different initial conditions each time
    random_seed = np.random.randint(0, 100000)
    cm, en, syn = _create_dataset(point, degree, n_samples=n_samples, amplitude=0.05, mu=mu, seed=random_seed)
    idx = np.random.randint(0, len(cm))
    return cm[idx], float(en[idx]), mu, point, syn[idx]


def main():
    print("Loading trained DeepONet from", MODEL_PATH)
    model = load_trained_model()

    # Sample 6 random initial conditions (1 original + 5 additional)
    n_samples = 6
    print(f"Sampling {n_samples} random initial conditions via hiten ...")
    
    all_cm_states = []
    all_energies = []
    all_mu_vals = []
    all_lag_idxs = []
    all_syn_truths = []
    all_syn_preds = []
    hiten_times = []
    deeponet_times = []
    errors = []
    
    for i in range(n_samples):
        print(f"\nSample {i+1}/{n_samples}:")
        
        # Time hiten computation
        start_time = time.time()
        cm_state, energy, mu_val, lag_idx, syn_truth = sample_initial_condition(n_samples=1)
        hiten_time = time.time() - start_time
        
        print("Selected Libration Point: L" + str(lag_idx))
        print("Centre-manifold state (cm)", cm_state)
        print("Energy", energy)
        print(f"Hiten computation time: {hiten_time:.6f}s")
        
        # Time DeepONet prediction
        start_time = time.time()
        syn_pred = deeponet_predict(model, cm_state, energy, mu_val, lag_idx)
        deeponet_time = time.time() - start_time
        
        print(f"DeepONet computation time: {deeponet_time:.6f}s")
        print(f"Speedup factor: {hiten_time/deeponet_time:.1f}x")
        
        # Calculate error
        error = np.abs(syn_pred - syn_truth)
        l2_error = np.linalg.norm(error)
        
        print("Ground-truth synodic state   :", syn_truth)
        print("DeepONet predicted synodic   :", syn_pred)
        print("Absolute error per component :", error)
        print("L2 norm of error             :", l2_error)
        
        # Store results
        all_cm_states.append(cm_state)
        all_energies.append(energy)
        all_mu_vals.append(mu_val)
        all_lag_idxs.append(lag_idx)
        all_syn_truths.append(syn_truth)
        all_syn_preds.append(syn_pred)
        hiten_times.append(hiten_time)
        deeponet_times.append(deeponet_time)
        errors.append(error)
    
    # Convert to numpy arrays for analysis
    all_syn_truths = np.array(all_syn_truths)
    all_syn_preds = np.array(all_syn_preds)
    errors = np.array(errors)
    hiten_times = np.array(hiten_times)
    deeponet_times = np.array(deeponet_times)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Component-wise error plots
    component_names = ['q1', 'q2', 'q3', 'p1', 'p2', 'p3']
    for i in range(6):
        row = i // 3
        col = i % 3
        axes[row, col].bar(range(n_samples), errors[:, i])
        axes[row, col].set_title(f'Absolute Error in {component_names[i]}')
        axes[row, col].set_xlabel('Sample Index')
        axes[row, col].set_ylabel('Absolute Error')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # L2 norm errors plot
    l2_errors = np.linalg.norm(errors, axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_samples), l2_errors)
    plt.title('L2 Norm of Prediction Errors')
    plt.xlabel('Sample Index')
    plt.ylabel('L2 Error')
    plt.grid(True, alpha=0.3)
    for i, err in enumerate(l2_errors):
        plt.text(i, err + max(l2_errors)*0.01, f'{err:.2e}', ha='center', va='bottom')
    plt.savefig('l2_errors.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Timing comparison plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    x = np.arange(n_samples)
    width = 0.35
    plt.bar(x - width/2, hiten_times * 1000, width, label='Hiten', alpha=0.8)
    plt.bar(x + width/2, deeponet_times * 1000, width, label='DeepONet', alpha=0.8)
    plt.title('Computation Time Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    speedups = hiten_times / deeponet_times
    plt.bar(range(n_samples), speedups)
    plt.title('DeepONet Speedup Factor')
    plt.xlabel('Sample Index')
    plt.ylabel('Speedup (x times faster)')
    plt.grid(True, alpha=0.3)
    for i, speedup in enumerate(speedups):
        plt.text(i, speedup + max(speedups)*0.01, f'{speedup:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('timing_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of samples: {n_samples}")
    print(f"Average L2 error: {np.mean(l2_errors):.2e} ± {np.std(l2_errors):.2e}")
    print(f"Max L2 error: {np.max(l2_errors):.2e}")
    print(f"Min L2 error: {np.min(l2_errors):.2e}")
    print(f"\nAverage Hiten time: {np.mean(hiten_times)*1000:.2f} ± {np.std(hiten_times)*1000:.2f} ms")
    print(f"Average DeepONet time: {np.mean(deeponet_times)*1000:.2f} ± {np.std(deeponet_times)*1000:.2f} ms")
    print(f"Average speedup: {np.mean(speedups):.1f}x ± {np.std(speedups):.1f}x")
    
    # Plot orbits for first sample as example
    print(f"\nPlotting orbits for first sample (L{all_lag_idxs[0]})...")
    _, point_obj, _ = _create_system(all_lag_idxs[0], 8, mu=all_mu_vals[0])

    real_orbit = GenericOrbit(point_obj, all_syn_truths[0])
    real_orbit.propagate()
    real_orbit.plot()

    pred_orbit = GenericOrbit(point_obj, all_syn_preds[0])
    pred_orbit.propagate()
    pred_orbit.plot()


if __name__ == "__main__":
    main()
