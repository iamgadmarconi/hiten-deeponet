import os
import sys
from typing import Tuple

import numpy as np
import torch

# Physics-NeMo
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

# Project utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))  # add project src to path
from hiteNN.training.data import _create_dataset  # type: ignore
from hiteNN.training.sys import _create_system
from hiten import GenericOrbit


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


def sample_initial_condition(point: int = 1, degree: int = 5, mu: float = 0.01215) -> Tuple[np.ndarray, float, float, int, np.ndarray]:
    """Generate one random IC and return cm_state, energy, mu, lag_idx, synodic_truth."""
    cm, en, syn = _create_dataset(point, degree, n_samples=1, amplitude=0.05, mu=mu)
    return cm[0], float(en[0]), mu, point, syn[0]


def main():
    print("Loading trained DeepONet from", MODEL_PATH)
    model = load_trained_model()

    print("Sampling one random initial condition via hiten ...")
    cm_state, energy, mu_val, lag_idx, syn_truth = sample_initial_condition()

    print("Centre-manifold state (cm)", cm_state)
    print("Energy", energy)

    syn_pred = deeponet_predict(model, cm_state, energy, mu_val, lag_idx)

    print("Ground-truth synodic state   :", syn_truth)
    print("DeepONet predicted synodic   :", syn_pred)

    error = np.abs(syn_pred - syn_truth)
    print("Absolute error per component :", error)
    print("L2 norm of error             :", np.linalg.norm(error))

    _, point_obj, _ = _create_system(lag_idx, 5, mu=mu_val)

    real_orbit = GenericOrbit(point_obj, syn_truth)
    real_orbit.propagate()
    real_orbit.plot()

    pred_orbit = GenericOrbit(point_obj, syn_pred)
    pred_orbit.propagate()
    pred_orbit.plot()


if __name__ == "__main__":
    main()
