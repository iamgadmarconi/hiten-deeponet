import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import physicsnemo.sym
from physicsnemo.sym.dataset.discrete import DictGridDataset
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint.continuous import DeepONetConstraint
from physicsnemo.sym.domain.validator.discrete import GridValidator
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

# Project
from hiteNN.training.data import load_dataset


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # ─────────────────────────────── 1. Load & reshape data ──────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "..", "training", "_data", "training_data.npz")
    cm_states, energies, mus, lag_idxs, synodic_states = load_dataset(data_path)

    # ───────────────────────── Build inputs ──────────────────────────
    # Branch input: 6-D centre-manifold state only  → (N,6)
    branch_in = cm_states.astype(np.float32)
    N = branch_in.shape[0]

    # replicate each sample 6× to predict every synodic component separately
    trunk_idx  = np.tile(np.arange(6, dtype=np.float32), N)[:, None]       # (N*6,1)
    branch_rep = np.repeat(branch_in, 6, axis=0)                           # (N*6,6)
    energy_rep = np.repeat(energies[:, None], 6, axis=1).reshape(-1, 1).astype(np.float32)  # (N*6,1)
    mu_rep     = np.repeat(mus[:, None],      6, axis=1).reshape(-1, 1).astype(np.float32)  # (N*6,1)
    lag_rep    = np.repeat(lag_idxs[:, None], 6, axis=1).reshape(-1, 1).astype(np.float32)  # (N*6,1)
    target_u   = synodic_states.reshape(-1, 1).astype(np.float32)          # (N*6,1)

    # shuffle & split 90/10
    rng = np.random.default_rng(42)
    perm = rng.permutation(branch_rep.shape[0])
    val_size = int(0.1 * len(perm))
    train_idx, val_idx = perm[val_size:], perm[:val_size]

    a_train = branch_rep[train_idx]
    a_val   = branch_rep[val_idx]

    x_train = trunk_idx[train_idx]
    x_val   = trunk_idx[val_idx]

    e_train = energy_rep[train_idx]
    e_val   = energy_rep[val_idx]

    mu_train = mu_rep[train_idx]
    mu_val   = mu_rep[val_idx]

    l_train = lag_rep[train_idx]
    l_val   = lag_rep[val_idx]

    u_train = target_u[train_idx]
    u_val   = target_u[val_idx]

    # ─────────────────────────────── 2. Build DeepONet ──────────────────────────────────
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
    nodes = [deeponet.make_node("deepo")]

    # ─────────────────────────────── 3. Domain & constraints ────────────────────────────
    domain = Domain()

    train_constraint = DeepONetConstraint.from_numpy(
        nodes      = nodes,
        invar      = {
            "cm"     : a_train,
            "x"      : x_train,
            "energy" : e_train,
            "mu"     : mu_train,
            "lag_idx": l_train,
        },
        outvar     = {"u": u_train},
        batch_size = cfg.batch_size.train,
    )
    domain.add_constraint(train_constraint, "train")

    # validators (single GridValidator with full val set)
    val_dataset = DictGridDataset(
        {"cm": a_val, "x": x_val, "energy": e_val, "mu": mu_val, "lag_idx": l_val},
        {"u": u_val},
    )
    validator   = GridValidator(nodes=nodes, dataset=val_dataset, plotter=None)
    domain.add_validator(validator, "val")

    # ─────────────────────────────── 4. Solver ──────────────────────────────────────────
    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    run()