"""
Shared two-phase SOM training (Gibson et al. 2017 style) for all three SOMs.

The defaults reproduce the published call sequence exactly (same MiniSom
constructor arguments, initialization, phase order and RNG use), so a
published run is reproduced bit-for-bit with the same seed. Every training
hyperparameter is a CLI argument so the configuration chosen by the
robustness experiment (som_analysis/robustness_experiment.py) can be passed
to the training scripts and to run_pipeline.sh.
"""

import json

import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA


def add_training_args(parser, *, n1, n2, lr1, lr2, topology=True):
    """Add the hyperparameter arguments (defaults = published values)."""
    g = parser.add_argument_group("training hyperparameters (defaults = published)")
    g.add_argument(
        "--length",
        type=float,
        default=1.0,
        help="Multiplier on n1 and n2 (training length; default: 1).",
    )
    g.add_argument(
        "--lr1", type=float, default=lr1, help=f"Phase-1 learning rate ({lr1})."
    )
    g.add_argument(
        "--lr2", type=float, default=lr2, help=f"Phase-2 learning rate ({lr2})."
    )
    g.add_argument(
        "--sigma1",
        type=float,
        default=None,
        help="Phase-1 initial radius (default: grid diagonal), decaying to 1.",
    )
    g.add_argument(
        "--sigma2",
        type=float,
        default=1.0,
        help="Phase-2 radius, held constant (default: 1).",
    )
    g.add_argument(
        "--neighborhood",
        choices=["gaussian", "bubble", "mexican_hat", "triangle"],
        default="gaussian",
        help="Neighbourhood function (default: gaussian).",
    )
    g.add_argument(
        "--mode",
        choices=["random", "offline"],
        default="random",
        help="'random' = sequential training on randomly drawn samples "
        "(MiniSom train_random; published). 'offline' = true batch SOM "
        "(train_batch_offline, constant learning rate; n1/n2 are then full "
        "passes over the data, e.g. --n1 30 --n2 20 --lr1 1 --lr2 1).",
    )
    if topology:
        g.add_argument(
            "--topology",
            choices=["rectangular", "hexagonal"],
            default="rectangular",
            help="SOM grid topology (default: rectangular).",
        )
    return parser


def training_config(args, xdim, ydim):
    """The effective training configuration as a dict (saved with the SOM)."""
    sig1 = args.sigma1 if args.sigma1 is not None else float(np.sqrt(xdim**2 + ydim**2))
    return dict(
        xdim=xdim,
        ydim=ydim,
        seed=args.seed,
        init=args.init,
        mode=args.mode,
        n1=int(round(args.n1 * args.length)),
        n2=int(round(args.n2 * args.length)),
        length=args.length,
        lr1=args.lr1,
        lr2=args.lr2,
        sigma1=sig1,
        sigma2=args.sigma2,
        neighborhood=args.neighborhood,
        topology=getattr(args, "topology", "rectangular"),
    )


def train_two_phase(X, xdim, ydim, args):
    """Train a SOM in two phases; returns (som, config dict)."""
    cfg = training_config(args, xdim, ydim)
    som = MiniSom(
        xdim,
        ydim,
        input_len=X.shape[1],
        sigma=cfg["sigma1"],
        learning_rate=cfg["lr1"],
        decay_function="linear_decay_to_zero",
        sigma_decay_function="linear_decay_to_one",
        neighborhood_function=cfg["neighborhood"],
        topology=cfg["topology"],
        random_seed=cfg["seed"],
    )

    if cfg["init"] == "pca":
        pca = PCA(n_components=2, random_state=cfg["seed"])
        pca.fit(X)
        w = np.zeros((xdim, ydim, X.shape[1]))
        for i, c1 in enumerate(np.linspace(-1, 1, xdim)):
            for j, c2 in enumerate(np.linspace(-1, 1, ydim)):
                w[i, j] = pca.mean_ + c1 * pca.components_[0] + c2 * pca.components_[1]
        som._weights = w
        print("Initialized weights via PCA.")
    else:
        som.random_weights_init(X)

    if cfg["mode"] == "offline":
        # True batch SOM; keep the learning rate constant across passes.
        som._learning_rate_decay_function = lambda lr, t, T: lr
        train = som.train_batch_offline
    else:
        train = som.train_random

    print(
        f"\nPhase 1: {cfg['n1']} {'passes' if cfg['mode'] == 'offline' else 'iterations'}"
        f" (sigma={cfg['sigma1']:.2f}, lr={cfg['lr1']}, {cfg['neighborhood']}, "
        f"{cfg['topology']})"
    )
    train(X, cfg["n1"], verbose=True)

    # Phase 2: constant radius sigma2. linear_decay_to_one keeps sigma = 1
    # constant; any other sigma2 needs an explicitly constant decay.
    som._sigma = cfg["sigma2"]  # type: ignore
    if cfg["sigma2"] != 1.0:
        som._sigma_decay_function = lambda s, t, T: s
    som._learning_rate = cfg["lr2"]
    print(
        f"\nPhase 2: {cfg['n2']} {'passes' if cfg['mode'] == 'offline' else 'iterations'}"
        f" (sigma={cfg['sigma2']:.2f}, lr={cfg['lr2']})"
    )
    train(X, cfg["n2"], verbose=True)
    return som, cfg


def config_json(cfg, **extra):
    return json.dumps({**cfg, **extra}, sort_keys=True)


# ── Grid orientation ──────────────────────────────────────────────────────────
# A trained SOM is only defined up to a symmetry of its grid, so node names
# (A1, B1, ...) depend on the seed. These helpers reorder the node grid by a
# symmetry (flip of the columns, flip of the rows and, for square grids, a
# transpose), which keeps the topology, so retrained SOMs get stable names.


def add_orient_args(parser):
    g = parser.add_argument_group("grid orientation (node naming only)")
    g.add_argument(
        "--orient-to",
        metavar="NPZ",
        default=None,
        help="Reorder the grid (by a symmetry) to best agree with the BMUs of a "
        "reference SOM cache over the same samples, e.g. the published cache.",
    )
    g.add_argument(
        "--orient-risk",
        action="store_true",
        help="Flip columns/rows so the highest flash-flood-risk node is in the "
        "last column of the first row (all-days SOM).",
    )
    return parser


def _symmetries(xdim, ydim):
    for t in (False, True) if xdim == ydim else (False,):
        for fi in (False, True):
            for fj in (False, True):
                yield dict(transpose=t, flip_i=fi, flip_j=fj)


def orient_bmus(bmus, xdim, ydim, transpose=False, flip_i=False, flip_j=False):
    i, j = bmus[:, 0], bmus[:, 1]
    if transpose:
        i, j = j, i
    if flip_i:
        i = xdim - 1 - i
    if flip_j:
        j = ydim - 1 - j
    return np.column_stack([i, j])


def orient_weights(weights, xdim, ydim, transpose=False, flip_i=False, flip_j=False):
    """weights: (xdim * ydim, F), node = i * ydim + j."""
    w = weights.reshape(xdim, ydim, -1)
    if transpose:
        w = w.swapaxes(0, 1)
    if flip_i:
        w = w[::-1]
    if flip_j:
        w = w[:, ::-1]
    return np.ascontiguousarray(w).reshape(xdim * ydim, -1)


def choose_orientation(bmus, xdim, ydim, *, ref_npz=None, is_ff=None):
    """The grid symmetry for --orient-to (max agreement with the reference
    BMUs) or --orient-risk (highest-risk node to i = xdim-1, j = 0)."""
    if ref_npz is not None:
        ref = np.load(ref_npz)["bmus"]
        if len(ref) != len(bmus):
            raise SystemExit(f"{ref_npz}: {len(ref)} BMUs, expected {len(bmus)}")
        best = max(
            _symmetries(xdim, ydim),
            key=lambda s: (orient_bmus(bmus, xdim, ydim, **s) == ref).all(1).sum(),
        )
        agree = (orient_bmus(bmus, xdim, ydim, **best) == ref).all(1).mean()
        print(f"Orientation vs {ref_npz}: {best} (agreement {agree:.0%})")
        return best
    flat = bmus[:, 0] * ydim + bmus[:, 1]
    n = xdim * ydim
    risk = np.bincount(flat[is_ff], minlength=n) / np.maximum(
        np.bincount(flat, minlength=n), 1
    )
    i, j = divmod(int(risk.argmax()), ydim)
    best = dict(transpose=False, flip_i=i != xdim - 1, flip_j=j != 0)
    print(f"Orientation by flash-flood risk: {best}")
    return best


def apply_orientation(args, weights, bmus, xdim, ydim, is_ff=None):
    """Apply --orient-to / --orient-risk to (weights, bmus); returns them and
    the symmetry used (None when neither option is given)."""
    if args.orient_to is None and not args.orient_risk:
        return weights, bmus, None
    if getattr(args, "rotate", False):
        raise SystemExit("--rotate cannot be combined with --orient-to/--orient-risk")
    if args.orient_risk and is_ff is None:
        raise SystemExit("--orient-risk needs flash-flood days (all-days SOM only)")
    s = choose_orientation(bmus, xdim, ydim, ref_npz=args.orient_to, is_ff=is_ff)
    return (
        orient_weights(weights, xdim, ydim, **s),
        orient_bmus(bmus, xdim, ydim, **s),
        s,
    )
