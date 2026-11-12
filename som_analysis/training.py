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
