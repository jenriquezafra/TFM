# TFM
Computational algorithms for pricing financial derivatives under local volatility models


## Repository structure

(may not be totally up to date)
```text

.
├── figures/                 # Generated figures (plots, final results)
├── .venv/                   # Local virtual environment (not versioned)
├── configs/                 # Experiment configuration files
│   ├── synth.yaml              # Synthetic data generation (Heston + COS)
│   └── train.yaml              # Neural network training configuration
│
├── data/
│   ├── market/                 # Market data (when used)
│   └── synth/                  # Synthetic data generated with COS
│
├── notebooks/               # Exploratory notebooks (non-core)
│   ├── 01_toy_NN.ipynb
│   ├── 02_Heston_solver.ipynb
│   ├── 03_NN_surrogate.ipynb
│   ├── figures.ipynb
│   └── some_checks.ipynb
│
├── scripts/                 # Executable entry points
│   ├── gen_synth.py            # Synthetic data generation
│   ├── train_pricer.py         # Surrogate pricer training
│   └── eval_pricer.py          # COS vs NN evaluation
│
├── src/                     # Reusable project code
│   ├── datasets/
│   │   ├── make_synth.py           # Synthetic dataset generation logic
│   │   └── market.py               # Market data ingestion/preprocessing (future)
│   ├── models/
│   │   ├── pricer_nn.py            # Surrogate pricer architecture (TF/Keras)
│   │   └── calibrator_nn.py        # NN architecture for calibration
│   ├── solvers/
│   │   └── heston_cos.py           # Heston COS solver (ground truth)
│   ├── eval.py                 # Metrics and benchmarks
│   ├── train.py                # Generic training loops
│   └── utils.py                # Common utilities (checks, seeds, helpers)
│
├── artifacts/               # Data from the models
│   ├── metrics/            
│   ├── models/                 # trained NN models and related artifacts
│       ├── pricer_nn/
│       │   ├── baseline/           # reference experiment used as benchmark
│       │   │   ├── checkpoints/        # intermediate models saved during training via callbacks 
│       │   │   └── final/              # selected final model and associated configuration
│       │   └── v2/                 # alternative experiment (TBD)
│       └── calibrator_nn/ 
│
├── tests/                   # Minimal tests (sanity checks)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```
Note:
- Core logic lives in 'src/'
- 'scripts/' only orchestrade experiments and ensure reproducibility
- Notebooks are used for exploration only
