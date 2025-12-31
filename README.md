# BayesCal: Bayesian Neural Network Calibration Research

A research project demonstrating Bayesian Neural Networks using Bayes by Backprop, comparing calibration performance against feedforward neural networks with dropout.

## Project Structure

```
bayescal/
├── README.md                 # Project overview and setup instructions
├── GOALS.md                  # Project goals and objectives
├── pyproject.toml            # Modern Python project configuration
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── .env.example            # Example environment variables
│
├── bayescal/                # Main package
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   │
│   ├── data/                # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py       # Data loaders (CIFAR-10, CIFAR-100, toy dataset)
│   │   └── preprocessing.py # Data preprocessing utilities
│   │
│   ├── models/              # Neural network models
│   │   ├── __init__.py
│   │   ├── bayesian.py      # Bayes by Backprop implementation
│   │   ├── feedforward.py   # Feedforward NN with dropout
│   │   └── layers/          # Custom JAX layers
│   │       ├── __init__.py
│   │       ├── bayesiandense.py  # Bayesian Dense layer
│   │       └── bayesianconv2d.py  # Bayesian Conv2D layer
│   │
│   ├── training/            # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop
│   │   └── optimizers.py    # Optimizer configurations
│   │
│   ├── evaluation/          # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── calibration.py   # ECE, Brier score, calibration curves
│   │   └── ood.py           # Out-of-distribution detection
│   │
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py       # Logging configuration
│   │   ├── visualization.py # Plotting utilities
│   │   └── toy_dataset.py  # Toy dataset analysis utilities
│   │
│   └── api/                 # FastAPI application (if needed)
│       ├── __init__.py
│       ├── main.py          # FastAPI app
│       └── endpoints.py     # API endpoints
│
├── scripts/                 # Executable scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── compare.py           # Comparison script
│
├── notebooks/               # Jupyter notebooks for exploration
│   └── toy.ipynb            # Toy dataset demonstration and analysis
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_data.py
│
├── experiments/             # Experiment configurations and results
│   ├── configs/             # Experiment configs (YAML/JSON)
│   └── results/             # Saved results and checkpoints
│
└── docs/                    # Documentation
    └── api.md               # API documentation
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or using pip with pyproject.toml
pip install -e .
```

## Usage

```bash
# Train a model
python scripts/train.py --config experiments/configs/bayesian_config.yaml

# Evaluate calibration
python scripts/evaluate.py --model-path experiments/results/bayesian_model.pkl

# Compare models
python scripts/compare.py
```

## Features

- **Bayesian Neural Networks**: Implementation of Bayes by Backprop
- **Custom JAX Layers**: Registered custom layers for Bayesian inference
- **Calibration Metrics**: ECE, Brier score, and calibration curves
- **Comparison Framework**: Systematic comparison with feedforward dropout-based models

