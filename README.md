# Quantum Transfer Learning for Image Classification

An experimental hybrid classical-quantum image classifier for CIFAR-10. The
project combines transfer learning from an ImageNet-pretrained ResNet-18 with
a trainable variational quantum circuit built with PennyLane.

## What this project demonstrates

- Reusing a pretrained CNN as a frozen visual feature extractor.
- Compressing 512-dimensional CNN features into four quantum-ready values.
- Encoding those values with four `RY` rotations.
- Learning a variational circuit with
  `StronglyEntanglingLayers` and differentiable adjoint simulation.
- Using quantum expectation values as inputs to a ten-class classifier.

## Architecture

```text
CIFAR-10 image (32 x 32)
        |
Resize to 224 x 224 + normalization
        |
Frozen ImageNet ResNet-18 (512 features)
        |
Linear(512 -> 128) -> ReLU -> Linear(128 -> 4) -> tanh
        |
4-qubit variational quantum circuit
        |
4 Pauli-Z expectation values
        |
Linear(4 -> 10)
        |
CIFAR-10 class prediction
```

Only the feature-compression layers, quantum circuit parameters, and final
classifier are trained. The ResNet-18 weights remain frozen.

## Dataset

The script downloads CIFAR-10 automatically using `torchvision`. CIFAR-10
contains 60,000 colour images in ten classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`,
`ship`, and `truck`.

The training pipeline uses random horizontal flips. Evaluation uses the same
normalization and resize steps without random augmentation.

## Installation

Python 3.10+ is recommended. Create an environment and install the
dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run also downloads the CIFAR-10 dataset and the pretrained ResNet-18
weights. An internet connection is therefore required for initial setup.

## Usage

Run the default 20-epoch experiment:

```bash
python quantum_transfer_learning.py
```

Useful options:

```bash
python quantum_transfer_learning.py --epochs 5 --batch-size 64
python quantum_transfer_learning.py --data-dir ./data --workers 0
```

The code automatically uses CUDA when available and otherwise runs on CPU.
Quantum simulation is performed locally with PennyLane Lightning; no quantum
hardware account is required.

## Reproducibility

The default random seed is `42`. It can be changed with `--seed`:

```bash
python quantum_transfer_learning.py --seed 123
```

Exact results can still vary between hardware, library versions, and CUDA
settings. The circuit is simulated one sample at a time, so training is
intentionally focused on clarity and experimentation rather than throughput.

## Project structure

```text
.
├── quantum_transfer_learning.py  # Training and evaluation entry point
├── requirements.txt              # Python dependencies
├── data/                         # Downloaded CIFAR-10 files (ignored)
├── .gitignore
└── LICENSE
```

## Limitations and next steps

This is a research/learning prototype, not a production benchmark. The
ResNet backbone is frozen, the quantum circuit has only four qubits, and no
checkpointing or experiment tracking is included yet. Natural extensions
include comparing against a classical head, fine-tuning selected ResNet
layers, adding validation metrics, batching quantum execution, and comparing
different circuit ansatzes.

## License

This project is released under the [MIT License](LICENSE).
