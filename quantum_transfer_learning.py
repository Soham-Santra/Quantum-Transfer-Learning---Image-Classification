"""Hybrid quantum transfer-learning image classifier.

The model uses a frozen ImageNet-pretrained ResNet-18 as a feature extractor,
compresses its features to four values, processes those values with a
variational quantum circuit, and classifies CIFAR-10 images.
"""

from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models


SEED = 42
NUM_CLASSES = 10
NUM_QUBITS = 4
NUM_LAYERS = 2
IMAGE_SIZE = 224
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed: int) -> None:
    """Make data ordering and model initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_dir: str, batch_size: int, workers: int
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Download CIFAR-10 and return separate training and evaluation loaders."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(test_dataset, shuffle=False, **common),
        train_dataset.classes,
    )


def create_quantum_layer() -> nn.Module:
    """Create the differentiable four-qubit variational layer."""
    quantum_device = qml.device("lightning.qubit", wires=NUM_QUBITS)

    @qml.qnode(quantum_device, interface="torch", diff_method="adjoint")
    def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
        for qubit in range(NUM_QUBITS):
            qml.RY(inputs[qubit], wires=qubit)
        qml.templates.StronglyEntanglingLayers(
            weights, wires=range(NUM_QUBITS)
        )
        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(NUM_QUBITS)]

    weight_shapes = {"weights": (NUM_LAYERS, NUM_QUBITS, 3)}
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class HybridModel(nn.Module):
    """Frozen ResNet-18 plus a trainable classical-quantum classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.fc = nn.Identity()

        self.pre_quantum = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_QUBITS),
        )
        self.quantum = create_quantum_layer()
        self.classifier = nn.Linear(NUM_QUBITS, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        encoded = torch.tanh(self.pre_quantum(features))
        quantum_outputs = torch.stack(
            [self.quantum(sample) for sample in encoded]
        )
        return self.classifier(quantum_outputs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    loss = total_loss / total
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch:02d} | loss {loss:.4f} | accuracy {accuracy:.2f}%")
    return loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Evaluate the model and return classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        outputs = model(images.to(device))
        correct += (outputs.argmax(dim=1) == labels.to(device)).sum().item()
        total += labels.size(0)
    accuracy = 100.0 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, classes = build_dataloaders(
        args.data_dir, args.batch_size, args.workers
    )
    print(f"Classes: {', '.join(classes)}")

    model = HybridModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
    )

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
