from wandb import config

from trainer import Trainer

import yaml
import argparse

import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from torchvision.models import resnet18

from torch.utils.data import Subset

import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    print("Config argument:", args.config)
    print("Absolute path:", os.path.abspath(args.config))


    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded:")
    print(config)

    # Step 1: Load Model
    model = resnet18(num_classes=10)

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]

    bottom_weights_percentage = config["bottom_weights_percentage"]
    
    update_strategy = config.get(
        "update_strategy",
        "adampython scripts/train_resnet.py --config config/config.yaml",
    )

    sparsity = config.get(
        "sparsity",
        0.2,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Transform Dataset
    transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
    )

    # Step 3: Get data
    train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    
    train_dataset = Subset(
        train_dataset,
        range(5000)
    )

    test_dataset = Subset(
        test_dataset,
        range(1000)
    )
    
    

    model.to(device)

    # Step 4: Train :)
    trainer = Trainer(
        model,
        train_dataset,
        test_dataset,
        epochs,
        batch_size,
        lr,
        device=device,
        use_smallest_weight=bottom_weights_percentage,
        update_strategy=update_strategy,
        sparsity=sparsity,
    )

    trainer.train()


if __name__ == "__main__":
    main()
