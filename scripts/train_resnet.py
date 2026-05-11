from trainer import Trainer

import yaml
import argparse

import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Step 1: Load Model
    model_name = config["model"]

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=10,
        ignore_mismatched_sizes=True,
    )

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]

    bottom_weights_percentage = config["bottom_weights_percentage"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: Image Processor
    processor = AutoImageProcessor.from_pretrained(model_name)

    if "height" in processor.size:
        image_size = processor.size["height"]
    elif "shortest_edge" in processor.size:
        image_size = processor.size["shortest_edge"]
    else:
        image_size = 224

    # Transform Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=getattr(processor, "image_mean", [0.5, 0.5, 0.5]),
                std=getattr(processor, "image_std", [0.5, 0.5, 0.5]),
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
    )

    trainer.train()


if __name__ == "__main__":
    main()
