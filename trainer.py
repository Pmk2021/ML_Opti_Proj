import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

"""
Trainer Class for models.
Implements smallest-weight gradient update

"""


def get_bottom_k_weights(model, percent=0.10):
    """
    Calculates bottom k percent of all weights, and returns mask to apply during gradient update to onlz train them
    """

    all_weights = []

    for param in model.parameters():
        if param.requires_grad:
            all_weights.append(param.data.abs().flatten())

    all_weights = torch.cat(all_weights)

    threshold = torch.quantile(
        all_weights,
        percent,
    )

    print(f"Freezing weights with |w| <= {threshold:.6f}")

    masks = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # True = trainable
            mask = param.data.abs() < threshold

            masks[name] = mask

    return masks


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        epochs,
        batch_size,
        lr,
        device=None,
        use_smallest_weight=None,
    ):
        self.model = model

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.use_smallest_weight = use_smallest_weight

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
        )

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self.single_epoch()

            val_loss, val_acc = self.validate()

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc : {train_acc:.4f}")
            print(f"Val Loss  : {val_loss:.4f}")
            print(f"Val Acc   : {val_acc:.4f}")

    def single_epoch(self):
        """Train Single Epoch"""
        self.model.train()

        weight_mask = get_bottom_k_weights(self.model)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader)

        for batch in progress_bar:
            images, labels = batch

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            loss = self.loss(logits, labels)

            loss.backward()

            # Apply weight mask to set gradient of all weights below above thresshold to 0
            for name, param in weight_mask.named_parameters():
                if param.grad is not None:
                    param.grad *= weight_mask[name]

            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def validate(self):
        """Validate Single Epoch"""
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch

                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)

                loss = self.loss(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def loss(self, logits, labels):
        """Return Cross Entropy Loss"""
        return self.criterion(logits, labels)
