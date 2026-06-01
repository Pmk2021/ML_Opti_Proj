import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

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


def collect_weight_gradient_stats(model, max_points=5000):
    """
    Randomly sample weight/gradient pairs
    to avoid huge memory usage.
    """

    all_weights = []
    all_grads = []

    for param in model.parameters():

        if param.requires_grad and param.grad is not None:

            weights = param.data.abs().flatten()
            grads = param.grad.abs().flatten()

            # random subsample
            if len(weights) > max_points:

                idx = torch.randperm(len(weights))[:max_points]

                weights = weights[idx]
                grads = grads[idx]

            all_weights.append(weights.cpu())
            all_grads.append(grads.cpu())

    all_weights = torch.cat(all_weights)
    all_grads = torch.cat(all_grads)

    return all_weights.numpy(), all_grads.numpy()



def plot_weight_gradient_correlation(weight_history, grad_history):

    weights = np.concatenate(weight_history)
    grads = np.concatenate(grad_history)

    eps = 1e-12

    weights = weights + eps
    grads = grads + eps

    print("Num points:", len(weights))

    max_points = 50000

    if len(weights) > max_points:
        idx = np.random.choice(len(weights), max_points, replace=False)

        weights = weights[idx]
        grads = grads[idx]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        weights,
        grads,
        alpha=0.1,
        s=2,
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("|Weight|")
    plt.ylabel("|Gradient|")

    plt.title("Gradient magnitude vs Weight magnitude")

    plt.tight_layout()

    plt.savefig("weight_gradient_correlation.png")

    print("Saved plot.")

    plt.show()
    
def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
):

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig("loss_curves.png")

    # Accuracy plot
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, train_accuracies, label="Train")
    plt.plot(epochs, val_accuracies, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Training and Validation Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig("accuracy_curves.png")
    
def analyze_weight_gradient_buckets(weight_history, grad_history):

    weights = np.concatenate(weight_history)
    grads = np.concatenate(grad_history)

    quantiles = np.linspace(0, 1, 11)

    boundaries = np.quantile(weights, quantiles)

    print("\nWeight bucket analysis")
    print("-" * 60)

    for i in range(len(boundaries) - 1):

        low = boundaries[i]
        high = boundaries[i + 1]

        mask = (weights >= low) & (weights <= high)

        bucket_grads = grads[mask]

        mean_grad = np.mean(bucket_grads)
        median_grad = np.median(bucket_grads)

        print(
            f"Bucket {i}: "
            f"[{low:.2e}, {high:.2e}] "
            f"mean_grad={mean_grad:.3e} "
            f"median_grad={median_grad:.3e}"
        )

            

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
        update_strategy="adam",
        sparsity=0.1,
    ):
        self.model = model

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.train_losses = []
        self.val_losses = []

        self.train_accuracies = []
        self.val_accuracies = []

        self.use_smallest_weight = use_smallest_weight
        
        self.update_strategy = update_strategy
        self.sparsity = sparsity

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
        
        self.weight_history = []
        self.grad_history = []
        
    def apply_adam_mask(self):
        return
    
    
    def apply_topk_gradient_mask(self):

            for param in self.model.parameters():

                if param.grad is None:
                    continue

                grad_abs = param.grad.abs()

                threshold = torch.quantile(
                    grad_abs.flatten(),
                    1 - self.sparsity,
                )

                mask = grad_abs >= threshold

                param.grad *= mask
                
        
    def apply_random_mask(self):

            for param in self.model.parameters():

                if param.grad is None:
                    continue

                mask = (
                    torch.rand_like(param.grad)
                    < self.sparsity
                )

                param.grad *= mask
                
    def apply_small_weight_mask(self):

            weight_mask = get_bottom_k_weights(
                self.model,
                self.sparsity,
            )

            for name, param in self.model.named_parameters():

                if param.grad is not None:

                    param.grad *= weight_mask[name]
        
    def apply_gradient_mask(self):
        
        if self.update_strategy == "adam":
            return

        elif self.update_strategy == "small_weight":
            self.apply_small_weight_mask()

        elif self.update_strategy == "topk":
            self.apply_topk_gradient_mask()

        elif self.update_strategy == "random":
            self.apply_random_mask()   
    
    def train(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self.single_epoch()

            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc : {train_acc:.4f}")
            print(f"Val Loss  : {val_loss:.4f}")
            print(f"Val Acc   : {val_acc:.4f}")
            

            
            print(len(self.weight_history)) #debug
            print(len(self.grad_history))
            
        analyze_weight_gradient_buckets(
            self.weight_history,
            self.grad_history,
        )

        plot_weight_gradient_correlation(
            self.weight_history,
            self.grad_history,
        )

        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_accuracies,
            self.val_accuracies,
        )

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

            loss = self.criterion(logits, labels)

            loss.backward()
            
            # print("Backward done")
            # print("Collecting stats...")
            
            if len(self.weight_history) < 10:      #do NOT store all the batches otherwise huge RAM.
                weights, grads = collect_weight_gradient_stats(self.model)

                self.weight_history.append(weights)
                self.grad_history.append(grads)

            # print("Stats collected")

            # Keep gradients only for smallest-magnitude weights
            self.apply_gradient_mask()

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

                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
