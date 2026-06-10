"""
Unified fine-tuning script for Alpaca-style instruction data with multiple optimization strategies.
Supports: standard training, LoRA, magnitude-based sparsity (top-k/bottom-k), and gradient-based sparsity.

Requires: pip install transformers datasets torch accelerate safetensors peft
"""

import argparse
import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

scaler = GradScaler("cuda")


def format_prompt(example: dict) -> str:
    """Format an Alpaca example into a prompt+response string."""
    if example.get("input", "").strip():
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return prompt


class AlpacaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.encodings = []
        for example in raw:
            text = format_prompt(example)
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze()
            attention_mask = enc["attention_mask"].squeeze()
            # Labels are the same as input_ids; mask padding tokens with -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.encodings.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


@torch.no_grad()
def evaluate(model, dataloader):
    """Compute mean loss over the validation set."""
    model.eval()
    total_loss, total_batches = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        total_loss += outputs.loss.item()
        total_batches += 1
    model.train()
    return total_loss / total_batches if total_batches > 0 else float("nan")


def compute_top_k_mask(model, density):
    """Return a dict of masks selecting the top `density` fraction by magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            threshold = torch.quantile(param.data.abs().float(), 1.0 - density)
            masks[name] = param.data.abs() >= threshold
    return masks


def compute_bottom_k_mask(model, density):
    """Return a dict of masks selecting the bottom `density` fraction by magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            threshold = torch.quantile(param.data.abs().float(), density)
            masks[name] = param.data.abs() <= threshold
    return masks


def compute_gradient_top_mask(model, density):
    """Return a dict of masks selecting the top `density` fraction by gradient magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            threshold = torch.quantile(param.grad.data.abs().float(), 1.0 - density)
            masks[name] = param.grad.data.abs() >= threshold
    return masks


def compute_gradient_bottom_mask(model, density):
    """Return a dict of masks selecting the bottom `density` fraction by gradient magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            threshold = torch.quantile(param.grad.data.abs().float(), density)
            masks[name] = param.grad.data.abs() <= threshold
    return masks


def apply_mask_to_gradients(model, masks):
    """Zero out gradients not selected by masks."""
    for name, param in model.named_parameters():
        if param.grad is not None and name in masks:
            param.grad[~masks[name]] = 0.0


# Default paths & hyperparameters


def output_dir_default(strategy, density):
    ans = ""
    match strategy:
        case "standard":
            ans = "finetuned_model"
        case "lora":
            ans = "lora_model"
        case "magnitude_top":
            ans = f"magnitude_top_{int(density * 100)}"
        case "magnitude_bottom":
            ans = f"magnitude_bottom_{int(density * 100)}"
        case "gradient_top":
            ans = f"gradient_top_{int(density * 100)}"
        case "gradient_bottom":
            ans = f"gradient_bottom_{int(density * 100)}"
    return "./output/" + ans


def train(config):

    strategy = config.get("strategy", "standard")
    model_dir = config.get("model_dir", "base_model")
    data_path = config.get("data_path", "alpaca_data_cleaned.json")
    max_length = config.get("max_length", 216)
    batch_size = config.get("batch_size", 16)
    grad_accum_steps = config.get(
        "grad_accum_steps", 4
    )  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    learning_rate = config.get("learning_rate", 1e-5)
    num_epochs = config.get("num_epochs", 5)
    warmup_ratio = config.get("warmup_ratio", 0.03)
    weight_decay = config.get("weight_decay", 0.01)
    val_split = config.get("val_split", 0.1)
    eval_steps = config.get("eval_steps", 500)
    log_steps = config.get("log_steps", 50)
    sparse_density = config.get("sparse_density", 1.0)
    mask_interval = config.get("mask_interval", 100)
    output_dir = config.get("output_dir", output_dir_default(strategy, sparse_density))

    print(f"Using device: {DEVICE}")
    print(f"Training strategy: {strategy}")
    if strategy != "standard" and strategy != "lora":
        print(f"Sparsity density: {sparse_density}")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
    )
    model.to(DEVICE)

    # Apply LoRA if requested
    if strategy == "lora":
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.train()

    # Load dataset
    print("Loading dataset...")
    dataset = AlpacaDataset(data_path, tokenizer, max_length)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_size = max(1, int(len(dataset) * val_split))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    print(f"  {len(train_idx)} train / {len(val_idx)} val examples")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Select mask computation function and determine if we use gradient-based masks
    mask_fn = None
    use_gradient_mask = False
    masks = {}

    if strategy == "magnitude_top":
        mask_fn = compute_top_k_mask
        masks = mask_fn(model, sparse_density)
    elif strategy == "magnitude_bottom":
        mask_fn = compute_bottom_k_mask
        masks = mask_fn(model, sparse_density)
    elif strategy == "gradient_top":
        mask_fn = compute_gradient_top_mask
        use_gradient_mask = True
    elif strategy == "gradient_bottom":
        mask_fn = compute_gradient_bottom_mask
        use_gradient_mask = True

    # Training loop
    print(f"Starting training: {num_epochs} epochs")
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                global_step += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)

                # Apply masks if using sparsity
                if mask_fn is not None:
                    if use_gradient_mask:
                        if global_step % mask_interval == 0:
                            masks = mask_fn(model, sparse_density)
                    apply_mask_to_gradients(model, masks)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % log_steps == 0:
                    avg_train_loss = running_loss / log_steps
                    running_loss = 0.0
                    print(
                        f"Epoch {epoch + 1} | Step {global_step} | Train loss: {avg_train_loss:.4f}",
                        end="",
                    )

                if global_step % eval_steps == 0:
                    val_loss = evaluate(model, val_loader)
                    # Print on same line if LOG_STEPS == EVAL_STEPS, else new line
                    if global_step % log_steps == 0:
                        print(f" | Val loss: {val_loss:.4f}")
                    else:
                        print(
                            f"Epoch {epoch + 1} | Step {global_step} | Val loss: {val_loss:.4f}"
                        )
                elif global_step % log_steps == 0:
                    print()  # newline after train loss if no eval this step

        print(f"Epoch {epoch + 1} complete.")

    # Save
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
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

    train(config)
