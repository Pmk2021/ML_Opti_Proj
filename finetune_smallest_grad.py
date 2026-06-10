"""
Fine-tuning script for Alpaca-style instruction data.
Requires: pip install transformers datasets torch accelerate safetensors
"""

import json
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
import argparse
# ─────────────────────────────────────────────
# HARDCODED PATHS & HYPERPARAMETERS — edit here
# ─────────────────────────────────────────────

scaler = GradScaler("cuda")

MODEL_DIR = "C:\\Users\\prana\\OneDrive\\Desktop\\optimizer_project\\base_model"  # folder containing config.json + model.safetensors
DATA_PATH = "alpaca_data_cleaned.json"


MASK_INTERVAL = 100  # recalculate mask every N optimizer steps

MAX_LENGTH = 216
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT = 0.1  # fraction of data held out for validation
EVAL_STEPS = 500  # run validation every N optimizer steps
LOG_STEPS = 50  # print training loss every N optimizer steps

# ─────────────────────────────────────────────


def compute_gradient_mask(model, density):
    """Return a dict of masks selecting the top `density` fraction by gradient magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            threshold = torch.quantile(param.grad.data.abs().float(), density)
            masks[name] = param.grad.data.abs() <= threshold
    return masks


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


def train():
    print(f"Using device: {DEVICE}")

    # Load tokenizer and model from local directory
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32 if DEVICE == "cuda" else torch.float32,
    )
    model.to(DEVICE)
    model.train()

    # Dataset — random train/val split
    print("Loading dataset...")
    dataset = AlpacaDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    print(f"  {len(train_idx)} train / {len(val_idx)} val examples")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # Training loop
    print(f"Starting training: {NUM_EPOCHS} epochs")
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    masks = {}
    for epoch in range(NUM_EPOCHS):
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
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                global_step += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)

                # Compute gradient mask and zero out small gradients
                if global_step % MASK_INTERVAL == 0:
                    masks = compute_gradient_mask(model, SPARSE_DENSITY)

                for name, param in model.named_parameters():
                    if param.grad is not None and name in masks:
                        param.grad[~masks[name]] = 0.0

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % LOG_STEPS == 0:
                    avg_train_loss = running_loss / LOG_STEPS
                    running_loss = 0.0
                    print(
                        f"Epoch {epoch + 1} | Step {global_step} | Train loss: {avg_train_loss:.4f}",
                        end="",
                    )

                if global_step % EVAL_STEPS == 0:
                    val_loss = evaluate(model, val_loader)
                    # Print on same line if LOG_STEPS == EVAL_STEPS, else new line
                    if global_step % LOG_STEPS == 0:
                        print(f" | Val loss: {val_loss:.4f}")
                    else:
                        print(
                            f"Epoch {epoch + 1} | Step {global_step} | Val loss: {val_loss:.4f}"
                        )
                elif global_step % LOG_STEPS == 0:
                    print()  # newline after train loss if no eval this step

        print(f"Epoch {epoch + 1} complete.")

    # Save
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=0.1)
    args = parser.parse_args()
    OUTPUT_DIR = f"./top_grad_{int(args.density * 100)}"
    SPARSE_DENSITY = args.density
    train()
