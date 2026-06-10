import json
import random
import torch
import argparse
import yaml
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from peft import get_peft_model, LoraConfig, TaskType


def load_config(config_path="config_finetune.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        raise


def merge_configs(file_config, cli_args):
    """Merge file-based config with CLI arguments (CLI takes precedence)."""
    config = file_config.copy()

    if cli_args.config:
        config["config_path"] = cli_args.config
    if cli_args.model_dir:
        config["model_dir"] = cli_args.model_dir
    if cli_args.data_path:
        config["data_path"] = cli_args.data_path
    if cli_args.max_length:
        config["max_length"] = cli_args.max_length
    if cli_args.batch_size:
        config["batch_size"] = cli_args.batch_size
    if cli_args.learning_rate:
        config["learning_rate"] = cli_args.learning_rate
    if cli_args.num_epochs:
        config["num_epochs"] = cli_args.num_epochs
    if cli_args.device:
        config["device"] = cli_args.device

    return config


# Default config values (fallback)
DEFAULT_CONFIG = {
    "model_dir": "base_model",
    "data_path": "alpaca_data_cleaned.json",
    "max_length": 216,
    "batch_size": 16,
    "grad_accum_steps": 4,
    "learning_rate": 1e-5,
    "num_epochs": 5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "val_split": 0.1,
    "eval_steps": 500,
    "log_steps": 50,
    "mask_interval": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def create_grad_scaler(device):
    """Create gradient scaler if using CUDA."""
    if device == "cuda":
        return GradScaler("cuda")
    return None


def compute_bottom_k_mask(model, density):
    """Mask selecting the bottom `density` fraction by magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            threshold = torch.quantile(param.data.abs().float(), 1.0 - density)
            masks[name] = param.data.abs() <= threshold
    return masks


def compute_top_k_weight_mask(model, density):
    """Mask selecting the top `density` fraction by magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            threshold = torch.quantile(param.data.abs().float(), density)
            masks[name] = param.data.abs() >= threshold
    return masks


def compute_top_k_gradient_mask(model, density):
    """Mask selecting the top `density` fraction by gradient magnitude."""
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            threshold = torch.quantile(param.grad.data.abs().float(), 1.0 - density)
            masks[name] = param.grad.data.abs() >= threshold
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
def evaluate(model, dataloader, device):
    """Compute mean loss over the validation set."""

    model.eval()
    total_loss, total_batches = 0.0, 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        total_loss += outputs.loss.item()
        total_batches += 1
    model.train()
    return total_loss / total_batches if total_batches > 0 else float("nan")


def apply_mask_to_gradients(model, masks):
    """Zero out gradients outside the mask."""
    for name, param in model.named_parameters():
        if param.grad is not None and name in masks:
            param.grad[~masks[name]] = 0.0


def train(config, strategy, density=0.1, output_dir=None):
    """
    Train model with specified strategy.

    Args:
        config: Configuration dictionary from YAML
        strategy: 'standard', 'small_weight', 'top_weight', 'top_gradient', 'lora'
        density: Fraction of weights/gradients to update (0-1)
        output_dir: Output directory for model
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_dir = config.get("model_dir")
    data_path = config.get("data_path")
    max_length = config.get("max_length")
    val_split = config.get("val_split")
    batch_size = config.get("batch_size")
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")
    grad_accum_steps = config.get("grad_accum_steps")
    num_epochs = config.get("num_epochs")
    warmup_ratio = config.get("warmup_ratio")
    eval_steps = config.get("eval_steps")
    log_steps = config.get("log_steps")
    mask_interval = config.get("mask_interval")

    scaler = create_grad_scaler(device)

    if output_dir is None:
        if strategy == "standard":
            output_dir = "./finetuned_model"
        elif strategy == "small_weight":
            output_dir = f"./bot_{int(density * 100)}_mag_size"
        elif strategy == "top_weight":
            output_dir = f"./top_{int(density * 100)}_mag_size"
        elif strategy == "top_gradient":
            output_dir = f"./top_grad_{int(density * 100)}"
        elif strategy == "lora":
            output_dir = "./lora_model"

    print(f"Using device: {device}")
    print(f"Strategy: {strategy} (density: {density})")
    print(f"Output directory: {output_dir}")
    print("\nConfiguration:")
    for key, value in config.items():
        if key != "device":
            print(f"  {key}: {value}")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
    )
    model.to(device)

    # Apply LoRA if specified
    if strategy == "lora":
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

    # Training loop
    print(f"Starting training: {num_epochs} epochs")
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    masks = {}
    if strategy == "small_weight":
        masks = compute_bottom_k_mask(model, density)
    elif strategy == "top_weight":
        masks = compute_top_k_weight_mask(model, density)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device):
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

                # Apply gradient masks
                if strategy == "small_weight" and masks:
                    apply_mask_to_gradients(model, masks)
                elif strategy == "top_weight":
                    if global_step % mask_interval == 0:
                        masks = compute_top_k_weight_mask(model, density)
                    if masks:
                        apply_mask_to_gradients(model, masks)
                elif strategy == "top_gradient":
                    if global_step % mask_interval == 0:
                        masks = compute_top_k_gradient_mask(model, density)
                    if masks:
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
                    val_loss = evaluate(model, val_loader, device)
                    # Print on same line if LOG_STEPS == EVAL_STEPS, else new line
                    if global_step % log_steps == 0:
                        print(f" | Val loss: {val_loss:.4f}")
                    else:
                        print(
                            f"Epoch {epoch + 1} | Step {global_step} | Val loss: {val_loss:.4f}"
                        )
                elif global_step % log_steps == 0:
                    print()

        print(f"Epoch {epoch + 1} complete.")

    # Save
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM with various optimization strategies"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_finetune.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["standard", "small_weight", "top_weight", "top_gradient", "lora"],
        default="standard",
        help="Training strategy to use",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.1,
        help="Fraction of weights/gradients to update (0-1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model (auto-generated if not provided)",
    )

    # Optional CLI overrides for config values
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory")
    parser.add_argument("--data-path", type=str, default=None, help="Data path")
    parser.add_argument(
        "--max-length", type=int, default=None, help="Max sequence length"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Using default config (file not found: {args.config})")
        config = DEFAULT_CONFIG.copy()

    config = merge_configs(config, args)

    train(config, args.strategy, args.density, args.output_dir)
