import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
from functools import partial

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dna_gpt.model.dna_gpt import DNAGPT
from dna_gpt.tokenizer import KmerTokenizer

try:
    from dna_gpt.model.adapter_lm import (
        build_lm_model,
        count_trainable_params,
        reinit_top_transformer_layers,
    )
except ModuleNotFoundError:
    _adapter_path = Path(__file__).resolve().parent / "dna_gpt" / "model" / "adapter_lm.py"
    if _adapter_path.exists():
        _spec = importlib.util.spec_from_file_location("dna_gpt.model.adapter_lm", _adapter_path)
        _module = importlib.util.module_from_spec(_spec)
        assert _spec is not None and _spec.loader is not None
        _spec.loader.exec_module(_module)
        build_lm_model = _module.build_lm_model
        count_trainable_params = _module.count_trainable_params
        reinit_top_transformer_layers = _module.reinit_top_transformer_layers
    else:
        raise ModuleNotFoundError(
            f"Cannot find adapter_lm module. Expected file: {_adapter_path}"
        )


SPECIAL_TOKENS = (
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    + ["+", '-', '*', '/', '=', "&", "|", "!"]
    + ['M', 'B']
    + ['P']
    + ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V']
    + ['W', 'Y', 'X', 'Z']
)
IGNORE_INDEX = -100


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_name: str) -> KmerTokenizer:
    dynamic_kmer = model_name != "dna_gpt0.1b_h"
    return KmerTokenizer(k=6, reserved_tokens=SPECIAL_TOKENS, dynamic_kmer=dynamic_kmer)


def build_token_base_lengths(tokenizer: KmerTokenizer):
    lengths = []
    for tid in range(len(tokenizer)):
        piece = tokenizer.id_to_piece(tid)
        if piece.startswith("<") and piece.endswith(">"):
            lengths.append(0)
        else:
            lengths.append(sum(ch in "ATCGN" for ch in piece))
    return lengths


def split_prefix_body(seq: str):
    if seq.startswith("<"):
        end = seq.find(">")
        if end != -1:
            return seq[: end + 1], seq[end + 1 :]
    return "", seq


def normalize_sequence(
    seq: str,
    species_token: str,
    kmer_k: int,
    trim_to_k: bool,
    allow_n_base: bool,
):
    seq = seq.strip().upper()
    if not seq:
        return ""
    if species_token and not seq.startswith("<"):
        seq = f"<{species_token}>{seq}"

    prefix, body = split_prefix_body(seq)
    allowed = set("ATCGN" if allow_n_base else "ATCG")
    body = "".join(ch for ch in body if ch in allowed)
    if trim_to_k and kmer_k > 1:
        body = body[: (len(body) // kmer_k) * kmer_k]
    if len(body) == 0:
        return ""
    return prefix + body


class KmerNextTokenDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: KmerTokenizer,
        max_tokens: int,
        species_token: str,
        trim_to_k: bool,
        allow_n_base: bool,
        max_samples: int = 0,
    ):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                seq = normalize_sequence(
                    line,
                    species_token=species_token,
                    kmer_k=tokenizer.k,
                    trim_to_k=trim_to_k,
                    allow_n_base=allow_n_base,
                )
                if not seq:
                    continue

                token_ids = tokenizer.encode(seq, max_len=max_tokens + 1, to_tensor=False)
                if len(token_ids) < 2:
                    continue

                self.samples.append(torch.tensor(token_ids, dtype=torch.long))
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

        if not self.samples:
            raise ValueError(f"No valid samples found in {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return ids[:-1], ids[1:]


def collate_batch(batch, pad_id: int):
    xs, ys = zip(*batch)
    x = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    y = pad_sequence(ys, batch_first=True, padding_value=IGNORE_INDEX)
    return x, y


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_optimizer(model, args, device):
    optim_kwargs = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    if args.fused_adamw and device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optim_kwargs)
            print("Using fused AdamW.")
            return optimizer
        except TypeError:
            print("fused AdamW is not available in this torch build, fallback to regular AdamW.")
    return torch.optim.AdamW(model.parameters(), **optim_kwargs)


def repetition_unlikelihood_loss(logits, targets, ignore_index=-100):
    # Penalize repeating the previous token when the current ground-truth token is different.
    if targets.size(1) < 2:
        return logits.new_zeros(())

    prev_tokens = targets[:, :-1]
    curr_tokens = targets[:, 1:]
    valid = (prev_tokens != ignore_index) & (curr_tokens != ignore_index) & (prev_tokens != curr_tokens)
    if not torch.any(valid):
        return logits.new_zeros(())

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    repeat_logp = log_probs[:, 1:, :].gather(-1, prev_tokens.unsqueeze(-1)).squeeze(-1)
    repeat_prob = repeat_logp.exp().clamp(max=1.0 - 1e-6)
    ul = -torch.log((1.0 - repeat_prob).clamp_min(1e-6))
    ul = ul * valid.to(ul.dtype)
    return ul.sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, token_base_len_table, fallback_k):
    model.eval()
    loss_sum = 0.0
    total_correct = 0
    total_valid = 0
    total_base_tokens = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        preds = torch.argmax(logits, dim=-1)
        valid_mask = y.ne(IGNORE_INDEX)
        total_correct += (preds.eq(y) & valid_mask).sum().item()
        valid_tokens = valid_mask.sum().item()
        total_valid += valid_tokens
        if valid_tokens > 0:
            total_base_tokens += token_base_len_table[y[valid_mask]].sum().item()
        loss_sum += loss.item()

    avg_loss = loss_sum / max(1, len(loader))
    ppl = math.exp(min(20.0, avg_loss))
    avg_bases_per_token = total_base_tokens / max(1, total_valid)
    if avg_bases_per_token <= 0:
        avg_bases_per_token = float(max(1, fallback_k))
    base_ppl = math.exp(min(20.0, avg_loss / max(1e-6, avg_bases_per_token)))
    token_acc = total_correct / max(1, total_valid)
    return avg_loss, ppl, base_ppl, token_acc, avg_bases_per_token


def save_training_plots(
    plot_dir: str,
    train_step_points,
    train_step_losses,
    val_step_points,
    val_step_losses,
    val_step_ppls,
    val_step_base_ppls,
    train_epoch_points,
    train_epoch_losses,
    dpi: int,
):
    os.makedirs(plot_dir, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skip training plot export: matplotlib is not installed ({exc}).")
        return

    # trainloss_batch.png (legacy name kept)
    if train_step_points and train_step_losses:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(train_step_points, train_step_losses, color="#1f77b4", linewidth=1.8)
        ax.set_title("Train Loss (Update Steps)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "trainloss_batch.png"), dpi=dpi)
        plt.close(fig)

    # valloss_epoch.png (legacy name kept)
    if val_step_points and val_step_losses:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(val_step_points, val_step_losses, color="#d62728", linewidth=1.8)
        ax.set_title("Validation Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "valloss_epoch.png"), dpi=dpi)
        plt.close(fig)

    # valperplexity.png
    if val_step_points and val_step_ppls:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(val_step_points, val_step_ppls, color="#d62728", linewidth=1.8, label="Token PPL")
        if val_step_base_ppls:
            ax.plot(val_step_points, val_step_base_ppls, color="#2ca02c", linewidth=1.8, label="Base-eq PPL")
        ax.set_title("Validation Perplexity")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.legend(loc="best")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "valperplexity.png"), dpi=dpi)
        plt.close(fig)

    # trainloss_epoch.png
    if train_epoch_points and train_epoch_losses:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(train_epoch_points, train_epoch_losses, color="#9467bd", linewidth=1.8, marker="o")
        ax.set_title("Train Loss (Epoch)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "trainloss_epoch.png"), dpi=dpi)
        plt.close(fig)

    # train_val_loss_step.png (new combined chart)
    if train_step_points and val_step_points and train_step_losses and val_step_losses:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(train_step_points, train_step_losses, color="#1f77b4", linewidth=1.6, label="Train loss")
        ax.plot(val_step_points, val_step_losses, color="#d62728", linewidth=1.8, label="Val loss")
        ax.set_title("Train/Val Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "train_val_loss_step.png"), dpi=dpi)
        plt.close(fig)

    history = {
        "train_step_points": train_step_points,
        "train_step_losses": train_step_losses,
        "val_step_points": val_step_points,
        "val_step_losses": val_step_losses,
        "val_step_ppls": val_step_ppls,
        "val_step_base_ppls": val_step_base_ppls,
        "train_epoch_points": train_epoch_points,
        "train_epoch_losses": train_epoch_losses,
    }
    with open(os.path.join(plot_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main(args):
    set_seed(args.seed)
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be > 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be > 0")
    if args.warmup_steps < -1:
        raise ValueError("--warmup_steps must be >= -1")
    if args.adapter_dim <= 0:
        raise ValueError("--adapter_dim must be > 0")
    if args.adapter_dropout < 0:
        raise ValueError("--adapter_dropout must be >= 0")
    if args.reinit_top_layers < 0:
        raise ValueError("--reinit_top_layers must be >= 0")
    if args.unlikelihood_alpha < 0:
        raise ValueError("--unlikelihood_alpha must be >= 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.use_amp and device.type == "cuda"

    tokenizer = build_tokenizer(args.model_name)
    token_base_len_table = torch.tensor(build_token_base_lengths(tokenizer), dtype=torch.float32, device=device)
    vocab_size = len(tokenizer)
    backbone = DNAGPT.from_name(args.model_name, vocab_size=vocab_size)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    raw_checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt_args = raw_checkpoint.get("args", {}) if isinstance(raw_checkpoint, dict) else {}
    model_state = raw_checkpoint["model"] if isinstance(raw_checkpoint, dict) and "model" in raw_checkpoint else raw_checkpoint
    is_wrapped_checkpoint = isinstance(model_state, dict) and any(k.startswith("backbone.") for k in model_state.keys())
    if is_wrapped_checkpoint:
        backbone_state = {k[len("backbone.") :]: v for k, v in model_state.items() if k.startswith("backbone.")}
    else:
        backbone_state = model_state

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    print(f"Backbone loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys.")
    if missing:
        print("Backbone missing key sample:", missing[:10])
    if unexpected:
        print("Backbone unexpected key sample:", unexpected[:10])
    if args.strict_load and (not is_wrapped_checkpoint) and (len(missing) > 0 or len(unexpected) > 0):
        raise RuntimeError("Checkpoint keys mismatch on backbone. Use --no_strict_load to bypass.")

    if args.reinit_top_layers > 0:
        reinit_top_transformer_layers(backbone, args.reinit_top_layers)
        print(f"Reinitialized top {args.reinit_top_layers} transformer layers.")

    adapter_mode = args.adapter_mode
    if adapter_mode == "auto":
        adapter_mode = ckpt_args.get("adapter_mode", "none")
    adapter_layers = args.adapter_layers
    if adapter_layers == "auto":
        adapter_layers = ckpt_args.get("adapter_layers", "")
    adapter_dim = args.adapter_dim if args.adapter_dim > 0 else int(ckpt_args.get("adapter_dim", 256))
    adapter_dropout = (
        args.adapter_dropout if args.adapter_dropout >= 0 else float(ckpt_args.get("adapter_dropout", 0.1))
    )

    model, adapter_layer_ids = build_lm_model(
        backbone=backbone,
        adapter_mode=adapter_mode,
        adapter_layers=adapter_layers,
        adapter_dim=adapter_dim,
        adapter_dropout=adapter_dropout,
    )

    if is_wrapped_checkpoint:
        missing_full, unexpected_full = model.load_state_dict(model_state, strict=False)
        print(
            f"Full model restored from wrapped checkpoint with {len(missing_full)} missing keys "
            f"and {len(unexpected_full)} unexpected keys."
        )
        if missing_full:
            print("Full model missing key sample:", missing_full[:10])
        if unexpected_full:
            print("Full model unexpected key sample:", unexpected_full[:10])
        if args.strict_load and (len(missing_full) > 0 or len(unexpected_full) > 0):
            raise RuntimeError("Checkpoint keys mismatch on wrapped model. Use --no_strict_load to bypass.")

    model = model.to(device)
    run_args = vars(args).copy()
    run_args["adapter_mode"] = adapter_mode
    run_args["adapter_layers"] = adapter_layers
    run_args["adapter_dim"] = adapter_dim
    run_args["adapter_dropout"] = adapter_dropout
    print(
        f"Model ready: adapter_mode={adapter_mode}, adapter_layers={adapter_layer_ids}, "
        f"trainable_params={count_trainable_params(model)/1e6:.2f}M"
    )

    train_dataset = KmerNextTokenDataset(
        args.train_file,
        tokenizer,
        max_tokens=args.max_tokens,
        species_token=args.species_token,
        trim_to_k=args.trim_to_k,
        allow_n_base=args.allow_n_base,
        max_samples=args.max_train_samples,
    )
    val_dataset = KmerNextTokenDataset(
        args.val_file,
        tokenizer,
        max_tokens=args.max_tokens,
        species_token=args.species_token,
        trim_to_k=args.trim_to_k,
        allow_n_base=args.allow_n_base,
        max_samples=args.max_val_samples,
    )
    collate_fn = partial(collate_batch, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    optimizer = create_optimizer(model, args, device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_steps >= 0:
        warmup_steps = min(args.warmup_steps, max(0, total_steps - 1))
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_lr_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(
        f"Dataset loaded: train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, "
        f"total_steps={total_steps}, warmup_steps={warmup_steps}"
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    global_batch_step = 0
    last_eval_step = 0
    train_loss_since_eval = 0.0
    train_steps_since_eval = 0
    stop_training = False
    train_step_points = []
    train_step_losses = []
    val_step_points = []
    val_step_losses = []
    val_step_ppls = []
    val_step_base_ppls = []
    train_epoch_points = []
    train_epoch_losses = []

    if args.eval_before_train:
        init_val_loss, init_val_ppl, init_val_base_ppl, init_val_token_acc, init_avg_bases = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            token_base_len_table,
            tokenizer.k,
        )
        writer.add_scalar("Loss/val_step", init_val_loss, 0)
        writer.add_scalar("Perplexity/val_step", init_val_ppl, 0)
        writer.add_scalar("Perplexity/val_base_step", init_val_base_ppl, 0)
        writer.add_scalar("Accuracy/val_token_step", init_val_token_acc, 0)
        best_val_loss = init_val_loss
        best_val_acc = init_val_token_acc
        val_step_points.append(0)
        val_step_losses.append(init_val_loss)
        val_step_ppls.append(init_val_ppl)
        val_step_base_ppls.append(init_val_base_ppl)
        print(
            f"Step 0: val_loss={init_val_loss:.4f}, val_ppl={init_val_ppl:.3f}, "
            f"val_base_ppl={init_val_base_ppl:.3f}, bases/token={init_avg_bases:.2f}, "
            f"val_token_acc={init_val_token_acc:.4f}"
        )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)

        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0

        for batch_idx, (x, y) in enumerate(progress):
            x = x.to(device)
            y = y.to(device)
            global_batch_step += 1

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                if args.unlikelihood_alpha > 0:
                    ul_loss = repetition_unlikelihood_loss(logits, y, ignore_index=IGNORE_INDEX)
                else:
                    ul_loss = ce_loss.new_zeros(())
                loss = ce_loss + args.unlikelihood_alpha * ul_loss

            scaler.scale(loss / args.grad_accum_steps).backward()
            accum_counter += 1

            running_loss += loss.item()
            train_loss_since_eval += loss.item()
            train_steps_since_eval += 1
            writer.add_scalar("Loss/train_batch", loss.item(), global_batch_step)
            writer.add_scalar("Loss/train_batch_ce", ce_loss.item(), global_batch_step)
            if args.unlikelihood_alpha > 0:
                writer.add_scalar("Loss/train_batch_ul", ul_loss.item(), global_batch_step)
            progress.set_postfix(train_batch_loss=f"{loss.item():.4f}", ce=f"{ce_loss.item():.4f}")

            should_step = (accum_counter >= args.grad_accum_steps) or (batch_idx == len(train_loader) - 1)
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

            global_step += 1
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            if global_step % args.eval_interval == 0:
                avg_train_step_loss = train_loss_since_eval / max(1, train_steps_since_eval)
                val_loss, val_ppl, val_base_ppl, val_token_acc, avg_bases = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    use_amp,
                    token_base_len_table,
                    tokenizer.k,
                )

                writer.add_scalar("Loss/train_step", avg_train_step_loss, global_step)
                writer.add_scalar("Loss/val_step", val_loss, global_step)
                writer.add_scalar("Perplexity/val_step", val_ppl, global_step)
                writer.add_scalar("Perplexity/val_base_step", val_base_ppl, global_step)
                writer.add_scalar("Accuracy/val_token_step", val_token_acc, global_step)
                train_step_points.append(global_step)
                train_step_losses.append(avg_train_step_loss)
                val_step_points.append(global_step)
                val_step_losses.append(val_loss)
                val_step_ppls.append(val_ppl)
                val_step_base_ppls.append(val_base_ppl)

                print(
                    f"Step {global_step}: train_loss={avg_train_step_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.3f}, "
                    f"val_base_ppl={val_base_ppl:.3f}, bases/token={avg_bases:.2f}, "
                    f"val_token_acc={val_token_acc:.4f}"
                )

                is_better = (val_loss < best_val_loss) or (
                    abs(val_loss - best_val_loss) < 1e-8 and val_token_acc > best_val_acc
                )
                if is_better:
                    best_val_acc = val_token_acc
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_obj = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "best_val_token_acc": best_val_acc,
                        "args": run_args,
                    }
                    torch.save(save_obj, args.save_path)
                    print(f"Saved best checkpoint to {args.save_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stop_patience:
                        print(f"Early stopping triggered at step {global_step}.")
                        stop_training = True

                train_loss_since_eval = 0.0
                train_steps_since_eval = 0
                last_eval_step = global_step
                model.train()

                if stop_training:
                    break

        train_epoch_loss = running_loss / max(1, len(train_loader))
        writer.add_scalar("Loss/train_epoch", train_epoch_loss, epoch + 1)
        train_epoch_points.append(epoch + 1)
        train_epoch_losses.append(train_epoch_loss)
        print(f"Epoch {epoch + 1}: train_epoch_loss={train_epoch_loss:.4f}")

        if stop_training:
            break

    # If the final step is not exactly on eval boundary, run one last validation.
    if global_step > 0 and last_eval_step != global_step and train_steps_since_eval > 0:
        avg_train_step_loss = train_loss_since_eval / max(1, train_steps_since_eval)
        val_loss, val_ppl, val_base_ppl, val_token_acc, avg_bases = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            token_base_len_table,
            tokenizer.k,
        )
        writer.add_scalar("Loss/train_step", avg_train_step_loss, global_step)
        writer.add_scalar("Loss/val_step", val_loss, global_step)
        writer.add_scalar("Perplexity/val_step", val_ppl, global_step)
        writer.add_scalar("Perplexity/val_base_step", val_base_ppl, global_step)
        writer.add_scalar("Accuracy/val_token_step", val_token_acc, global_step)
        train_step_points.append(global_step)
        train_step_losses.append(avg_train_step_loss)
        val_step_points.append(global_step)
        val_step_losses.append(val_loss)
        val_step_ppls.append(val_ppl)
        val_step_base_ppls.append(val_base_ppl)
        print(
            f"Final Step {global_step}: train_loss={avg_train_step_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.3f}, "
            f"val_base_ppl={val_base_ppl:.3f}, bases/token={avg_bases:.2f}, "
            f"val_token_acc={val_token_acc:.4f}"
        )
        is_better = (val_loss < best_val_loss) or (
            abs(val_loss - best_val_loss) < 1e-8 and val_token_acc > best_val_acc
        )
        if is_better:
            best_val_acc = val_token_acc
            best_val_loss = val_loss
            save_obj = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_val_token_acc": best_val_acc,
                "args": run_args,
            }
            torch.save(save_obj, args.save_path)
            print(f"Saved best checkpoint to {args.save_path}")

    if args.save_plots:
        save_training_plots(
            plot_dir=args.plot_dir,
            train_step_points=train_step_points,
            train_step_losses=train_step_losses,
            val_step_points=val_step_points,
            val_step_losses=val_step_losses,
            val_step_ppls=val_step_ppls,
            val_step_base_ppls=val_step_base_ppls,
            train_epoch_points=train_epoch_points,
            train_epoch_losses=train_epoch_losses,
            dpi=args.plot_dpi,
        )
        print(f"Saved training plots to {os.path.abspath(args.plot_dir)}")

    writer.close()
    print(
        f"Training complete. Best val_token_acc={best_val_acc:.4f}, "
        f"best_val_loss={best_val_loss:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dna_gpt0.1b_h")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/dna_gpt0.1b_h.pth")
    parser.add_argument("--train_file", type=str, default="data/train.txt")
    parser.add_argument("--val_file", type=str, default="data/val.txt")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="checkpoints/dnargpt_finetune_best.pth")
    parser.add_argument("--log_dir", type=str, default="runs/dnargpt_finetune")
    parser.add_argument("--plot_dir", type=str, default="results/finetuning")
    parser.add_argument("--species_token", type=str, default="R")
    parser.add_argument(
        "--adapter_mode",
        type=str,
        default="none",
        choices=["auto", "none", "all", "last4", "last6", "last8", "custom"],
    )
    parser.add_argument("--adapter_layers", type=str, default="")
    parser.add_argument("--adapter_dim", type=int, default=256)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--reinit_top_layers", type=int, default=0)
    parser.add_argument("--unlikelihood_alpha", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--trim_to_k", dest="trim_to_k", action="store_true")
    parser.add_argument("--no_trim_to_k", dest="trim_to_k", action="store_false")
    parser.set_defaults(trim_to_k=True)
    parser.add_argument("--allow_n_base", dest="allow_n_base", action="store_true")
    parser.add_argument("--no_allow_n_base", dest="allow_n_base", action="store_false")
    parser.set_defaults(allow_n_base=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--fused_adamw", action="store_true")
    parser.add_argument("--strict_load", dest="strict_load", action="store_true")
    parser.add_argument("--no_strict_load", dest="strict_load", action="store_false")
    parser.set_defaults(strict_load=True)
    parser.add_argument("--eval_before_train", dest="eval_before_train", action="store_true")
    parser.add_argument("--no_eval_before_train", dest="eval_before_train", action="store_false")
    parser.set_defaults(eval_before_train=True)
    parser.add_argument("--save_plots", dest="save_plots", action="store_true")
    parser.add_argument("--no_save_plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=True)
    parser.add_argument("--plot_dpi", type=int, default=200)
    main(parser.parse_args())
