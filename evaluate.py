import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from dna_gpt.model.dna_gpt import DNAGPT
from dna_gpt.tokenizer import KmerTokenizer

try:
    from dna_gpt.model.adapter_lm import build_lm_model
except ModuleNotFoundError:
    _adapter_path = Path(__file__).resolve().parent / "dna_gpt" / "model" / "adapter_lm.py"
    if _adapter_path.exists():
        _spec = importlib.util.spec_from_file_location("dna_gpt.model.adapter_lm", _adapter_path)
        _module = importlib.util.module_from_spec(_spec)
        assert _spec is not None and _spec.loader is not None
        _spec.loader.exec_module(_module)
        build_lm_model = _module.build_lm_model
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
DNA_BASES = ("A", "T", "C", "G")


def build_tokenizer(model_name: str) -> KmerTokenizer:
    dynamic_kmer = model_name != "dna_gpt0.1b_h"
    return KmerTokenizer(k=6, reserved_tokens=SPECIAL_TOKENS, dynamic_kmer=dynamic_kmer)


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


def auto_num_workers(num_workers: int) -> int:
    if num_workers >= 0:
        return num_workers
    cpu_count = os.cpu_count() or 4
    return min(8, max(1, cpu_count // 2))


def default_cache_path(args) -> str:
    base = (
        f"{os.path.abspath(args.data_file)}|{args.model_name}|"
        f"{args.max_tokens}|{args.species_token}|{args.trim_to_k}|{args.allow_n_base}"
    )
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(os.path.dirname(args.data_file), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"eval_tokens_{digest}.pt")


class KmerNextTokenDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: KmerTokenizer,
        max_tokens: int,
        species_token: str,
        trim_to_k: bool,
        allow_n_base: bool,
        cache_file: str = "",
        rebuild_cache: bool = False,
    ):
        self.samples = []
        if cache_file and os.path.exists(cache_file) and not rebuild_cache:
            self.samples = torch.load(cache_file, map_location="cpu")
            if not isinstance(self.samples, list):
                raise ValueError(f"Invalid cache format: {cache_file}")
            return

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

        if not self.samples:
            raise ValueError(f"No valid samples found in {file_path}")

        if cache_file:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(self.samples, cache_file)

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


def preview_piece_sequence(tokenizer, token_ids, max_tokens=24):
    ids = token_ids[:max_tokens].tolist()
    pieces = [tokenizer.id_to_piece(tid) for tid in ids]
    return " ".join(pieces)


def token_ids_to_bases(tokenizer: KmerTokenizer, token_ids) -> str:
    pieces = [tokenizer.id_to_piece(int(tid)) for tid in token_ids]
    seq = "".join(pieces)
    if "<" in seq and ">" in seq:
        seq = re.sub(r"<[^>]*>", "", seq)
    return "".join(ch for ch in seq if ch in "ATCGN")


def build_token_base_lengths(tokenizer: KmerTokenizer):
    lengths = []
    for tid in range(len(tokenizer)):
        piece = tokenizer.id_to_piece(tid)
        if piece.startswith("<") and piece.endswith(">"):
            lengths.append(0)
        else:
            lengths.append(sum(ch in "ATCGN" for ch in piece))
    return lengths


def base_frequency(sequences):
    counter = Counter()
    for seq in sequences:
        counter.update(ch for ch in seq if ch in DNA_BASES)
    total = sum(counter[b] for b in DNA_BASES)
    if total == 0:
        return {b: 0.0 for b in DNA_BASES}
    return {b: counter[b] / total for b in DNA_BASES}


def gc_content(seq: str) -> float:
    valid = sum(ch in "ATCG" for ch in seq)
    if valid == 0:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return gc / valid


def kmer_frequency(sequences, k=3):
    counter = Counter()
    for seq in sequences:
        seq = "".join(ch for ch in seq if ch in "ATCG")
        if len(seq) < k:
            continue
        for i in range(len(seq) - k + 1):
            counter[seq[i : i + k]] += 1
    total = sum(counter.values())
    if total == 0:
        return {}
    return {kmer: v / total for kmer, v in counter.items()}


def save_eval_plots(
    plot_dir: str,
    batch_losses,
    batch_ppls,
    batch_base_ppls,
    seq_accs,
    real_sequences,
    pred_sequences,
    avg_loss: float,
    perplexity: float,
    base_perplexity: float,
    effective_token_bases: float,
    token_acc: float,
    tokens_per_sec: float,
    dpi: int,
):
    os.makedirs(plot_dir, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skip plot export: matplotlib is not installed ({exc}).")
        return

    steps = list(range(1, len(batch_losses) + 1))
    if not steps:
        steps = [1]
        batch_losses = [avg_loss]
        batch_ppls = [perplexity]
        batch_base_ppls = [base_perplexity]

    # test loss.png
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(steps, batch_losses, color="#1f77b4", linewidth=2.0)
    ax.set_title("Test Loss")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Cross Entropy")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "test loss.png"), dpi=dpi)
    plt.close(fig)

    # test perplexity.png
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(steps, batch_ppls, color="#d62728", linewidth=2.0, label="Token PPL")
    ax.plot(
        steps,
        batch_base_ppls,
        color="#2ca02c",
        linewidth=2.0,
        label=f"Base-equivalent PPL (~{effective_token_bases:.2f} bases/token)",
    )
    ax.set_title("Test Perplexity")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Perplexity")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "test perplexity.png"), dpi=dpi)
    plt.close(fig)

    # perplexity_intuition.png
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.bar(
        ["Token PPL", "Base-eq PPL"],
        [perplexity, base_perplexity],
        color=["#d62728", "#2ca02c"],
    )
    ax.set_title("Perplexity Intuition")
    ax.set_ylabel("Perplexity")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "perplexity_intuition.png"), dpi=dpi)
    plt.close(fig)

    # prediction_accuracy.png
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    if seq_accs:
        ax.hist(seq_accs, bins=20, color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axvline(token_acc, color="#d62728", linestyle="--", linewidth=2.0, label=f"mean={token_acc:.4f}")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No sequence-level accuracy data", ha="center", va="center")
    ax.set_title("Prediction Accuracy Distribution")
    ax.set_xlabel("Sequence Token Accuracy")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "prediction_accuracy.png"), dpi=dpi)
    plt.close(fig)

    real_base_freq = base_frequency(real_sequences)
    pred_base_freq = base_frequency(pred_sequences)

    # dna frequency.png
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = list(range(len(DNA_BASES)))
    width = 0.35
    real_base_vals = [real_base_freq[b] for b in DNA_BASES]
    pred_base_vals = [pred_base_freq[b] for b in DNA_BASES]
    ax.bar([i - width / 2 for i in x], real_base_vals, width=width, label="Real", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], pred_base_vals, width=width, label="Predicted", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(DNA_BASES)
    ax.set_title("DNA Base Frequency")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "dna frequency.png"), dpi=dpi)
    plt.close(fig)

    real_gc = [gc_content(s) for s in real_sequences if s]
    pred_gc = [gc_content(s) for s in pred_sequences if s]
    real_gc_mean = sum(real_gc) / max(1, len(real_gc))
    pred_gc_mean = sum(pred_gc) / max(1, len(pred_gc))

    # gc_content_comparison.png
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.bar(["Real", "Predicted"], [real_gc_mean, pred_gc_mean], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("GC Content Comparison")
    ax.set_ylabel("GC Ratio")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "gc_content_comparison.png"), dpi=dpi)
    plt.close(fig)

    # gc_content_distirbution.png (keep legacy file name)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    if real_gc:
        ax.hist(real_gc, bins=20, alpha=0.6, label="Real", color="#1f77b4")
    if pred_gc:
        ax.hist(pred_gc, bins=20, alpha=0.6, label="Predicted", color="#ff7f0e")
    ax.set_title("GC Content Distribution")
    ax.set_xlabel("GC Ratio")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "gc_content_distirbution.png"), dpi=dpi)
    plt.close(fig)

    # top20_3mers.png
    real_3mer = kmer_frequency(real_sequences, k=3)
    pred_3mer = kmer_frequency(pred_sequences, k=3)
    top_kmers = [k for k, _ in sorted(real_3mer.items(), key=lambda kv: kv[1], reverse=True)[:20]]
    if len(top_kmers) < 20:
        for kmer, _ in sorted(pred_3mer.items(), key=lambda kv: kv[1], reverse=True):
            if kmer not in top_kmers:
                top_kmers.append(kmer)
                if len(top_kmers) == 20:
                    break

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(111)
    if top_kmers:
        x = list(range(len(top_kmers)))
        width = 0.38
        real_kmer_vals = [real_3mer.get(kmer, 0.0) for kmer in top_kmers]
        pred_kmer_vals = [pred_3mer.get(kmer, 0.0) for kmer in top_kmers]
        ax.bar([i - width / 2 for i in x], real_kmer_vals, width=width, label="Real", color="#1f77b4")
        ax.bar([i + width / 2 for i in x], pred_kmer_vals, width=width, label="Predicted", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(top_kmers, rotation=60, ha="right")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No 3-mer statistics available", ha="center", va="center")
    ax.set_title("Top 20 3-mer Frequency")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top20_3mers.png"), dpi=dpi)
    plt.close(fig)

    # result1.png
    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(steps, batch_losses, color="#1f77b4", linewidth=2.0)
    ax1.set_title("Test Loss")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax2.plot(steps, batch_ppls, color="#d62728", linewidth=2.0)
    ax2.set_title("Test Perplexity")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("PPL")
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    if seq_accs:
        ax3.hist(seq_accs, bins=20, color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax3.set_title("Seq Token Acc")
    ax3.set_xlabel("Accuracy")
    ax3.set_ylabel("Count")
    ax3.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "result1.png"), dpi=dpi)
    plt.close(fig)

    # result2.png
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)
    x = list(range(len(DNA_BASES)))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], real_base_vals, width=width, label="Real", color="#1f77b4")
    ax1.bar([i + width / 2 for i in x], pred_base_vals, width=width, label="Predicted", color="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(DNA_BASES)
    ax1.set_title("Base Frequency")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
    ax2.bar(["Real", "Pred"], [real_gc_mean, pred_gc_mean], color=["#1f77b4", "#ff7f0e"])
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("GC Mean")
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
    if top_kmers:
        top_small = top_kmers[:10]
        xs = list(range(len(top_small)))
        rv = [real_3mer.get(kmer, 0.0) for kmer in top_small]
        pv = [pred_3mer.get(kmer, 0.0) for kmer in top_small]
        ax3.plot(xs, rv, marker="o", color="#1f77b4", label="Real")
        ax3.plot(xs, pv, marker="o", color="#ff7f0e", label="Pred")
        ax3.set_xticks(xs)
        ax3.set_xticklabels(top_small, rotation=45, ha="right")
        ax3.legend(loc="best")
    ax3.set_title("Top 10 3-mer Trend")
    ax3.set_ylabel("Frequency")
    ax3.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "result2.png"), dpi=dpi)
    plt.close(fig)

    metrics = {
        "loss": float(avg_loss),
        "perplexity": float(perplexity),
        "base_perplexity": float(base_perplexity),
        "avg_bases_per_token": float(effective_token_bases),
        "token_accuracy": float(token_acc),
        "tokens_per_sec": float(tokens_per_sec),
        "batches": int(len(batch_losses)),
        "num_real_sequences": int(len(real_sequences)),
        "num_pred_sequences": int(len(pred_sequences)),
    }
    with open(os.path.join(plot_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


@torch.inference_mode()
def main(args):
    if args.eval_max_batches < 0:
        raise ValueError("--eval_max_batches must be >= 0")

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    raw_checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt_args = raw_checkpoint.get("args", {}) if isinstance(raw_checkpoint, dict) else {}
    model_state = raw_checkpoint["model"] if isinstance(raw_checkpoint, dict) and "model" in raw_checkpoint else raw_checkpoint

    model_name = args.model_name or ckpt_args.get("model_name", "dna_gpt0.1b_h")
    tokenizer = build_tokenizer(model_name)
    backbone = DNAGPT.from_name(model_name, vocab_size=len(tokenizer))

    adapter_mode = args.adapter_mode
    if adapter_mode == "auto":
        adapter_mode = ckpt_args.get("adapter_mode", "none")
    if adapter_mode == "auto":
        adapter_mode = "none"
    adapter_layers = args.adapter_layers
    if adapter_layers == "auto":
        adapter_layers = ckpt_args.get("adapter_layers", "")
    adapter_dim = args.adapter_dim if args.adapter_dim > 0 else int(ckpt_args.get("adapter_dim", 256))
    adapter_dropout = (
        args.adapter_dropout if args.adapter_dropout >= 0 else float(ckpt_args.get("adapter_dropout", 0.1))
    )
    model, active_layers = build_lm_model(
        backbone=backbone,
        adapter_mode=adapter_mode,
        adapter_layers=adapter_layers,
        adapter_dim=adapter_dim,
        adapter_dropout=adapter_dropout,
    )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print("Missing key sample:", missing[:10])
    if unexpected:
        print("Unexpected key sample:", unexpected[:10])
    if args.strict_load and (len(missing) > 0 or len(unexpected) > 0):
        raise RuntimeError("Checkpoint keys mismatch. Use --no_strict_load to bypass.")
    print(f"Loaded model_name={model_name}, adapter_mode={adapter_mode}, adapter_layers={active_layers}")
    model = model.to(device).eval()

    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=args.compile_mode)

    cache_file = args.cache_file or default_cache_path(args)
    dataset = KmerNextTokenDataset(
        args.data_file,
        tokenizer,
        max_tokens=args.max_tokens,
        species_token=args.species_token,
        trim_to_k=args.trim_to_k,
        allow_n_base=args.allow_n_base,
        cache_file=cache_file if args.use_cache else "",
        rebuild_cache=args.rebuild_cache,
    )

    num_workers = auto_num_workers(args.num_workers)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**loader_kwargs)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    writer = SummaryWriter(log_dir=args.log_dir)
    token_base_len_table = torch.tensor(build_token_base_lengths(tokenizer), dtype=torch.float32, device=device)

    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    total_base_tokens = 0.0
    shown = 0
    seen_batches = 0
    start_time = time.perf_counter()
    batch_losses = []
    batch_ppls = []
    batch_base_ppls = []
    seq_accs = []
    real_sequences = []
    pred_sequences = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        batch_ppls.append(math.exp(min(20.0, batch_loss)))
        preds = torch.argmax(logits, dim=-1)
        valid_mask = y.ne(IGNORE_INDEX)
        valid_tokens = valid_mask.sum().item()
        base_tokens = token_base_len_table[y[valid_mask]].sum().item() if valid_tokens > 0 else 0.0
        total_correct += (preds.eq(y) & valid_mask).sum().item()
        total_valid += valid_tokens
        total_base_tokens += base_tokens
        seen_batches += 1
        if valid_tokens > 0 and base_tokens > 0:
            batch_bases_per_token = base_tokens / valid_tokens
            batch_base_ppls.append(math.exp(min(20.0, batch_loss / max(1e-6, batch_bases_per_token))))
        else:
            batch_base_ppls.append(math.exp(min(20.0, batch_loss / max(1, tokenizer.k))))

        # Keep sequence-level stats for plot export, but cap to avoid extra overhead.
        if args.save_plots and len(real_sequences) < args.plot_max_sequences:
            for i in range(x.size(0)):
                mask = valid_mask[i]
                if not torch.any(mask):
                    continue
                seq_acc = (preds[i][mask].eq(y[i][mask])).float().mean().item()
                seq_accs.append(seq_acc)
                real_seq = token_ids_to_bases(tokenizer, y[i][mask].detach().cpu().tolist())
                pred_seq = token_ids_to_bases(tokenizer, preds[i][mask].detach().cpu().tolist())
                if real_seq:
                    real_sequences.append(real_seq)
                if pred_seq:
                    pred_sequences.append(pred_seq)
                if len(real_sequences) >= args.plot_max_sequences:
                    break

        if shown < args.num_preview:
            for i in range(x.size(0)):
                if shown >= args.num_preview:
                    break
                mask = valid_mask[i]
                seq_acc = (preds[i][mask].eq(y[i][mask])).float().mean().item()
                print(f"[Preview {shown + 1}] seq_token_acc={seq_acc:.4f}")
                print("input:", preview_piece_sequence(tokenizer, x[i][mask]))
                print("pred :", preview_piece_sequence(tokenizer, preds[i][mask]))
                print("label:", preview_piece_sequence(tokenizer, y[i][mask]))
                print("-" * 80)
                shown += 1

        if args.eval_max_batches > 0 and seen_batches >= args.eval_max_batches:
            break

    elapsed = max(1e-6, time.perf_counter() - start_time)
    avg_loss = total_loss / max(1, seen_batches)
    perplexity = math.exp(min(20.0, avg_loss))
    avg_bases_per_token = total_base_tokens / max(1, total_valid)
    if avg_bases_per_token <= 0:
        avg_bases_per_token = float(max(1, tokenizer.k))
    base_perplexity = math.exp(min(20.0, avg_loss / max(1e-6, avg_bases_per_token)))
    token_acc = total_correct / max(1, total_valid)
    tokens_per_sec = total_valid / elapsed

    writer.add_scalar("Eval/loss", avg_loss, 0)
    writer.add_scalar("Eval/perplexity", perplexity, 0)
    writer.add_scalar("Eval/base_perplexity", base_perplexity, 0)
    writer.add_scalar("Eval/token_accuracy", token_acc, 0)
    writer.add_scalar("Eval/tokens_per_sec", tokens_per_sec, 0)
    writer.close()

    if args.save_plots:
        save_eval_plots(
            plot_dir=args.plot_dir,
            batch_losses=batch_losses,
            batch_ppls=batch_ppls,
            batch_base_ppls=batch_base_ppls,
            seq_accs=seq_accs,
            real_sequences=real_sequences,
            pred_sequences=pred_sequences,
            avg_loss=avg_loss,
            perplexity=perplexity,
            base_perplexity=base_perplexity,
            effective_token_bases=avg_bases_per_token,
            token_acc=token_acc,
            tokens_per_sec=tokens_per_sec,
            dpi=args.plot_dpi,
        )
        print(f"Saved eval plots to: {os.path.abspath(args.plot_dir)}")

    print(f"Device: {device}")
    print(f"Batches Evaluated: {seen_batches}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Base-equivalent Perplexity: {base_perplexity:.4f} (avg {avg_bases_per_token:.2f} bases/token)")
    print(f"Token Accuracy: {token_acc:.4f}")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    if args.use_cache:
        print(f"Token cache: {cache_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dna_gpt0.1b_h")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/dnargpt_finetune_best.pth")
    parser.add_argument(
        "--adapter_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "all", "last4", "last6", "last8", "custom"],
    )
    parser.add_argument("--adapter_layers", type=str, default="auto")
    parser.add_argument("--adapter_dim", type=int, default=-1)
    parser.add_argument("--adapter_dropout", type=float, default=-1.0)
    parser.add_argument("--strict_load", dest="strict_load", action="store_true")
    parser.add_argument("--no_strict_load", dest="strict_load", action="store_false")
    parser.set_defaults(strict_load=True)
    parser.add_argument("--data_file", type=str, default="data/test.txt")
    parser.add_argument("--log_dir", type=str, default="runs/dnargpt_eval")
    parser.add_argument("--plot_dir", type=str, default="results/eval")
    parser.add_argument("--species_token", type=str, default="R")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--trim_to_k", dest="trim_to_k", action="store_true")
    parser.add_argument("--no_trim_to_k", dest="trim_to_k", action="store_false")
    parser.set_defaults(trim_to_k=True)
    parser.add_argument("--allow_n_base", dest="allow_n_base", action="store_true")
    parser.add_argument("--no_allow_n_base", dest="allow_n_base", action="store_false")
    parser.set_defaults(allow_n_base=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--num_preview", type=int, default=0)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="max-autotune", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--cache_file", type=str, default="")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--save_plots", dest="save_plots", action="store_true")
    parser.add_argument("--no_save_plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=True)
    parser.add_argument("--plot_dpi", type=int, default=200)
    parser.add_argument("--plot_max_sequences", type=int, default=2000)
    main(parser.parse_args())
