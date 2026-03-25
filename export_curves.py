import argparse
import math
import os
import re
from pathlib import Path


def lazy_imports():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e
    # Avoid pulling full tensorflow runtime via tensorboard compat path.
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    # Work around protobuf binary compatibility issues on some local setups.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        from tensorboard.backend.event_processing import event_accumulator  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "tensorboard is required to read event files. "
            "If import fails, try: pip install --upgrade tensorboard protobuf==3.20.*"
        ) from e
    return plt, event_accumulator


def sanitize_filename(name: str) -> str:
    name = name.replace("/", "__").replace("\\", "__")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name.strip("._") or "curve"


def find_run_dirs(runs_root: Path):
    run_dirs = set()
    for event_file in runs_root.rglob("events.out.tfevents*"):
        run_dirs.add(event_file.parent)
    return sorted(run_dirs)


def dedupe_by_step(steps, values):
    merged = {}
    for s, v in zip(steps, values):
        merged[int(s)] = float(v)
    sorted_items = sorted(merged.items(), key=lambda x: x[0])
    x = [i[0] for i in sorted_items]
    y = [i[1] for i in sorted_items]
    return x, y


def write_curve(plt, out_dir: Path, tag: str, steps, values, dpi: int, export_csv: bool, filename_stem: str = ""):
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(steps, values, linewidth=1.6)
    ax.set_title(tag)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    stem = filename_stem or sanitize_filename(tag)
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)

    if export_csv:
        csv_path = out_dir / f"{stem}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,value\n")
            for s, v in zip(steps, values):
                f.write(f"{s},{v}\n")


def pick_loss_tag_for_derived_perplexity(tags):
    priority = [
        "Loss/val_step",
        "Loss/val_epoch",
        "Eval/loss",
        "Test/Loss",
        "Loss/val",
    ]
    for t in priority:
        if t in tags:
            return t
    for t in tags:
        tl = t.lower()
        if "loss" in tl and ("val" in tl or "eval" in tl or "test" in tl):
            return t
    for t in tags:
        if "loss" in t.lower():
            return t
    return None


def export_one_run(run_dir: Path, out_dir: Path, dpi: int, export_csv: bool):
    plt, event_accumulator = lazy_imports()
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    if not tags:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    curves = {}
    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue

        steps = [ev.step for ev in events]
        values = [ev.value for ev in events]
        steps, values = dedupe_by_step(steps, values)
        if len(steps) == 0:
            continue

        curves[tag] = (steps, values)
        write_curve(
            plt=plt,
            out_dir=out_dir,
            tag=f"{run_dir.name} | {tag}",
            steps=steps,
            values=values,
            dpi=dpi,
            export_csv=export_csv,
            filename_stem=sanitize_filename(tag),
        )
        n += 1

    # Always export a canonical perplexity curve per run.
    perplexity_tags = [t for t in curves.keys() if "perplexity" in t.lower()]
    if perplexity_tags:
        preferred = None
        for t in perplexity_tags:
            tl = t.lower()
            if "val" in tl or "eval" in tl:
                preferred = t
                break
        if preferred is None:
            preferred = perplexity_tags[0]
        steps, values = curves[preferred]
        write_curve(
            plt=plt,
            out_dir=out_dir,
            tag=f"{run_dir.name} | {preferred} (canonical)",
            steps=steps,
            values=values,
            dpi=dpi,
            export_csv=export_csv,
            filename_stem="perplexity",
        )
        n += 1
    else:
        loss_tag = pick_loss_tag_for_derived_perplexity(list(curves.keys()))
        if loss_tag is not None:
            steps, loss_values = curves[loss_tag]
            ppl_values = [math.exp(min(20.0, float(v))) for v in loss_values]
            write_curve(
                plt=plt,
                out_dir=out_dir,
                tag=f"{run_dir.name} | Derived Perplexity from {loss_tag}",
                steps=steps,
                values=ppl_values,
                dpi=dpi,
                export_csv=export_csv,
                filename_stem="perplexity",
            )
            n += 1

    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--output_root", type=str, default="results/curves")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--export_csv", action="store_true")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        print(f"No event files found under {runs_root}")
        return

    total = 0
    for run_dir in run_dirs:
        rel = run_dir.relative_to(runs_root)
        out_dir = output_root / rel
        exported = export_one_run(run_dir, out_dir, dpi=args.dpi, export_csv=args.export_csv)
        total += exported
        print(f"[{run_dir}] exported {exported} curves -> {out_dir}")

    print(f"Done. Total exported curves: {total}")


if __name__ == "__main__":
    main()
