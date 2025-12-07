#!/usr/bin/env python3
"""Run QAT training across multiple folds."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATASET_ROOT = REPO_ROOT / "ultralytics10.24" / "dataset_root" / "combined_all"


def run_fold(
    fold_num: int,
    epochs: int = 1,
    backend: str = "qnnpack",
    device: str = "0",
    batch: int = None,
    workers: int = None,
    weights: str = None,
    lr0: float = None,
    lrf: float = None,
    warmup_epochs: float = None,
    optimizer: str = None,
    pretrained: bool = None,
    patience: int = None,
    extra_args: list = None,
):
    """Run QAT training for a specific fold."""
    fold_yaml = DATASET_ROOT / f"fold{fold_num}.yaml"

    if not fold_yaml.exists():
        print(f"Error: Fold YAML not found at {fold_yaml}")
        return False

    run_name = f"qat_fold{fold_num}_{backend}"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "train_qat.py"),
        "--epochs", str(epochs),
        "--backend", backend,
        "--device", str(device),
        "--data", str(fold_yaml),
        "--name", run_name,
    ]

    if batch is not None:
        cmd.extend(["--batch", str(batch)])
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    if weights is not None:
        cmd.extend(["--weights", str(weights)])
    if lr0 is not None:
        cmd.extend(["--lr0", str(lr0)])
    if lrf is not None:
        cmd.extend(["--lrf", str(lrf)])
    if warmup_epochs is not None:
        cmd.extend(["--warmup-epochs", str(warmup_epochs)])
    if optimizer is not None:
        cmd.extend(["--optimizer", optimizer])
    if pretrained is not None:
        cmd.extend(["--pretrained", str(pretrained).lower()])
    if patience is not None:
        cmd.extend(["--patience", str(patience)])

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*80}")
    print(f"Starting QAT training for Fold {fold_num}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Fold {fold_num} completed successfully")
        return True
    else:
        print(f"\n✗ Fold {fold_num} failed with return code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run QAT training across multiple folds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 5 folds with 1 epoch each
  python train_qat_folds.py --all --epochs 1 --backend qnnpack

  # Run specific folds
  python train_qat_folds.py --folds 1 2 3 --epochs 5 --backend fbgemm

  # Run fold 1 only with custom settings
  python train_qat_folds.py --folds 1 --epochs 10 --batch 16 --lr0 0.001

  # Run all folds with weights
  python train_qat_folds.py --all --epochs 5 --weights path/to/best.pt
        """,
    )

    # Fold selection
    fold_group = parser.add_mutually_exclusive_group(required=True)
    fold_group.add_argument(
        "--folds",
        type=int,
        nargs="+",
        metavar="N",
        help="Specific fold numbers to run (e.g., --folds 1 2 3)",
    )
    fold_group.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 folds",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs per fold (default: 1)")
    parser.add_argument("--backend", default="fbgemm", choices=["qnnpack", "fbgemm"], help="Quantization backend (default: fbgemm for x86)")
    parser.add_argument("--device", default="0", help="Device to use (default: 0)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate fraction")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer (SGD, Adam, AdamW, etc.)")
    parser.add_argument("--pretrained", type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, help="Use pretrained weights (true/false)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: None)")

    # Control options
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution if any fold fails (default: continue)",
    )

    args, extra_args = parser.parse_known_args()

    # Determine which folds to run
    if args.all:
        folds = [1, 2, 3, 4, 5]
    else:
        folds = sorted(set(args.folds))
        # Validate fold numbers
        invalid_folds = [f for f in folds if f < 1 or f > 5]
        if invalid_folds:
            parser.error(f"Invalid fold numbers: {invalid_folds}. Must be between 1 and 5.")

    print(f"Running QAT training for folds: {folds}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Backend: {args.backend}")
    print(f"Device: {args.device}")

    results = {}
    for fold_num in folds:
        success = run_fold(
            fold_num=fold_num,
            epochs=args.epochs,
            backend=args.backend,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            weights=args.weights,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            optimizer=args.optimizer,
            pretrained=args.pretrained,
            patience=args.patience,
            extra_args=extra_args,
        )
        results[fold_num] = success

        if not success and args.stop_on_error:
            print(f"\nStopping due to error in fold {fold_num}")
            break

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for fold_num, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Fold {fold_num}: {status}")
    print(f"{'='*80}\n")

    # Exit with error if any fold failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
