# Learning Rate Issue - Explanation and Fix

## Problem Identified

The training results showed incorrect learning rates:
- **Expected**: `lr0=2e-4` (0.0002)
- **Actual in optimizer**: `lr=0.01` (default)
- **Actual in results.csv**: Learning rates around 0.0099 (close to 0.01)

## Root Cause

When `optimizer='auto'` is used in Ultralytics YOLO, the `build_optimizer` function has this logic:

```python
if name == 'auto':
    nc = getattr(model, 'nc', 10)
    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
```

**The problem**: When iterations > 10000 (which is true for 300 epochs), it **hardcodes** `lr=0.01` and **completely ignores** the `lr0` parameter you passed!

## Solution

**Explicitly set the optimizer** instead of using `'auto'`:

```bash
python train_qat.py --epochs 300 --device 0 --imgsz 640 --lr0 2e-4 --lrf 2e-4 --optimizer SGD
```

This ensures that:
1. The optimizer is SGD (not auto-selected)
2. The learning rate `lr0=2e-4` is **actually used** instead of being overridden
3. The optimizer will initialize with `lr=0.0002` instead of `lr=0.01`

## Verification

After restarting with `--optimizer SGD`, check the log:
```bash
grep "optimizer:" qat_300_epochs_lr2e4_fixed.log
```

You should see:
```
optimizer: SGD(lr=0.0002, momentum=0.9) ...
```

Instead of:
```
optimizer: SGD(lr=0.01, momentum=0.9) ...
```

## Impact on Training

The previous training (epochs 1-16) was using:
- **Learning rate**: ~0.01 (50x higher than intended!)
- **Warmup**: Scaling from 0 to 0.01 over 3 epochs
- **Final LR**: ~0.01 * 0.0002 = 0.000002 (very small)

This explains why:
1. The metrics were very low (precision ~0.0007, mAP50 ~0.0005)
2. The learning rate in results.csv was around 0.0099
3. Training was likely unstable or not learning properly

## Corrected Training

The new training with `--optimizer SGD` will use:
- **Initial LR**: 0.0002 (2e-4)
- **Warmup**: Scaling from 0 to 0.0002 over 3 epochs
- **Final LR**: 0.0002 * 0.0002 = 0.00000004 (4e-8)

This is a much more reasonable learning rate schedule for QAT fine-tuning!

