# QAT Training Guide - AWS Persistent Training

## Current Training Status

**Training Configuration:**
- **Epochs**: 300
- **Learning Rate**: lr0=2e-4, lrf=2e-4
- **Image Size**: 640
- **Device**: GPU 0
- **Log File**: `qat_300_epochs_lr2e4.log`
- **Training Directory**: `runs/detect/train_qat17/`

## Training is Running in Background

The training was started with `nohup`, which means it will **continue running even after you disconnect from SSH or close your IDE**.

## How to Monitor Training

### 1. Check Training Status
```bash
cd /home/ubuntu/obc-yolov8
./check_training.sh
```

### 2. View Live Training Log
```bash
tail -f qat_300_epochs_lr2e4.log
```
Press `Ctrl+C` to exit the live view.

### 3. View Latest Training Progress
```bash
tail -n 50 qat_300_epochs_lr2e4.log
```

### 4. Check GPU Usage
```bash
nvidia-smi
```

### 5. Check Training Process
```bash
ps aux | grep train_qat.py | grep -v grep
```

## How to Check Training Results

### View Checkpoints
```bash
ls -lht runs/detect/train_qat*/weights/*.pt
```

### View Training Metrics
```bash
# View results CSV
cat runs/detect/train_qat17/results.csv | tail -n 20

# View training plots (if generated)
ls -lh runs/detect/train_qat17/*.png
```

## How to Stop Training (if needed)

```bash
# Find the process ID
ps aux | grep train_qat.py | grep -v grep

# Kill the training process (replace PID with actual process ID)
kill <PID>

# Or kill all training processes
pkill -f train_qat.py
```

## How to Resume Training (if interrupted)

If training stops unexpectedly, you can resume from the last checkpoint:

```bash
cd /home/ubuntu/obc-yolov8
source .venv/bin/activate
python train_qat.py --epochs 300 --device 0 --imgsz 640 --lr0 2e-4 --lrf 2e-4 --resume --weights runs/detect/train_qat17/weights/last.pt
```

## Expected Training Time

- **300 epochs** with ~246 batches per epoch
- Estimated time: **~12-15 hours** (depending on GPU and data loading)
- Checkpoints are saved automatically to `runs/detect/train_qat17/weights/`

## Output Files

After training completes, you'll find:
- `runs/detect/train_qat17/weights/best.pt` - Best FP32 checkpoint
- `runs/detect/train_qat17/weights/best_qat.pt` - Best QAT checkpoint (with FakeQuantize)
- `runs/detect/train_qat17/weights/last.pt` - Last epoch checkpoint
- `runs/detect/train_qat17/results.csv` - Training metrics
- `qat_300_epochs_lr2e4.log` - Full training log

## Reconnecting to AWS

When you reconnect to your AWS instance:

1. SSH back into your instance
2. Navigate to the project: `cd /home/ubuntu/obc-yolov8`
3. Check if training is still running: `./check_training.sh`
4. View the log: `tail -f qat_300_epochs_lr2e4.log`

The training will continue running in the background even if you disconnect!

## Notes

- The training uses `nohup` which makes it immune to SSH disconnections
- All output is logged to `qat_300_epochs_lr2e4.log`
- Checkpoints are saved periodically (best model and last epoch)
- GPU memory usage is around 4GB (Tesla T4 with 15GB total)

