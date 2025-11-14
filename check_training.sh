#!/bin/bash
# Helper script to check training status

echo "=== Training Process Status ==="
ps aux | grep 'train_qat.py' | grep -v grep

echo ""
echo "=== Latest Training Log (last 30 lines) ==="
tail -n 30 qat_300_epochs_lr2e4.log 2>/dev/null || echo "Log file not found yet"

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== Training Directory ==="
ls -lht runs/detect/train_qat*/weights/*.pt 2>/dev/null | head -n 5 || echo "No checkpoints found yet"

