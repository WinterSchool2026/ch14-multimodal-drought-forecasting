#!/bin/bash
# Train and evaluate ContextUNet on both regions
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "Training ContextUNet on africa"
echo "=========================================="
python train.py --model ContextUNet --region africa --epochs 80 --lr 5e-4 --patience 15

echo ""
echo "=========================================="
echo "Training ContextUNet on latam"
echo "=========================================="
python train.py --model ContextUNet --region latam --epochs 80 --lr 5e-4 --patience 15

echo ""
echo "=========================================="
echo "Evaluating all 4 transfer conditions"
echo "=========================================="
for train_region in africa latam; do
    for test_region in africa latam; do
        echo ""
        echo "--- ${train_region} -> ${test_region} ---"
        python evaluate.py --model ContextUNet --train-region ${train_region} --test-region ${test_region}
    done
done

echo ""
echo "Done!"
