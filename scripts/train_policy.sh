#!/bin/bash
# Train a policy on collected demonstrations
# Usage: ./scripts/train_policy.sh --policy bc --dataset datasets/pick_place

set -e

# Default values
POLICY="bc"
DATASET="datasets/pick_place"
EPOCHS=100
BATCH_SIZE=32
LR=0.0001
OUTPUT_DIR="outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)
            POLICY="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "ESP32 Robot - Policy Training"
echo "========================================"
echo "Policy: $POLICY"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Output: $OUTPUT_DIR"
echo "========================================"

case $POLICY in
    bc)
        python -m host.training.behavioral_cloning \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --output_dir "$OUTPUT_DIR"
        ;;
    diffusion)
        python -m host.training.diffusion_policy \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --output_dir "$OUTPUT_DIR"
        ;;
    act)
        python -m host.training.act_policy \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --output_dir "$OUTPUT_DIR"
        ;;
    *)
        echo "Unknown policy type: $POLICY"
        echo "Available: bc, diffusion, act"
        exit 1
        ;;
esac

echo "Training complete!"
