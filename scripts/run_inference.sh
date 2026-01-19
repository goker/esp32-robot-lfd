#!/bin/bash
# Run trained policy on robot
# Usage: ./scripts/run_inference.sh --checkpoint outputs/bc_policy_best.pt

set -e

# Default values
CHECKPOINT="outputs/bc_policy_best.pt"
EPISODES=1
MAX_STEPS=200
ARM_URL="ws://robot-arm.local:81"
CAMERA_URL="http://robot-cam.local"
SAVE_VIDEOS=false
OUTPUT_DIR="evaluation"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --save_videos)
            SAVE_VIDEOS=true
            shift
            ;;
        --arm_url)
            ARM_URL="$2"
            shift 2
            ;;
        --camera_url)
            CAMERA_URL="$2"
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
echo "ESP32 Robot - Policy Inference"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Episodes: $EPISODES"
echo "Max steps: $MAX_STEPS"
echo "Arm URL: $ARM_URL"
echo "Camera URL: $CAMERA_URL"
echo "Save videos: $SAVE_VIDEOS"
echo "========================================"

VIDEO_FLAG=""
if [ "$SAVE_VIDEOS" = true ]; then
    VIDEO_FLAG="--save_videos"
fi

python -m host.deployment.realtime_control \
    --policy "$CHECKPOINT" \
    --num_episodes "$EPISODES" \
    --max_steps "$MAX_STEPS" \
    --arm_url "$ARM_URL" \
    --camera_url "$CAMERA_URL" \
    --output_dir "$OUTPUT_DIR" \
    $VIDEO_FLAG

echo "Inference complete!"
