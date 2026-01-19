#!/bin/bash
# Collect demonstration episodes for a task
# Usage: ./scripts/collect_demo.sh --task pick_place --episodes 50

set -e

# Default values
TASK="pick_place"
EPISODES=10
ARM_URL="ws://robot-arm.local:81"
CAMERA_URL="http://robot-cam.local"
SAVE_DIR="datasets"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --arm_url)
            ARM_URL="$2"
            shift 2
            ;;
        --camera_url)
            CAMERA_URL="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "ESP32 Robot - Data Collection"
echo "========================================"
echo "Task: $TASK"
echo "Episodes: $EPISODES"
echo "Arm URL: $ARM_URL"
echo "Camera URL: $CAMERA_URL"
echo "Save directory: $SAVE_DIR/$TASK"
echo "========================================"

python -m host.data_collection.teleop_recorder \
    --task "$TASK" \
    --episodes "$EPISODES" \
    --arm_url "$ARM_URL" \
    --camera_url "$CAMERA_URL" \
    --save_dir "$SAVE_DIR"

echo "Data collection complete!"
