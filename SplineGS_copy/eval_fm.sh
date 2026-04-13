#!/bin/bash
# Eval FMMotionModel results (output/<SCENE>_fm/)
# Usage:
#   bash eval_fm.sh              # all scenes with fine_best
#   bash eval_fm.sh Balloon1     # single scene

SCENES=(Balloon1 Balloon2 Playground Jumping Truck Skating Umbrella)

eval_scene() {
    SCENE=$1
    CHECKPOINT="output/${SCENE}_fm/point_cloud/fine_best"
    if [ ! -d "$CHECKPOINT" ]; then
        echo "[SKIP] $SCENE: checkpoint not found at $CHECKPOINT"
        return
    fi
    echo "=== Evaluating: $SCENE ==="
    python eval_nvidia.py \
        -s ../SplineGS/data/nvidia_rodynrf/${SCENE}/ \
        --expname "${SCENE}_fm" \
        --configs arguments/nvidia_rodynrf/${SCENE}.py \
        --checkpoint "$CHECKPOINT" \
        --motion_model_type fm
}

if [ $# -eq 1 ]; then
    eval_scene "$1"
else
    for SCENE in "${SCENES[@]}"; do
        eval_scene "$SCENE"
    done
fi
