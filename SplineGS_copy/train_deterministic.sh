#!/bin/bash
# Day 3: Train DeterministicMotionModel (Ours w/o FM baseline)
#
# Usage:
#   bash train_deterministic.sh              # all 7 scenes, sequential
#   bash train_deterministic.sh Balloon1     # single scene
#   bash train_deterministic.sh --parallel   # all 7 scenes in parallel (A800 80GB)
#
# Results saved to: output/<SCENE>_det/
#   point_cloud/fine_best/   <- best PSNR checkpoint
#   point_cloud/iteration_*/ <- periodic saves

MOTION_TYPE="deterministic"
SMOOTH_W=1e-3
SCENES=(Balloon1 Balloon2 Playground Jumping Truck Skating Umbrella)

run_scene() {
    SCENE=$1
    GPU=${2:-0}
    echo "=== [GPU ${GPU}] DeterministicMotionModel: ${SCENE} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        -s data/nvidia_rodynrf/${SCENE}/ \
        --expname "${SCENE}_det" \
        --configs arguments/nvidia_rodynrf/${SCENE}.py \
        --motion_model_type ${MOTION_TYPE} \
        --w_smooth ${SMOOTH_W}
}

if [ $# -eq 1 ] && [ "$1" != "--parallel" ]; then
    # Single scene
    run_scene "$1"
elif [ "$1" == "--parallel" ]; then
    # Parallel: each scene on GPU 0 (A800 has 80GB, 7 scenes * ~8GB = ~56GB fits)
    # Adjust CUDA_VISIBLE_DEVICES if you have multiple GPUs
    PIDS=()
    for SCENE in "${SCENES[@]}"; do
        run_scene "${SCENE}" 0 &
        PIDS+=($!)
        sleep 5  # stagger launches to avoid init collisions
    done
    echo "All scenes launched. PIDs: ${PIDS[*]}"
    for PID in "${PIDS[@]}"; do
        wait $PID && echo "PID $PID done" || echo "PID $PID failed"
    done
else
    # Sequential (default)
    for SCENE in "${SCENES[@]}"; do
        run_scene "${SCENE}"
    done
fi
