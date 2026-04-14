#!/bin/bash
# Day 4: Train FMMotionModel (Flow Matching, "Ours full")
#
# Usage:
#   bash train_fm.sh              # all 7 scenes, sequential
#   bash train_fm.sh Balloon1     # single scene
#   bash train_fm.sh --parallel   # all 7 scenes in parallel (A800 80GB)
#
# Results saved to: output/<SCENE>_fm/
#   point_cloud/fine_best/   <- best PSNR checkpoint
#   point_cloud/iteration_*/ <- periodic saves

MOTION_TYPE="fm"
SMOOTH_W=1e-3
FM_W=0.1
SCENES=(Balloon1 Balloon2 Playground Jumping Truck Skating Umbrella)
# Day 4 target scenes (4 GPUs in parallel)
TARGET_SCENES=(Balloon1 Balloon2 Umbrella Playground)

run_scene() {
    SCENE=$1
    GPU=${2:-0}
    echo "=== [GPU ${GPU}] FMMotionModel: ${SCENE} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        -s ../SplineGS/data/nvidia_rodynrf/${SCENE}/ \
        --expname "${SCENE}_fm" \
        --configs arguments/nvidia_rodynrf/${SCENE}.py \
        --motion_model_type ${MOTION_TYPE} \
        --w_smooth ${SMOOTH_W} \
        --w_fm ${FM_W}
}

if [ $# -eq 1 ] && [ "$1" != "--parallel" ] && [ "$1" != "--day4" ]; then
    # Single scene
    run_scene "$1"
elif [ "$1" == "--day4" ]; then
    # Day 4: Balloon1/Balloon2/Umbrella/Playground, all on GPU 0 (A800 80GB fits 4 scenes)
    PIDS=()
    for SCENE in "${TARGET_SCENES[@]}"; do
        run_scene "${SCENE}" 0 &
        PIDS+=($!)
        sleep 5
    done
    echo "Day4 scenes launched on GPU 0-3. PIDs: ${PIDS[*]}"
    for PID in "${PIDS[@]}"; do
        wait $PID && echo "PID $PID done" || echo "PID $PID failed"
    done
elif [ "$1" == "--parallel" ]; then
    PIDS=()
    for SCENE in "${SCENES[@]}"; do
        run_scene "${SCENE}" 0 &
        PIDS+=($!)
        sleep 5
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
