#!/bin/bash

# Configuration
MODEL_NAME="mistral"
MODEL_PATH="../../models/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR="./${MODEL_NAME}_gpqa"
LOG_DIR="./logs"
TOTAL_SPLITS=3

ERROR_SPLITS=(2)

# Manually set GPU IDs for each split
GPUS=(0)    # Modify these numbers as needed

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Launch processes for each split
# for i in "${ERROR_SPLITS[@]}"; do
for ((i=0; i<1; i++)); do

    GPU=${GPUS[$i]}
    LOG_FILE="$LOG_DIR/${MODEL_NAME}_split_$i.txt"
    
    CUDA_VISIBLE_DEVICES=$GPU python -u run_split.py \
        --split_number $i \
        --total_splits $TOTAL_SPLITS \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        > "$LOG_FILE" 2>&1 &
    
    echo "Started split $i on GPU $GPU. Logs: $LOG_FILE"
done

# Wait for all processes to complete
wait

echo "All splits completed. Check logs in $LOG_DIR"


# MODEL_NAME="llama1b"
# MODEL_PATH="../../models/Llama-3.2-1B-Instruct" # Keep your model path
# DATASET_PATH="../data/gpqa_test_with_rephrased_inputs_v2"   # <<< IMPORTANT: SET THIS PATH to your saved DatasetDict
# OUTPUT_DIR="./${MODEL_NAME}_gpqa"             # Output directory for this evaluation
# LOG_DIR="./logs"                  # Log directory for this evaluation
# TASK_NAME="Domain_Knowledge_Test"
# TOTAL_SPLITS=8  # Number of splits to divide the dataset into

# # Create output directory
# mkdir -p $OUTPUT_DIR

# # Run evaluations in parallel across GPUs
# GPU_IDS=(3 3 3 3 2 2 2 2)  # Example: split 0 on GPU 0, split 1 on GPU 1, etc.

# # Run evaluations in parallel across GPUs
# for split in $(seq 0 $((TOTAL_SPLITS-1))); do
#     # Get the custom GPU ID for this split
#     GPU_ID=${GPU_IDS[$split]}
    
#     # Launch job in background
#     python gpqa_inf.py \
#         --dataset_path $DATASET_PATH \
#         --output_dir $OUTPUT_DIR \
#         --model_path $MODEL_PATH \
#         --task_name $TASK_NAME \
#         --gpu_id $GPU_ID \
#         --split_num $split \
#         --total_splits $TOTAL_SPLITS \
#         > $LOG_DIR/split${split}_${MODEL_NAME}.log 2>&1 &
    
#     echo "Launched evaluation for split $split on GPU $GPU_ID"
    
#     # Add a small delay to prevent GPU memory issues
#     sleep 1
# done

# # Wait for all background jobs to finish
# wait
# echo "All evaluations completed. Check the logs in $OUTPUT_DIR"