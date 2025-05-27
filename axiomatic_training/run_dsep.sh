#!/bin/bash

NUM_EPOCHS=100
POS_ENCODING="none"
CKPT_FREQ=5
NUM_LAYERS=12
NUM_HEADS=8
HIDDEN_SIZE=512
LEARNING_RATE=1e-4
SEED=42
TRAIN_DATA="train_files/dsep_instances_formatted_175k_train_dict.pkl"
VAL_DATA="train_files/dsep_instances_formatted_25k_val_dict.pkl"

# Parse named arguments
while getopts "e:p:l:h:c:d:s:" opt; do
    case $opt in
        e) NUM_EPOCHS="$OPTARG";;
        p) POS_ENCODING="$OPTARG";;
        l) NUM_LAYERS="$OPTARG";;
        h) NUM_HEADS="$OPTARG";;
        c) HIDDEN_SIZE="$OPTARG";;
        r) LEARNING_RATE="$OPTARG";;
        s) SEED="$OPTARG";;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    esac
done
CHECKPOINT_DIR="checkpoints/DSEP_v2_${POS_ENCODING}_${NUM_EPOCHS}EPOCHS_${NUM_LAYERS}LAYERS_${NUM_HEADS}HEADS_${HIDDEN_SIZE}HIDDEN_SIZE_${SEED}SEED"

python -m axiomatic_training.pretrain --train_data ${TRAIN_DATA} --num_epochs ${NUM_EPOCHS} --pos_encoding ${POS_ENCODING} --checkpoint_freq ${CKPT_FREQ} --checkpoint_dir ${CHECKPOINT_DIR} --learning_rate ${LEARNING_RATE} --num_layers ${NUM_LAYERS} --num_heads ${NUM_HEADS} --embedding_size ${HIDDEN_SIZE} --seed ${SEED}

echo "Training complete. Results saved in ${CHECKPOINT_DIR}"

echo "Running Evaluation"

python -m axiomatic_training.test_dsep --model_path ${CHECKPOINT_DIR}/final_model.pt --train_data ${TRAIN_DATA} --val_data ${VAL_DATA} --pos_encoding ${POS_ENCODING} --n_layer ${NUM_LAYERS} --n_head ${NUM_HEADS} --n_embd ${HIDDEN_SIZE}