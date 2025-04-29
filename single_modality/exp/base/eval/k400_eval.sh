export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='k400_small_f8_lr2e-4_w5e35_dp0.1_hdp0.1_ld0.75_g64'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='p2:s3://k400'
DATA_PATH='' # replace with your data_path
MODEL_PATH='' # replace with your model_path

PARTITION='video'
GPUS=64
GPUS_PER_NODE=8
CPUS_PER_TASK=16

MAX_TOKEN_COUNT=2048
TEST_RES=224
TEST_FRAME=8
OUTPUT_HEAD=1 # can be set with {0, 1, 2}, corresponding to {1024, 2048, 3072} tokens, use nearest head for your token budget

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u run_flexible_finetune.py \
    --eval \
    --limit_token_num 6144 \
    --largest_num_input_token 3072 \
    --middle_num_input_token 2048 \
    --least_num_input_token 1024 \
    --eval_max_token_num ${MAX_TOKEN_COUNT} \
    --eval_input_size ${TEST_RES} \
    --eval_short_side_size ${TEST_RES} \
    --eval_true_frame ${TEST_FRAME} \
    --output_head ${OUTPUT_HEAD} \
    --finetune ${MODEL_PATH} \
    --model fluxvit_base_patch14 \
    --data_path ${DATA_PATH} \
    --data_set 'Kinetics_sparse_ofa' \
    --prefix ${PREFIX} \
    --split ',' \
    --nb_classes 400 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 252 \
    --short_side_size 252 \
    --model_res 252 \
    --save_ckpt_freq 100 \
    --num_frames 24 \
    --eval_orig_frame 24 \
    --orig_t_size 24 \
    --num_workers 12 \
    --warmup_epochs 1 \
    --tubelet_size 1 \
    --epochs 5 \
    --lr 2e-4 \
    --drop_path 0.1 \
    --head_drop_path 0.1 \
    --fc_drop_rate 0.0 \
    --layer_decay 0.75 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --test_best