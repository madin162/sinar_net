CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
train.py --dataset 'Volleyball' --modality RGB --lr 5e-7 --max_lr 5e-5 --lr_step_down 25 --epochs 30  \
--device "0,1,2,3" --load_text --data_path 'data/' \
--num_frame 5  --num_total_frame 10 --num_activities 8 --batch 2 --test_batch 2 --num_tokens 12 \
--fp16-mode --enable_dali --test_freq 1 \
--enc_layers 6 --random_seed 7 --hidden_dim 128 \
--motion --multi_corr