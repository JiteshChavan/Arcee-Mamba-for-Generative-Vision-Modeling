export CUDA_VISIBLE_DEVICES=0

# 48 GB, H100 is 80GB supports 96 batchsize, lets set per gpu batch size = 48 to be safe

NUM_GPUS=1
BATCH_SIZE=8
EVAL_BS=4
GRAD_ACCUM_STEPS=4
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

torchrun --standalone --nproc_per_node=1 ../Arcee/train_grad_acc.py --exp acc_test --datadir ../data/celeba256/lmdb_new --dataset celeba_256 --eval-refdir ../data/celeba256/real_samples \
  --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model Arcee-XS/2 \
  --scan-type Arcee_2 \
  --ssm-dstate 256 \
  --grad-accum-step $GRAD_ACCUM_STEPS \
  --train-steps 50000 \
  --eval-every 5000 \
  --plot-every 50 \
  --ckpt-every 10000 \
  --log-every 2 \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --sample-bs $BATCH_SIZE \
  --eval-bs $EVAL_BS \
  --lr 3e-4 \
  --learnable-pe \
  --path-type GVP \
  --eval-nsamples 10000 \
  --rms-norm \
  --fused-add-norm \
  --drop-path 0.0 \
  --save-content-every 5000 \
  #--use-wandb