python train_image.py \
    --train_dataset /data0/dataset/OpenImage/train \
    --test_dataset /data0/dataset/Kodak/test \
    --gpu_num 6 \
    --batch_size 16 \
    --save_dir ./experiments/rt_v1 \
    --lr 5e-6 \
    --epochs 1200