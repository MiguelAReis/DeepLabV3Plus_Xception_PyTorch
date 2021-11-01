CUDA_VISIBLE_DEVICES=0 python3 train.py --backbone xception --lr 0.007 --workers 4 --epochs 100 --batch-size 16 --gpu-ids 0 --checkname rgbFire_xception --eval-interval 1 --dataset rgbFire
