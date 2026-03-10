IMAGENET_PATH=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train
CKPT=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/jit-b-16

OUTPUT_DIR=./output/JiT-B-raindrop01/16

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--img_size 256 \
--batch_size 64 --lr 5e-5 \
--epochs 600 --warmup_epochs 5 --eval_epoch 5 \
--eval_num_images 100 --cfg 1.0 \
--output_dir ${OUTPUT_DIR} --use_bg_subnet 0 --use_scene_dataset 0 \
--data_path ${IMAGENET_PATH} --resume ${CKPT}


OUTPUT_DIR=./output/JiT-B-raindrop02/16

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29502 main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--img_size 256 \
--batch_size 64 --lr 5e-5 \
--epochs 600 --warmup_epochs 5 --eval_epoch 5 \
--eval_num_images 100 --cfg 1.0 \
--output_dir ${OUTPUT_DIR} --use_bg_subnet 0 --use_scene_dataset 1 \
--data_path ${IMAGENET_PATH} --resume ${CKPT}


OUTPUT_DIR=./output/JiT-B-raindrop03/16

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29503 main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--img_size 256 \
--batch_size 64 --lr 5e-5 \
--epochs 600 --warmup_epochs 5 --eval_epoch 5 \
--eval_num_images 100 --cfg 1.0 \
--output_dir ${OUTPUT_DIR} --use_bg_subnet 1 --use_scene_dataset 0 \
--data_path ${IMAGENET_PATH} --resume ${CKPT}


OUTPUT_DIR=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/output/JiT-B-raindrop04/16

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29504 main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--img_size 256 \
--batch_size 64 --lr 5e-5 \
--epochs 600 --warmup_epochs 5 --eval_epoch 5 \
--eval_num_images 100 --cfg 1.0 \
--output_dir ${OUTPUT_DIR} --use_bg_subnet 1 --use_scene_dataset 5 \
--data_path ${IMAGENET_PATH} --resume ${CKPT}