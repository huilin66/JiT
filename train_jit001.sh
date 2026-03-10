IMAGENET_PATH=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train
OUTPUT_DIR=./output/JiT-B-raindrop001/16
CKPT=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/jit-b-16

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--img_size 256 \
--batch_size 64 --lr 5e-5 \
--epochs 300 --warmup_epochs 5 --eval_epoch 5 \
--eval_num_images 100 --cfg 1.0 \
--output_dir ${OUTPUT_DIR} --use_bg_subnet 0 --use_scene_dataset 1 \
--data_path ${IMAGENET_PATH} --resume ${CKPT}
