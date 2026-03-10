IMAGENET_PATH=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train
CKPT=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/jit-h-16


OUTPUT_DIR=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/output/JiT-H-raindrop04/16

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29514 main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--img_size 256 \
--batch_size 12 --lr 5e-5 \
--epochs 600 --warmup_epochs 5 --eval_epoch 5 \
--eval_num_images 100 --cfg 1.0 \
--output_dir ${OUTPUT_DIR} --use_bg_subnet 1 --use_scene_dataset 5 \
--data_path ${IMAGENET_PATH} --resume ${CKPT}