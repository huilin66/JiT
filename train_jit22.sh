IMAGENET_PATH=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2
CKPT=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/jit-h-16

OUTPUT_DIR=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/output/JiT-H-raindrop24/16

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29524 main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--img_size 256 \
--batch_size 12 --lr 5e-5 --lr_schedule cosine \
--epochs 600 --warmup_epochs 5 --eval_epoch 100 \
--eval_num_images 100 --cfg 1.0 \
--output_dir ${OUTPUT_DIR} --use_bg_subnet 1 --use_scene_dataset 1 \
--data_path ${IMAGENET_PATH} --resume ${CKPT}
