IMAGENET_PATH=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train
OUTPUT_DIR=./output/JiT-H-raindrop/16
CKPT=/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/jit-h-16

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 12 --lr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 12 --num_images 100 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval --num_sampling_steps 1 --resume ${CKPT}


