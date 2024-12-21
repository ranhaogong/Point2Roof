# CUDA_VISIBLE_DEVICES=1 \
# python -m debugpy --listen 5679 --wait-for-client ../train.py \
# --data_path /data/haoran/dataset/RoofReconstructionDataset/SyntheticDataset \
# --cfg_file ../cfg/model_cfg.yaml \
# --batch_size 64 \
# --extra_tag transformer_only \
# --epochs 90 \
# --lr 1e-3 

CUDA_VISIBLE_DEVICES=2 \
python -m debugpy --listen 5685 --wait-for-client ../test.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/RealDataset \
--cfg_file ../cfg/model_cfg.yaml \
--test_tag original_realdata