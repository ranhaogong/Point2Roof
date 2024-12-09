CUDA_VISIBLE_DEVICES=1 \
python -m debugpy --listen 5679 --wait-for-client ../train.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/SyntheticDataset \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 64 \
--extra_tag transformer_only \
--epochs 90 \
--lr 1e-3 