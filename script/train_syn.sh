CUDA_VISIBLE_DEVICES=1 \
python ../train.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/SyntheticDataset \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 64 \
--extra_tag edgeconv_syndata3 \
--epochs 90 \
--lr 1e-3

