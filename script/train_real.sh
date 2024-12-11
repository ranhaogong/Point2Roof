CUDA_VISIBLE_DEVICES=3 \
python ../train.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/RealDataset \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 64 \
--extra_tag pct_real \
--epochs 90 \
--lr 1e-4

