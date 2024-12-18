CUDA_VISIBLE_DEVICES=2 \
python ../train.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/RealDataset \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 16 \
--extra_tag original_syndata_realdata_finetune5 \
--epochs 110 \
--lr 1e-3