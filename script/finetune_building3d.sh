CUDA_VISIBLE_DEVICES=2 \
python ../train_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 64 \
--extra_tag finetune_syndata_building3d_edgeconv \
--epochs 150 \
--lr 1e-3