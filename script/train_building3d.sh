CUDA_VISIBLE_DEVICES=1 \
python ../train_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 64 \
--extra_tag building3d_1 \
--epochs 90 \
--lr 1e-3

