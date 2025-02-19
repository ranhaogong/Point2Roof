CUDA_VISIBLE_DEVICES=0 \
python ../train_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg_color_2048.yaml \
--batch_size 64 \
--extra_tag building3d_all_pointnet2_edge_color_2048 \
--epochs 90 \
--lr 1e-3

