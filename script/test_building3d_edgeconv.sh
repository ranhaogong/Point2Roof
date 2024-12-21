CUDA_VISIBLE_DEVICES=1 \
python ../test_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 4 \
--test_tag building3d_2_edgeconv