CUDA_VISIBLE_DEVICES=2 \
python ../test_save_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg.yaml \
--test_tag building3d_all_ptv3 \
--batch_size 4