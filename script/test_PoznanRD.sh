CUDA_VISIBLE_DEVICES=3 \
python ../test_PoznanRD.py \
--data_path /data/haoran/dataset/RoofDiffusion/dataset/PoznanRD/Point2Roof \
--cfg_file ../cfg/model_cfg.yaml \
--test_tag PoznanRD_original_epoch_150 \
--batch_size 4