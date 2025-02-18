CUDA_VISIBLE_DEVICES=3 \
python ../train_PoznanRD.py \
--data_path /data/haoran/dataset/RoofDiffusion/dataset/PoznanRD/Point2Roof \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 64 \
--extra_tag PoznanRD_original_epoch_150 \
--epochs 150 \
--lr 1e-3