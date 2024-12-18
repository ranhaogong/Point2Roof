CUDA_VISIBLE_DEVICES=2 \
python ../test.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/RealDataset \
--cfg_file ../cfg/model_cfg.yaml \
--test_tag edgeconv_multiscalefusion_syndata_realdata_finetune