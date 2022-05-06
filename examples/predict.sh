python predict_t5_copy.py \
--predict_file qg.jsonl \
--batch_size 6 \
--max_source_length 512 \
--max_target_length 200 \
--model_path /home/xianglingyang/pretrained_models/torch/t5-copy \
--gpus 4 \
--resume saved/0-epoch=05-val_bleu=0.0047.ckpt \
--output_path qg_predictions.txt