python train.py \
--train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
--dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
--batch_size 6 \
--max_epochs 10 \
--max_source_length 512 \
--max_target_length 300 \
--model_path /home/xianglingyang/pretrained_models/torch/t5-copy \
--gpus 4 \
--lr 5e-5 --model_type t5copy


python train.py \
--train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
--dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
--batch_size 6 \
--max_epochs 10 \
--max_source_length 512 \
--max_target_length 150 \
--model_path /home/xianglingyang/pretrained_models/torch/t5-copy  \
--gpus 4 \
--lr 5e-5 --model_type t5-pegasus


python train.py \
--train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
--dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
--batch_size 6 \
--max_epochs 10 \
--max_source_length 512 \
--max_target_length 300 \
--model_path /home/xianglingyang/pretrained_models/torch/cpt-large  \
--gpus 4 \
--lr 5e-5 --model_type cpt

python train.py \
--train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
--dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
--batch_size 6 \
--max_epochs 10 \
--max_source_length 512 \
--max_target_length 300 \
--model_path /home/xianglingyang/pretrained_models/torch/prophet  \
--gpus 4 \
--lr 5e-5 --model_type prophet