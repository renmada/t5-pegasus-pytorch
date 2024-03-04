# t5-pegasus pytorch
## 最新更新
- 重构代码，支持更多模型
- 支持transformers最新版本
[老版代码点这里](https://github.com/renmada/t5-pegasus-pytorch/tree/legacy)
## 模型效果对比
数据集：[LCSTS_new](https://www.luge.ai/#/luge/dataDetail?id=10)
训练集取前一万条，验证集取前一千条

| model                | bleu        | rouge-1       | rouge-2      | rouge-l      |
|----------------------|-------------|---------------|--------------|--------------|
| t5-pegasus-base      | 0.1276      | 0.3490        | 0.2123       | 0.3155       |
| t5-copy              | 0.0938      | 0.3369        | 0.1955       | 0.3086       |
| Pegasus-238M-Chinese | 0.1200      | 0.3252        | 0.1957       | 0.2924       |
| Pegasus-523M-Chinese | 0.1233      | 0.3313        | 0.2032       | 0.2996       |
| cpt-large            |  **0.1366** | **0.3550**    | **0.2242**   | **0.3220**   |
| prophet-zh           | 0.1240      | 0.3419        | 0.2109       | 0.3107       |

## 数据格式
[样例数据](https://github.com/renmada/t5-pegasus-pytorch/blob/legacy/examples/sample_data.json)
## huggingface模型

| model_type	 | model_type                             |
|-------------|----------------------------------------|
| t5-pegasus  | imxly/t5-pegasus                       |
| t5copy      | imxly/t5-copy                          |
| Pegasus     | IDEA-CCNL/Randeng-Pegasus-238M-Chinese |
| Pegasus     | IDEA-CCNL/Randeng-Pegasus-523M-Chinese |
| cpt         | fnlp/cpt-large                         |
| prophet     | imxly/prophetnet-zh                    |


## 训练命令
### requirements
环境可以参考这个[issue](https://github.com/renmada/t5-pegasus-pytorch/issues/58)
```
torch >=1.10.0
transformers
pytorch_lightning==1.4.9
torchmetrics==0.5.0
```
model_type见上方表格
```shell
python train.py \
--train_file train.json \
--dev_file dev.json \
--batch_size 6 \
--max_epochs 10 \
--max_source_length 512 \
--max_target_length 300 \
--model_path  imxly/t5-pegasus \
--gpus 4 \
--lr 5e-5 \
--model_type t5-pegasus
```
## 参考
https://github.com/ZhuiyiTechnology/t5-pegasus  
https://github.com/fastnlp/CPT  
https://github.com/IDEA-CCNL/Fengshenbang-LM  
https://github.com/microsoft/ProphetNet


