# t5-pegasus pytorch
## 最新更新
- 增加t5-copy模型，在t5-pegasus的基础上增加了pointer generator，用t5-pegasus的预训练任务继续训练
- 增加t5-copy-large模型，在t5-copy的基础上用公开的文本摘要数据集进行训练
- 增加examples，基于pytorch_lightning的多卡训练
## 结果对比
数据集：[AdvertiseGen](https://www.luge.ai/#/luge/dataDetail?id=9)

| model | bleu  | rouge-1 | rouge-2 | rouge-2 |
|-------|-------|---------|---------|---------|
|    t5-pegasus-base   | 0.087 | 0.4299  | 0.1834  | 0.2675  |
| t5-copy  |  0.089      |  0.4257 | 0.1814  | 0.2626  |


**使用t5-copy模型transformers的版本不能高于4.12.0**

## 下载模型
### huggingface model hub

| 模型名	            | MODEL_NAME  |
|-----------------| ----  |
| t5-pegasus-base | imxly/t5-pegasus |
| t5-pegasus-small | imxly/t5-pegasus-small |
| t5-copy         | imxly/t5-copy  |
| t5-copy-summary | imxly/t5-copy-summary |

## how to use
pytorch1.7.0 + transformers4.3.3

```python
from tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

model_path = './'
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
text = '蓝蓝的天上有一朵白白的云'
ids = tokenizer.encode(text, return_tensors='pt')
output = model.generate(ids,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        eos_token_id=tokenizer.sep_token_id,
                        max_length=30).numpy()[0]
print(''.join(tokenizer.decode(output[1:])).replace(' ', ''))
```

感谢追一科技开源的[t5-pegasus](https://github.com/ZhuiyiTechnology/t5-pegasus)
