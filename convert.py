from transformers import MT5Config, MT5ForConditionalGeneration, load_tf_weights_in_t5
import torch
config = MT5Config.from_pretrained('config.json')
model = MT5ForConditionalGeneration(config)


ckpt = 'D:\\BaiduNetdiskDownload\\chinese_t5_pegasus_base\\chinese_t5_pegasus_base\\model.ckpt'

model = load_tf_weights_in_t5(model, config, ckpt)

torch.save(model.state_dict(), 'pytorch_model.bin')