import os
import argparse
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from utils import *
from models import LightModel
from args import parser

if __name__ == '__main__':

    args = parser.parse_args()

    model = LightModel(args)
    data = EncoderDecoderData(args, model.tokenizer)
    dataloaders = data.get_dataloader()

    for fold in range(args.kfold):
        pl.seed_everything(args.seed + fold)
        train_data, dev_data = dataloaders['train'][fold], dataloaders['dev'][fold]
        if fold > 0:
            model = LightModel(args)
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{fold:02d}-{epoch:02d}-{bleu:.4f}-{rouge:.4f}-{rouge-1:.4f}-{rouge-2:.4f}-{rouge-l:.4f}',
            save_weights_only=True,
            save_on_train_epoch_end=True,
            monitor='rouge',
            mode='max',
        )
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint], logger=False)
        trainer.fit(model, train_data, dev_data)
        del model
        del trainer
        torch.cuda.empty_cache()
