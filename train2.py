import os
import pytorch_lightning as pl
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
            save_weights_only=True,
            every_n_train_steps=10000,
            save_last=True,
        )
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint], logger=False)
        trainer.fit(model, train_data, dev_data)
        del model
        del trainer
        torch.cuda.empty_cache()
