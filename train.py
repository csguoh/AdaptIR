import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptTrainDataset
from net.ipt import IPT
from net.edt import EDT
from utils.schedulers import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.val_utils import AverageMeter, compute_psnr_ssim


class MultiTaskIRModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        print('load from pretrained model...')
        self.args = args
        if args.arch == 'IPT':
            self.net = IPT(args)
            state_dict = torch.load('/data/guohang/pretrained/IPT_pretrain.pt')
            self.net.load_state_dict(state_dict, strict=False)
        elif args.arch == 'EDT':
            self.net = EDT(args)
            state_dict = torch.load('/data/guohang/pretrained/SRx2x3x4_EDTB_ImageNet200K.pth')
            self.net.load_state_dict(state_dict, strict=False)
        print('frezz parameters in body except adapter, head and tail are NOT trainable')
        for name, param in self.net.named_parameters():
            if "adaptir" not in name:
                param.requires_grad = False

        # for name, param in self.net.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        self.loss_fn = nn.L1Loss()
        self.save_hyperparameters()


    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch,de_id)
        loss = self.loss_fn(restored,clean_patch)
        self.log("train_loss", loss)
        return loss



    # def validation_step(self,batch, batch_idx):
    #     ([clean_name], degrad_patch, clean_patch) = batch
    #     restored = self.net(degrad_patch)
    #     psnr_to_y = False if 'denoise' in self.args.de_type else True
    #     temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch,to_y=psnr_to_y)
    #     self.log('psnr',temp_psnr,on_epoch=True,sync_dist=True)
    #     return temp_psnr



    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = MultiStepLR(optimizer, milestones=[250,400,450,475], gamma=0.5)
        return [optimizer],[scheduler]


def main():
    logger = TensorBoardLogger(save_dir = "./logs")
    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,
                                          save_last=True,)

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    model = MultiTaskIRModel(opt)

    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",
                         devices=opt.num_gpus,
                         logger=logger,callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



