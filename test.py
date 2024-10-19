import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
from utils.dataset_utils import DenoiseTestDataset, DerainLowlightDataset,SRHybridTestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.ipt import IPT
import lightning.pytorch as pl
import torch.nn.functional as F
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.common import calculate_psnr_ssim
from net.edt import EDT
device = torch.device('cuda')
from matplotlib import pyplot as plt



class MultiTaskIRModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        if args.arch == 'IPT':
            self.net = IPT(args)
            state_dict = torch.load('/data/guohang/pretrained/IPT_pretrain.pt')
            self.net.load_state_dict(state_dict, strict=False)
        elif args.arch == 'EDT':
            self.net = EDT(args)
            state_dict = torch.load('/data/guohang/pretrained/SRx2x3x4_EDTB_ImageNet200K.pth')
            self.net.load_state_dict(state_dict, strict=False)


    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def test_Denoise(net, dataset, sigma=15):
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))



def test_Derain_LowLight(net, dataset, task="derain"):
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            restored = net(degrad_patch)
            to_y = True if 'derain' in task else False
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch,to_y=to_y)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))




def test_SR(net,dataset,scale):
    dataset.set_scale(scale)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch,to_y=True,bd=scale)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

        print("SR scale=%d: psnr: %.2f, ssim: %.4f" % (scale, psnr.avg, ssim.avg))





def test_hybrid_degradation(net,dataset,scale):
    dataset.set_scale(scale)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch,to_y=True,bd=scale)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
        print("SR scale=%d: psnr: %.2f, ssim: %.4f" % (scale, psnr.avg, ssim.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--arch', type=str, default='IPT', choices=['IPT', 'EDT'])
    parser.add_argument('--de_type', nargs='+', default='lr4_noise30',
                        choices=['lr4_noise30', 'lr4_jpeg30', 'sr_2', 'sr_3', 'sr_4', 'denoise_30'
                                'denoise_50', 'derainL', 'derainH', 'low_light'],
                        help='which type of degradations is training and testing for.')
    parser.add_argument('--output_path', type=str, default="./test_output/", help='output save path')
    parser.add_argument('--base_path', type=str, default="/data/guohang/dataset", help='save path of test noisy images')
    parser.add_argument('--ckpt_name', type=str, default='/data/guohang/AdaptIR/train_ckpt/last.ckpt', help='checkpoint save path')
    testopt = parser.parse_args()



    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = testopt.ckpt_name

    denoise_splits = ["ColorDN/Urban100HQ"]
    derainH_splits = ["Rain100H/"]
    derainL_splits = ["Rain100L/"]
    low_light_splits = ["LOLv1/Test"]
    hybrid_splits = ['Set5','Set14','Urban100','B100','Manga109']
    sr_splits = ['Set5','Set14','Urban100','B100','Manga109']

    denoise_tests = []
    derain_tests = []
    sr_tests = []


    print("CKPT name : {}".format(ckpt_path))


    net = MultiTaskIRModel(testopt)
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    net.to(device)


    if 'denoise' in testopt.de_type:
        base_path = testopt.base_path
        for i in denoise_splits:
            testopt.denoise_path = os.path.join(base_path, i)
            denoise_testset = DenoiseTestDataset(testopt)
            denoise_tests.append(denoise_testset)
        for testset,name in zip(denoise_tests,denoise_splits):
            if 'denoise_30' in testopt.de_type:
                print('Start {} testing Sigma=30...'.format(name))
                test_Denoise(net, testset, sigma=30)
            if 'denoise_50' in testopt.de_type:
                print('Start {} testing Sigma=50...'.format(name))
                test_Denoise(net, testset, sigma=50)

    elif 'derainL' in testopt.de_type:
        print('Start testing light rain streak removal...')
        derain_base_path = testopt.base_path
        for name in derainL_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainLowlightDataset(testopt)
            test_Derain_LowLight(net, derain_set, task="derainL")

    elif 'derainH' in testopt.de_type:
        print('Start testing heavy rain streak removal...')
        derain_base_path = testopt.base_path
        for name in derainH_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainLowlightDataset(testopt)
            test_Derain_LowLight(net, derain_set, task="derainH")

    elif 'low_light' in testopt.de_type:
        print('Start testing heavy rain streak removal...')
        low_light_base_path = testopt.base_path
        for name in low_light_splits:
            print('Start testing {} low light enhancement...'.format(name))
            testopt.low_light_path = os.path.join(low_light_base_path,name)
            lowlight_set = DerainLowlightDataset(testopt)
            test_Derain_LowLight(net, lowlight_set, task="low_light")


    elif 'sr' in testopt.de_type:
        print('Start testing super-resolution...')
        sr_base_path = testopt.base_path
        for name in sr_splits:
            print('Start testing {} super-resolution...'.format(name))
            testopt.sr_path = os.path.join(sr_base_path,'ARTSR',name,'HR')
            sr_set = SRHybridTestDataset(testopt)
            sr_tests.append(sr_set)
        for testset,name in zip(sr_tests,sr_splits):
            if 'sr_2' in testopt.de_type:
                print('Start {} testing SRx2...'.format(name))
                test_SR(net, testset,scale=2)
            if 'sr_3' in testopt.de_type:
                print('Start {} testing SRx3...'.format(name))
                test_SR(net, testset,scale=3)
            if 'sr_4' in testopt.de_type:
                print('Start {} testing SRx4...'.format(name))
                test_SR(net, testset,scale=4)


    elif 'lr4_noise30' in testopt.de_type:
        print('Start testing super-resolution...')
        sr_base_path = testopt.base_path
        for name in sr_splits:
            print('Start testing {} LR4+Noise30...'.format(name))
            testopt.sr_path = os.path.join(sr_base_path,'ARTSR',name,'HR')
            sr_set = SRHybridTestDataset(testopt)
            sr_tests.append(sr_set)
            test_SR(net, sr_set, scale=4)


    elif 'lr4_jpeg30' in testopt.de_type:
        print('Start testing super-resolution...')
        sr_base_path = testopt.base_path
        for name in sr_splits:
            print('Start testing {} LR4+JPEG30...'.format(name))
            testopt.sr_path = os.path.join(sr_base_path,'ARTSR',name,'HR')
            sr_set = SRHybridTestDataset(testopt)
            sr_tests.append(sr_set)
            test_SR(net, sr_set, scale=4)


    else:
        raise NotImplementedError

