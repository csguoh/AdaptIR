import os
import random
import copy
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

def add_jpg_compression(img, quality=30):
    """Add JPG compression artifacts.

    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.

    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = np.clip(cv2.imdecode(encimg, 1), 0, 255).astype(np.uint8)
    return img


class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)
        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])  # only used in denoise

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'sr' in self.de_type:
            self._init_sr_idx()
        if 'lr4_noise30' in self.de_type:
            self._init_lr_dn_idx()
        if 'lr4_jpeg30' in self.de_type:
            self._init_lr_jpeg_idx()
        if 'denoise_30' in self.de_type or 'denoise_50' in self.de_type:
            self._init_dn_ids()
        if 'derainL' in self.de_type:
            self._init_rs_ids(mode='L')
        if 'derainH' in self.de_type:
            self._init_rs_ids(mode='H')
        if 'low_light' in self.de_type:
            self._init_low_light_ids()

    def _init_lr_dn_idx(self):
        clean_ids = []
        div2k = sorted(os.listdir(self.args.dataset_dir + '/DIV2K/DIV2K_train_HR'))[:800]
        filckr2k = os.listdir(self.args.dataset_dir + '/Flickr2K/Flickr2K_HR')
        clean_ids += [os.path.join(self.args.dataset_dir + '/DIV2K/DIV2K_train_HR', file) for file in div2k]
        clean_ids += [os.path.join(self.args.dataset_dir + '/Flickr2K/Flickr2K_HR', file) for file in filckr2k]  # total -- 3450


        self.lr_dn_ids = [{"clean_id": x, "de_type": self.de_type} for x in clean_ids]
        random.shuffle(self.lr_dn_ids)
        self.num_clean = len(self.lr_dn_ids)
        print("Total LR4&DN30 Ids : {}".format(self.num_clean))

    def _init_lr_jpeg_idx(self):
        clean_ids = []
        div2k = sorted(os.listdir(self.args.dataset_dir + '/DIV2K/DIV2K_train_HR'))[:800]
        filckr2k = os.listdir(self.args.dataset_dir + '/Flickr2K/Flickr2K_HR')
        clean_ids += [os.path.join(self.args.dataset_dir + '/DIV2K/DIV2K_train_HR', file) for file in div2k]
        clean_ids += [os.path.join(self.args.dataset_dir + '/Flickr2K/Flickr2K_HR', file) for file in filckr2k]  # total -- 3450


        self.lr_jpeg_ids = [{"clean_id": x, "de_type": self.de_type} for x in clean_ids]
        random.shuffle(self.lr_jpeg_ids)

        self.num_clean = len(self.lr_jpeg_ids)
        print("Total LR4&JPEG30 Ids : {}".format(self.num_clean))



    def _init_sr_idx(self):
        clean_ids = []
        div2k = sorted(os.listdir(self.args.dataset_dir + '/DIV2K/DIV2K_train_HR'))[:800]
        filckr2k = os.listdir(self.args.dataset_dir + '/Flickr2K/Flickr2K_HR')
        clean_ids += [os.path.join(self.args.dataset_dir + '/DIV2K/DIV2K_train_HR', file) for file in div2k]
        clean_ids += [os.path.join(self.args.dataset_dir + '/Flickr2K/Flickr2K_HR', file) for file in filckr2k]  # total -- 3450

        if 'sr_2' in self.de_type:
            self.sr2_ids = [{"clean_id": x, "de_type": self.de_type} for x in clean_ids]
            random.shuffle(self.sr2_ids)
        if 'sr_3' in self.de_type:
            self.sr3_ids = [{"clean_id": x, "de_type": self.de_type} for x in clean_ids]
            random.shuffle(self.sr3_ids)
        if 'sr_4' in self.de_type:
            self.sr4_ids = [{"clean_id": x, "de_type": self.de_type} for x in clean_ids]
            random.shuffle(self.sr4_ids)

        self.num_clean = len(clean_ids)
        print("Total SR Ids : {}".format(self.num_clean))

    def _init_low_light_ids(self):
        temp_ids = []
        temp_ids += os.listdir(os.path.join(self.args.dataset_dir, 'LOLv1', 'Train','input'))
        new_ids = []
        for name in temp_ids:
            if 'DS' not in name:
                new_ids.append(name)
        temp_ids = new_ids
        self.low_light_ids = [{"clean_id": os.path.join(self.args.dataset_dir, 'LOLv1', 'Train','input', x), "de_type": self.de_type} for x in temp_ids] * 16
        random.shuffle(self.low_light_ids)
        self.low_light_counter = 0
        self.num_low_light = len(self.low_light_ids)
        print("Total Low-light Ids : {}".format(self.num_low_light))


    def _init_dn_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.dataset_dir + '/WED')+os.listdir(self.args.dataset_dir + '/BSD400')
        clean_ids += [id_ for id_ in name_list if id_.strip() in temp_ids]
        tmp=[]
        for elem in clean_ids:
            if 'bmp' in elem:
                tmp.append(self.args.dataset_dir + '/WED/'+elem)
            else:
                tmp.append(self.args.dataset_dir + '/BSD400/'+elem)
        clean_ids = tmp
        # add DIV2K and Filkr2K
        div2k = sorted(os.listdir(self.args.dataset_dir+'/DIV2K/DIV2K_train_HR'))[:800]
        filckr2k = os.listdir(self.args.dataset_dir+'/Flickr2K/Flickr2K_HR')
        clean_ids += [os.path.join(self.args.dataset_dir + '/DIV2K/DIV2K_train_HR',file) for file in div2k]
        clean_ids += [os.path.join(self.args.dataset_dir+'/Flickr2K/Flickr2K_HR',file) for file in filckr2k] # total -- 8194

        if 'denoise_30' in self.de_type:
            self.s30_ids = [{"clean_id": x,"de_type":'denoise_30'} for x in clean_ids]
            random.shuffle(self.s30_ids)
            self.s30_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":'denoise_50'} for x in clean_ids]
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_rs_ids(self,mode):
        if mode == 'H':
            temp_ids = []
            rain_path = self.args.dataset_dir + '/RainTrainH'
            rain13k = os.listdir(rain_path)
            temp_ids += [os.path.join(rain_path, file.replace('norain', 'rain')) for file in rain13k if 'norain-' in file] * 4

            self.rsH_ids = [{"clean_id": x, "de_type": 'derainH'} for x in temp_ids]
            random.shuffle(self.rsH_ids)
            self.rlH_counter = 0
            self.num_rlH = len(self.rsH_ids)
            print("Total Heavy Rainy Ids : {}".format(self.num_rlH))

        else:
            temp_ids = []
            rain_path = self.args.dataset_dir + '/RainTrainL'
            rain13k = os.listdir(rain_path)
            temp_ids += [os.path.join(rain_path, file.replace('norain', 'rain')) for file in rain13k if 'norain-' in file] * 24

            self.rsL_ids = [{"clean_id": x, "de_type": 'derainL'} for x in temp_ids]
            random.shuffle(self.rsL_ids)
            self.rlL_counter = 0
            self.num_rlL = len(self.rsL_ids)
            print("Total Light Rainy Ids : {}".format(self.num_rlL))

    def _crop_patch(self, img_1, img_2, s_hr=1):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H * s_hr:(ind_H + self.args.patch_size) * s_hr,
                  ind_W * s_hr:(ind_W + self.args.patch_size) * s_hr]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rain-")[0] + 'norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_lr_name(self, hr_name):
        scale = self.de_type.split('_')[-1]
        base_name = os.path.basename(hr_name).split('.')[0]
        lr_path = os.path.join(hr_name.split('HR')[0] + 'LR_bicubic', 'X' + scale, base_name + 'x' + scale + '.png')
        return lr_path

    def _get_hybrid_name(self, hr_name):
        scale = '4'
        base_name = os.path.basename(hr_name).split('.')[0]
        lr_path = os.path.join(hr_name.split('HR')[0] + 'LR_bicubic', 'X' + scale, base_name + 'x' + scale + '.png')
        return lr_path


    def _get_normal_light_name(self, low_light_name):
        normal_light_name = low_light_name.replace('input', 'target')
        return normal_light_name



    def _merge_ids(self):
        self.sample_ids = []
        if 'lr4_noise30' in self.de_type:
            self.sample_ids += self.lr_dn_ids
        if 'lr4_jpeg30' in self.de_type:
            self.sample_ids += self.lr_jpeg_ids
        if 'denoise_30' in self.de_type:
            self.sample_ids += self.s30_ids
        if 'denoise_50' in self.de_type:
            self.sample_ids += self.s50_ids
        if "derainL" in self.de_type:
            self.sample_ids += self.rsL_ids
        if "derainH" in self.de_type:
            self.sample_ids += self.rsH_ids
        if "low_light" in self.de_type:
            self.sample_ids += self.low_light_ids
        if 'sr_2' in self.de_type:
            self.sample_ids += self.sr2_ids
        if 'sr_3' in self.de_type:
            self.sample_ids += self.sr3_ids
        if 'sr_4' in self.de_type:
            self.sample_ids += self.sr4_ids

        random.shuffle(self.sample_ids)
        print(len(self.sample_ids))


    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if 'denoise' in de_id:  # denoise
            clean_id = sample["clean_id"]
            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)
            clean_name = clean_id.split("/")[-1].split('.')[0]
            clean_patch = random_augmentation(clean_patch)[0]
            degrad_patch = self.D.single_degrade(clean_patch, de_id)

        if 'derain' in de_id:
            # Rain Streak Removal
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        if 'sr' in de_id:
            scale = int(self.de_type.split('_')[-1])
            hr_img = np.array(Image.open(sample["clean_id"]).convert('RGB'))
            clean_name = sample['clean_id'].split('/')[-1].split('.')[0]
            lr_name = self._get_hybrid_name(sample["clean_id"])
            lr_img = np.array(Image.open(lr_name).convert('RGB'))
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(lr_img, hr_img, s_hr=scale))

        if 'lr4_noise30' in de_id:
            scale = int(self.de_type[2])
            hr_img = np.array(Image.open(sample["clean_id"]).convert('RGB'))
            clean_name = sample['clean_id'].split('/')[-1].split('.')[0]
            lr_name = self._get_hybrid_name(sample["clean_id"])
            lr_img = np.array(Image.open(lr_name).convert('RGB'))
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(lr_img, hr_img, s_hr=scale))
            degrad_patch = self.D._add_gaussian_noise(degrad_patch,sigma=30)[0]

        if 'lr4_jpeg30' in de_id:
            scale = int(self.de_type[2])
            hr_img = np.array(Image.open(sample["clean_id"]).convert('RGB'))
            clean_name = sample['clean_id'].split('/')[-1].split('.')[0]
            lr_name = self._get_lr_name(sample["clean_id"])
            lr_img = np.array(Image.open(lr_name).convert('RGB'))
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(lr_img, hr_img, s_hr=scale))
            degrad_patch = add_jpg_compression(degrad_patch)

        if 'low_light' in de_id:
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_normal_light_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch


    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = copy.deepcopy(args)
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [os.path.join(self.args.denoise_path, id_) for id_ in name_list]
        self.clean_ids = sorted(self.clean_ids)
        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        # clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_img = np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')).astype(np.float32)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]
        noisy_img = self._add_gaussian_noise(clean_img)[0]
        # clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)
        clean_img, noisy_img = torch.from_numpy(np.transpose(clean_img / 255., (2, 0, 1))).float(), torch.from_numpy(
            np.transpose(noisy_img / 255., (2, 0, 1))).float()

        return [clean_name], noisy_img, clean_img

    def tile_degrad(input_, tile=128, tile_overlap=0):
        sigma_dict = {0: 0, 1: 15, 2: 25, 3: 50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
        restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored

    def __len__(self):
        return self.num_clean


class DerainLowlightDataset(Dataset):
    def __init__(self, args):
        super(DerainLowlightDataset, self).__init__()
        self.ids = []
        self.args = copy.deepcopy(args)
        self.task = self.args.de_type
        self.toTensor = ToTensor()
        self.set_dataset(self.task)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if  'derain' in self.task:  # derain
            self.ids = []
            name_list = os.listdir(os.path.join(self.args.derain_path, 'rainy'))
            # print(name_list)
            self.ids += [os.path.join(self.args.derain_path, 'rainy', id_) for id_ in name_list]
        elif self.task == 'low_light':
            self.ids = []
            name_list = os.listdir(os.path.join(self.args.low_light_path, 'input'))
            new_ids = []
            for name in name_list:
                if 'DS' not in name:
                    new_ids.append(name)
            name_list = new_ids
            self.ids += [os.path.join(self.args.low_light_path, 'input', id_) for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if 'derain' in self.task:
            filename = degraded_name.split('-')[-1]
            gt_name = os.path.join(self.args.derain_path, 'norain-' + filename)
        elif self.task == 'low_light':
            gt_name = degraded_name.replace('input', 'target')
        return gt_name

    def set_dataset(self, task):
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = np.array(Image.open(degraded_path).convert('RGB'))

        clean_img = np.array(Image.open(clean_path).convert('RGB'))

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class SRHybridTestDataset(Dataset):
    def __init__(self, args):
        super(SRHybridTestDataset, self).__init__()
        self.args = copy.deepcopy(args)
        self.clean_ids = []
        self.toTensor = ToTensor()

    def set_scale(self, s):
        self.scale = s
        self._init_sr_ids()

    def _init_sr_ids(self):
        hr_path = self.args.sr_path
        name_list = os.listdir(hr_path)
        self.clean_ids += [os.path.join(hr_path, id_) for id_ in name_list]
        self.clean_ids = sorted(self.clean_ids)
        self.num_clean = len(self.clean_ids)

    def __getitem__(self, clean_id):
        hr_img = np.array(Image.open(self.clean_ids[clean_id]).convert('RGB'))

        file_name, ext = os.path.splitext(os.path.basename(self.clean_ids[clean_id]))
        if 'Manga109' not in self.clean_ids[clean_id]:
            lr_path = os.path.join(os.path.dirname(self.clean_ids[clean_id]).replace('HR', 'LR_bicubic'),
                                   'X{}/{}x{}{}'.format(self.scale, file_name, self.scale, ext))
        else:
            lr_path = os.path.join(os.path.dirname(self.clean_ids[clean_id]).replace('HR', 'LR_bicubic'),
                                   'X{}/{}_LRBI_x{}{}'.format(self.scale, file_name, self.scale, ext))
        lr_img = np.array(Image.open(lr_path).convert('RGB'))


        if self.args.de_type == 'lr4_noise30':
            sigma=30
            noise = np.random.randn(*lr_img.shape)
            lr_img = np.clip(lr_img + noise * sigma, 0, 255).astype(np.uint8)
        if self.args.de_type == 'lr4_jpeg30':
            lr_img = add_jpg_compression(lr_img)


        ih, iw = lr_img.shape[:2]
        hr_img = hr_img[0:ih * self.scale, 0:iw * self.scale]
        hr_img, lr_img = self.toTensor(hr_img), self.toTensor(lr_img)

        return [lr_path.split('/')[-1]], lr_img, hr_img

    def __len__(self):
        return self.num_clean




