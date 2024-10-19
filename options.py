import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--arch', type=str, default='IPT',choices=['IPT','EDT'])
parser.add_argument('--de_type', nargs='+', default='lr4_noise30',
                    choices=['lr4_noise30', 'lr4_jpeg30', 'sr_2','sr_3','sr_4', 'denoise_30'
                            'denoise_50', 'derainL', 'derainH', 'low_light',],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=48, help='patchsize of input.')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs to train the total model.')
parser.add_argument('--val_every_n_epoch', type=int, default=25)
parser.add_argument('--batch_size', type=int,default=16,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float,default=1e-4,help="learning rate")

parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')
# path
parser.add_argument('--data_file_dir', type=str, default='./data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--dataset_dir', type=str, default='/data/guohang/dataset',
                    help='where training images of deraining saves.')
parser.add_argument('--output_path', type=str, default="./output/", help='output save path')
parser.add_argument("--wblogger",type=str,default=None,help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="./train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default=4,help = "Number of GPUs to use for training")

options = parser.parse_args()

