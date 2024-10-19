import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--base_path', type=str, default="/data/guohang/dataset", help='save path of test noisy images')
parser.add_argument('--denoise_path',type=str,default='/data/guohang/dataset/CBSD68/original_png')
parser.add_argument('--derain_path', type=str, default="/data/guohang/dataset/Rain100L",
                    help='save path of test raining images')
parser.add_argument('--sr_path', type=str, default="/data/guohang/dataset/SR/Set5/HR",
                    help='path to the sr dataset for validation')
parser.add_argument('--output_path', type=str, default="./test_output/", help='output save path')

testopt = parser.parse_args()




















