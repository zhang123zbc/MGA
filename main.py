import argparse
import os
import torch
from tqdm import tqdm
import MGA.MGA as MGA
from loader import ImageNet
from torch.utils.data import DataLoader
from PIL import Image


import numpy as np
def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('--batch_size', default=20, type=int, help='the bacth size')
    parser.add_argument('--size', default=224, type=int, help='the image size')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
    parser.add_argument('--input_dir', default='E:/integrant_path/dataset/images', type=str, help='the path for custom benign images, default: untargeted MGA data')
    parser.add_argument('--input_csv', default='E:/integrant_path/dataset/labels', type=str,
                        help='the path for custom benign csv, default: untargeted MGA data')
    parser.add_argument('--output_dir', default='./MIT', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--GPU_ID', default='0', type=str)
    return parser.parse_args()
def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir+"/" + name)

if __name__ == '__main__':
    args = get_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    X = ImageNet(args.input_dir, args.input_csv, size=args.size)
    data_loader = DataLoader(X, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID, gt_cpu in tqdm(data_loader):
        gt_cpu = torch.tensor(gt_cpu)
        gt = gt_cpu.to(device)
        images = images.to(device)
        adv_img5 = MGA.MGA
        adv_img_np5 = adv_img5(images.clone(),gt).cpu().numpy()
        adv_img_np5 = np.transpose(adv_img_np5, (0, 2, 3, 1)) * 255
        save_image(adv_img_np5, images_ID, args.output_dir)