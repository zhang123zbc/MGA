
import os

import torch

import torch.nn.functional as F
from torchvision import transforms as T

from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
from torchvision.models import resnet18,ResNet18_Weights
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/labels', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=1, help="How many images process1 at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations (Sampling Number)")
parser.add_argument("--portion", type=float, default=0.2, help="protion for the mixed image")
parser.add_argument("--gamma", type=float, default=0.5, help="protion for the mixed original image")
parser.add_argument("--zeta", type=float, default=3.0, help="weighted")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)
def calculate_image_distance(model, img_batch1, img_batch2,
                             distance_type: str = 'feature_l2',
                             feature_layer: str = 'avgpool',
                             device: torch.device = None):
    '''Calculate the feature distance on the avgpool layer of the proxy model, with the distance measured by the L2 norm'''
    if device is None:
        device = next(model.parameters()).device
    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)
    model = model.to(device).eval()


    if 'feature' in distance_type:
        feature_dict = {}

        def hook_fn(module, input, output):
            feature_dict['feat'] = output.detach()

        model=model[1]
        target_layer = dict(model.named_modules())[feature_layer]
        hook_handle = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = model(img_batch1)
            feat1 = feature_dict['feat']
            _ = model(img_batch2)
            feat2 = feature_dict['feat']
        hook_handle.remove()

        feat1 = feat1.view(feat1.shape[0], -1)
        feat2 = feat2.view(feat2.shape[0], -1)
        target_batch1, target_batch2 = feat1, feat2
    else:
        target_batch1 = img_batch1.view(img_batch1.shape[0], -1)
        target_batch2 = img_batch2.view(img_batch2.shape[0], -1)

    with torch.no_grad():
        if distance_type.endswith('l1'):
            distance = F.l1_loss(target_batch1, target_batch2, reduction='mean')
        elif distance_type.endswith('l2'):
            distance = F.mse_loss(target_batch1, target_batch2, reduction='mean').sqrt()
        elif distance_type.endswith('cosine'):

            cos_sim = F.cosine_similarity(target_batch1, target_batch2, dim=1).mean()
            distance = 1 - cos_sim

    return distance.item()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
def scatter(ds1, ds2, save_path="./scatter_plot.pdf", figsize=(6, 6), dpi=300):
    '''Scatter plot characterized by feature distances between all adversarial examples and the original template'''
    scale=1.35
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 1.2*scale
    plt.rcParams['xtick.major.width'] = 1.2*scale
    plt.rcParams['ytick.major.width'] = 1.2*scale
    plt.rcParams['xtick.labelsize'] = 10*scale
    plt.rcParams['ytick.labelsize'] = 10*scale
    plt.rcParams['axes.labelsize'] = 12*scale

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
    ax.set_aspect('equal')


    ax.set_title('',#Distance scatter diagram of counter samples and original samples
                 fontsize=14, fontweight='bold', pad=20)


    center_x, center_y = 0, 0
    center_radius = 0.10
    center_circle = Circle((center_x, center_y), center_radius,
                          color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.add_patch(center_circle)

    max_dist = max(
        max(ds1) if ds1 else 0,
        max(ds2) if ds2 else 0
    )

    ax.set_xlim(-max_dist*1.1, max_dist*1.1)
    ax.set_ylim(-max_dist*1.1, max_dist*1.1)

    if max_dist > 0:

        major_ticks = np.linspace(0, max_dist, 5)

        minor_interval = (max_dist / 4) / 2

        ax.set_xticks(major_ticks)
        ax.set_xticks(np.arange(0, max_dist*1.1, minor_interval), minor=True)
        ax.xaxis.set_major_locator(MultipleLocator(max_dist/4))

        ax.set_yticks(major_ticks)
        ax.set_yticks(np.arange(0, max_dist*1.1, minor_interval), minor=True)
        ax.yaxis.set_major_locator(MultipleLocator(max_dist/4))
        fmt = FuncFormatter(lambda v, pos: f"{abs(v):.3f}".rstrip('0').rstrip('.'))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        ax.set_xlabel('feature distance(L2)', fontweight='bold')
        ax.set_ylabel('feature distance(L2)', fontweight='bold')


        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=3, color='#999999')

    color_ds1 = '#A23B72'
    color_ds2 = '#F18F01'
    marker_size = 60

    n1 = len(ds1)
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    x1 = [d * np.cos(angle) for d, angle in zip(ds1, angles1)]
    y1 = [d * np.sin(angle) for d, angle in zip(ds1, angles1)]
    ax.scatter(x1, y1, s=marker_size, c=color_ds1, alpha=0.7, edgecolors='black', linewidth=1)

    n2 = len(ds2)
    angles2 = np.linspace(np.pi/n2, 2*np.pi + np.pi/n2, n2, endpoint=False)
    x2 = [d * np.cos(angle) for d, angle in zip(ds2, angles2)]
    y2 = [d * np.sin(angle) for d, angle in zip(ds2, angles2)]
    ax.scatter(x2, y2, s=marker_size, c=color_ds2, alpha=0.7, edgecolors='black', linewidth=1)

    center_patch = mpatches.Patch(color='#2E86AB', alpha=0.8, label='clean sample')
    ds1_patch = mpatches.Patch(color=color_ds1, alpha=0.7, label='MGA')
    ds2_patch = mpatches.Patch(color=color_ds2, alpha=0.7, label='RAP')
    ax.legend(handles=[center_patch, ds1_patch, ds2_patch],
              loc='upper right', frameon=True, framealpha=0.9,
              fontsize=11*scale, labelspacing=0.5, handletextpad=0.5,
              bbox_to_anchor=(1.01, 1.01))
    # ax.grid(True, which='minor', alpha=0.3, linestyle='--', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"The scatter plot has been saved to：{save_path}")
def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=torch.nn.Sequential(Normalize(opt.mean, opt.std),resnet18(weights=ResNet18_Weights).eval().cuda())
    size = 224
    X = ImageNet(opt.input_dir, opt.input_csv, size=size)
    data_loader_x = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    Y=ImageNet('./MGA', opt.input_csv, size=size)
    Z = ImageNet('./rap/', opt.input_csv, size=size)
    data_loader_y = DataLoader(Y, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    data_loader_z = DataLoader(Z, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    ds1 = []
    ds2 = []
    for (images_x, images_ID,  gt_cpu),(images_y, images_ID,  gt_cpu),(images_z, images_ID,  gt_cpu) in tqdm(zip(data_loader_x,data_loader_y,data_loader_z)):
        images_x = images_x.to(device)
        images_y = images_y.to(device)
        distance1 = calculate_image_distance(model,images_x,images_y,distance_type='feature_l2',feature_layer='avgpool',device=device)
        distance2 = calculate_image_distance(model, images_x, images_z, distance_type='feature_l2',
                                             feature_layer='avgpool', device=device)
        ds1.append(distance1)
        ds2.append(distance2)
    scatter(ds1,ds2)
if __name__ == '__main__':
    main()