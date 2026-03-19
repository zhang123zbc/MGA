
import os
import random

from torchvision.transforms import functional as TFF
import torch

import torch.nn.functional as F

import numpy as np


import MGA
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def dim(x,resize_rate=1.1):
    """
    Random transform the input images
    """
    if random.random()<0.2:
        return x
    # do not transform the input image
    # if torch.rand(1) > self.diversity_prob:
    #    return x
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    # resize the input image to random size
    rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)


def vertical_shift(x):
    _, _, w, _ = x.shape
    step = np.random.randint(low=0, high=w, dtype=np.int32)
    return x.roll(step, dims=2)


def horizontal_shift(x):
    _, _, _, h = x.shape
    step = np.random.randint(low=0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)


def vertical_flip(x):
    return x.flip(dims=(2,))


def horizontal_flip(x):
    return x.flip(dims=(3,))
def identity(x):
    return x
def rotate(x,angle):
    return TFF.rotate(img=x, angle=angle)

def paste_random_blocks_batch_sync_grad(a: torch.Tensor, b: torch.Tensor, n: int, s: int) -> torch.Tensor:
    size=a.shape[2]
    device = a.device
    result = a.clone()
    b_y1 = torch.randint(0, 224 - s + 1, (n,), device=device)
    b_x1 = torch.randint(0, 224 - s + 1, (n,), device=device)
    a_y1 = torch.randint(0, 224 - s + 1, (n,), device=device)
    a_x1 = torch.randint(0, 224 - s + 1, (n,), device=device)
    for i in range(n):
        y1_b, x1_b = b_y1[i], b_x1[i]
        y1_a, x1_a = a_y1[i], a_x1[i]
        result[:, :, y1_a:y1_a+s, x1_a:x1_a+s] = 0.0
        result[:, :, y1_a:y1_a+s, x1_a:x1_a+s] = b[:, :, y1_b:y1_b+s, x1_b:x1_b+s]
    return result

def batch_swirl_spatial(images, strength=1.0, radius=None, center=None):
    B, C, H, W = images.shape
    if center is None:
        center_x, center_y = W // 2, H // 2
    else:
        center_x, center_y = center
    if radius is None:
        radius = min(center_x, center_y, W - center_x, H - center_y) * 0.8
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=images.device, dtype=torch.float32),
        torch.arange(W, device=images.device, dtype=torch.float32),
        indexing='ij'
    )
    x_grid = x_grid.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
    y_grid = y_grid.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
    dx = x_grid - center_x
    dy = y_grid - center_y
    distance = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    angle = torch.atan2(dy, dx)

    mask = distance < radius

    twist = strength * (1 - distance / radius)
    twist = twist * mask.float()

    new_angle = angle + twist

    src_x = center_x + distance * torch.cos(new_angle)
    src_y = center_y + distance * torch.sin(new_angle)

    src_x_norm = 2.0 * src_x / (W - 1) - 1.0
    src_y_norm = 2.0 * src_y / (H - 1) - 1.0

    grid = torch.stack((src_x_norm, src_y_norm), dim=-1)  # (B, H, W, 2)

    swirled_images = F.grid_sample(
        images,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return swirled_images


def select(batch_sample, n):
    if n == 0:
        return identity(batch_sample)
    elif n == 1:
        return vertical_flip(batch_sample)
    elif n == 3:
        return horizontal_flip(batch_sample)
    elif n == 2:
        return vertical_shift(batch_sample)
    elif n == 4:
        return horizontal_shift(batch_sample)
    elif n == 5:
        return rotate(batch_sample, 5)
    elif n == 6:
        return rotate(batch_sample, -5)
    elif n == 7:
        return rotate(batch_sample, 15)
    elif n == 8:
        return rotate(batch_sample, -15)
    elif n == 9:
        return rotate(batch_sample, 45)
    elif n == 10:
        return rotate(batch_sample, -45)
    elif n == 11:
        return rotate(batch_sample, 90)
    elif n == 12:
        return rotate(batch_sample, -90)
    elif n == 13:
        return rotate(batch_sample, 180)
    elif n==14:
        return dim(batch_sample, 1+random.randint(1,9)*0.1)
    elif n==15:
        return paste_random_blocks_batch_sync_grad(batch_sample,batch_sample,4,100)
    elif n==16:
        return batch_swirl_spatial(batch_sample,random.random()*0.5+0.2)

def sort_trans(batch_sample,n):#Transformation Classification
    if n==1:
        return select(batch_sample,0)
    elif n==2:
        return select(batch_sample, random.randint(1,2))
    elif n==3:
        return select(batch_sample, random.randint(3, 4))
    elif n==4:
        return select(batch_sample, random.randint(5, 13))
    elif n==5:
        return select(batch_sample, 14)
    elif n==5:
        return select(batch_sample, 15)
    elif n==6:
        return select(batch_sample, 16)
    elif n==7:
        return batch_sample/random.randint(1,8)

def circle_trans(batch_sample,unique_arr,i):
    if i>=len(unique_arr):
        return batch_sample
    return circle_trans(sort_trans(batch_sample,unique_arr[i]),unique_arr,i+1)

def trans(batch_sample, n):#Mutually exclusive combination
    unique_arr = random.sample(range(1, 8), n)
    return circle_trans(batch_sample, unique_arr, 0)




import scipy.stats as st


class MGA(MGA.Attack):
    '''Multi-path Gradient Ascen(MGA)'''
    def __init__(
            self,
            x,
            gt,
            num_trans=3,
            Nk=5,
            is_trans=True,
            model_name='resnet18',
            decay=1.,
            zeta=2.0,
            epsilon=16.0/255,
            num_iter=10,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=None,
            m=7,
            k=8,
            ):
        super().__init__(model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / num_iter
        self.is_trans=is_trans
        self.epoch = num_iter
        self.decay = decay
        self.x=x
        self.gt=gt
        self.Nk=Nk
        self.k=k
        self.m=m
        self.zeta=zeta
        self.r=num_trans
        self.eps=epsilon
#If you want to use Tim
    def generate_kernel(self,kernel_size, nsig=3):#Gaussian kernel generation,
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).cuda()

    def get_grad(self, delta, **kwargs):#Enter mutually exclusive input changes or sampling
        x = delta.clone().detach().cuda()
        if self.is_trans:
            zeta_in = self.weighted_random(0.5, self.zeta)
            x_protect = x + torch.rand_like(delta).uniform_(-zeta_in * self.eps, zeta_in * self.eps)
            x_protect.requires_grad_(True)
            output_v3 = self.model(trans(x_protect, self.r))

        else:
            z=self.zeta+1.0
            x_protect = x + torch.rand_like(x).uniform_(-z * self.eps, z * self.eps)
            x_protect.requires_grad_(True)
            output_v3 = self.model(x_protect)
        loss1 = F.cross_entropy(output_v3, self.gt)
        g1 = torch.autograd.grad(loss1, x_protect, retain_graph=False, create_graph=False)[0]
        return g1

    def sr(self,x_start, x_end):  # SR
        return x_end - x_start

    def add_predict(self,grads, start, end):  # MGP,prediction horizon
        g = torch.zeros_like(grads[0]).detach().cuda()
        for i in range(start, end):
            g += grads[i]
        return g

    def weighted_random(self,a, b, is_integer=False):#[a, ¦Æ]
        if is_integer:
            values = np.arange(a, b + 1)
            weights = values
        else:
            n_segments = 1000
            values = np.linspace(a, b, n_segments)
            weights = values
        weights = weights / weights.sum()
        return np.random.choice(values, p=weights)
    def get_averaged_gradient(self, data, **kwargs):
        noise=torch.zeros_like(data).detach().cuda()
        for _ in range(self.Nk):
            noise+=self.get_grad(data)
        return noise #Averaging has no effect on direction
    def forward(self, data, **kwargs):
        g_x = torch.zeros_like(data).detach().cuda()
        g_start = torch.zeros_like(data).detach().cuda()
        grad_list = [torch.zeros_like(data).detach().cuda() for _ in range(self.epoch)]
        grad = torch.zeros_like(data).detach().cuda()
        for j in range(self.k-1):#Only the first k-1 paths are taken here
            grad_in = torch.zeros_like(data).detach().cuda()
            x_clone = data.clone().detach().cuda() + self.alpha * torch.sign(g_x)
            for i in range(self.epoch):
                noise=self.get_averaged_gradient(x_clone)
                grad_list[i] += noise/torch.abs(noise).mean([1, 2, 3], keepdim=True)
                grad_in = self.get_momentum(noise, grad_in)
                grad_in = grad_in + self.add_predict(grad_list, i + 1, min(i + self.m, self.epoch))
                x_clone=self.x+self.update_delta(x_clone-self.x,self.x,grad_in,alpha=self.alpha)
            g_x = self.sr(data, x_clone)
            g_start += g_x
        data = data + self.alpha * torch.sign(g_start)
        for i in range(self.epoch):#Integrate all paths on the final path
            noise = self.get_averaged_gradient(data)
            grad_list[i] += noise/torch.abs(noise).mean([1, 2, 3], keepdim=True)
            grad =self.decay * grad+grad_list[i]
            noise = grad + self.add_predict(grad_list, i + 1, min(i + self.m, self.epoch))
            data=self.x+self.update_delta(data-self.x,self.x,noise,alpha=self.alpha)
        return data.detach()

