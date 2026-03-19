import torch
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
import pretrainedmodels
from torch import nn
import timm


mean = [0.485, 0.456, 0.406]
std = [0.485, 0.456, 0.406]
from torchvision import models as models

def unadversarial(adv_dir,csv_path,net_name):
    if net_name == 'resnet34':
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                models.resnet34(weights=models.ResNet34_Weights).eval().cuda())
    elif net_name == 'resnet152':
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                models.resnet152(weights=models.ResNet152_Weights).eval().cuda())
    elif net_name == 'regnet_x_800mf':
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights).eval().cuda())
    elif net_name == 'convnext_large':
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                models.convnext_large(weights=models.ConvNeXt_Large_Weights).eval().cuda())
    elif net_name == 'maxvit_t':
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                models.maxvit_t(weights=models.MaxVit_T_Weights).eval().cuda())
    elif net_name=="resnet18":
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                    models.resnet18(weights=models.ResNet18_Weights).eval().cuda())
    elif net_name=="swin_v2_b":
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                    models.swin_v2_b(weights=models.Swin_V2_B_Weights).eval().cuda())
    elif net_name=="inception_v3":
        model = torch.nn.Sequential(T.Normalize(mean, std),
                                    models.inception_v3(weights=models.Inception_V3_Weights).eval().cuda())
    elif net_name=="inception_v4":
        model=torch.nn.Sequential(Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),pretrainedmodels.inceptionv4(num_classes=1000,pretrained="imagenet").eval().cuda())
    elif net_name == "inception_v2":
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained="imagenet").eval().cuda())
    elif net_name== "pit":
        model=torch.nn.Sequential(T.Normalize(mean, std),timm.create_model('pit_b_224',pretrained=True).eval().cuda())
    elif net_name== "visformer":
        model=torch.nn.Sequential(T.Normalize(mean, std),timm.create_model('visformer_small',pretrained=True).eval().cuda())
    elif net_name== "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k":
        model=torch.nn.Sequential(T.Normalize(mean, std),timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',pretrained=True).eval().cuda())
    else:
        print('Wrong model name!')
        return
    size=299
    if net_name == 'maxvit_t' or net_name == 'pit' or net_name == 'visformer' :
        size=224
    if net_name == 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k':
        size = 448
    images = ImageNet(adv_dir, csv_path=csv_path,size=size)
    data_loader = DataLoader(images, batch_size=1, shuffle=False, pin_memory=True)
    total = len(images)
    correct = 0
    with torch.no_grad():
        for imag, image_id, gt_cpu in tqdm(data_loader):
            gt = gt_cpu.cuda()
            image = imag.cuda()
            output = model(image)
            output = F.softmax(output, dim=1)
            _, pred = torch.max(output, dim=1)
            correct += 1 if (gt) == pred else 0
            print(correct)
    accuracy = correct / total
    print(f"ASR is {1-accuracy:.4f}")
from torch_net import tf2torch_ens4_adv_inc_v3,tf2torch_ens3_adv_inc_v3,tf2torch_ens_adv_inc_res_v2
def adversarial(adv_dir,csv_path,net_name):
    if net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('torch'),
            net.KitModel("./nets_weight/tf2torch_ens3_adv_inc_v3.npy").eval().cuda(),)
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('torch'),
            net.KitModel("./nets_weight/tf2torch_ens4_adv_inc_v3.npy").eval().cuda(),)
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('torch'),
            net.KitModel("./nets_weight/tf2torch_ens_adv_inc_res_v2.npy").eval().cuda(),)
        size = 299
        images = ImageNet(adv_dir, csv_path=csv_path, size=size)
        data_loader = DataLoader(images, batch_size=1, shuffle=False, pin_memory=True)
        total = len(images)
        correct = 0
        with torch.no_grad():
            for imag, image_id, gt_cpu in tqdm(data_loader):
                gt = gt_cpu.cuda()
                image = imag.cuda()
                output = model(image)
                output = F.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                correct += 1 if (gt+1) == pred[0] else 0
                print(correct)
        accuracy = correct / total
        print(f"ASR is {1-accuracy:.4f}")
        return
    else:
        print('Wrong model name!')
        return
    transforms = T.Compose([T.Resize(299), T.ToTensor()])
    size=299
    images = ImageNet(adv_dir, csv_path=csv_path,size=size)
    data_loader = DataLoader(images, batch_size=1, shuffle=False, pin_memory=True)
    total = len(images)
    correct = 0
    with torch.no_grad():
        for imag, image_id, gt_cpu in tqdm(data_loader):
            gt = gt_cpu.cuda()
            image = imag.cuda()
            output = model(image)
            output = F.softmax(output, dim=0)
            _, pred = torch.max(output, dim=0)
            correct += 1 if (gt+1) == pred else 0
    accuracy = correct / total
    print(f"ASR is {1-accuracy:.4f}")

def main():
    adv_dir= "./MIF" #Path for generating adversarial examples
    csv_path="./dataset/labels"#Image label path
    net_name="pit"#Test model name
    unadversarial(adv_dir, csv_path,net_name)#Test
if __name__ == "__main__":
    main()