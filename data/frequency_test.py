import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2 as cv
import numpy as np
#from scipy.fftpack import dct, idct
import torch_dct as dct
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import os

import gc
gc.collect()
torch.cuda.empty_cache()


def load(path):
    img = Image.open(path)
    return img


to_tensor = transforms.ToTensor()

to_pil = transforms.ToPILImage()


def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()[0]


def np_to_torch(img_np):
    return torch.from_numpy(img_np)[None, :]


def reverse_channels(img):
    return np.moveaxis(img, 0, -1)  # source, dest


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class ReNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            #            t.mul_(s).add_(m)
            t.sub_(m).div_(s)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
renorm = ReNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t

# implement 2D DCT
#def dct2(a):
    #return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
#def idct2(a):
    #return idct(idct(a.T, norm='ortho').T, norm='ortho')

# %% Loading Data
path = 'clean_images'

data = np.zeros((1000, 3, 224, 224)).astype('float32')  # Reshape the data as per the input size of Models
# ResNet takes an input dimension of 224 x 224
labels = (np.zeros(1000))
labels = labels.astype(int)
labels_target = (np.zeros(1000))
labels_target = labels_target.astype(int)

images = pd.read_csv("images.csv")
# data__.head()
# data__.iloc[0,0]

for i in range(1000):
    ImgID = images.iloc[i, 0]
    img = load(path + '/' + ImgID + '.png')
    img = img.resize((224, 224), Image.ANTIALIAS)  # Change the input image shape here
    img = to_tensor(img)[None, :]

    img_np = torch_to_np(img)
    #img_np = img_np[:3, :, :]
    data[i, :, :, :] = img_np
    labels[i] = images.iloc[i, 6]
    labels_target[i] = images.iloc[i, 7]
# %%
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (True and torch.cuda.is_available()) else "cpu")

# Choose the pytorch model for which you want to generate adversarial attack
resnet = models.resnet50(pretrained=True)
if True:
    resnet = nn.DataParallel(resnet).cuda()
resnet.eval()

dtype = torch.cuda.FloatTensor
correct = 0
with torch.no_grad():
    for i in range(1000):
        img = data[i]
        img = torch.from_numpy(img)[None, :]
        img_norm = renorm(img.clone())  # Normalize the Images as per ImageNet Dataset
        outputs = resnet(img_norm.type(dtype))
        predictions = outputs.data.max(1)[1]
        p = predictions.detach().cpu().numpy()[0]
        correct += (p + 1 == labels[i]).sum()  # add 1 because 0 is used for background in the NIPS_Dev Dataset
acc = correct * 100. / 1000

print("Accuracy of ResNet 50 Model:", acc)


# %% Loading Pre-trained Models
# resnet=models.resnet50(pretrained=True)
# if True:
#        resnet = nn.DataParallel(resnet).cuda()
# resnet.eval()

# inception = models.inception_v3(pretrained=True)
# if True:
#        inception = nn.DataParallel(inception).cuda()
# inception.eval()

# densenet=models.densenet121(pretrained=True)
# if True:
#        densenet = nn.DataParallel(densenet).cuda()
# densenet.eval()
#
# alexnet=models.alexnet(pretrained=True)
# if True:
#        alexnet = nn.DataParallel(alexnet).cuda()
# alexnet.eval()

# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters):
    adv = img.detach()
    adv_pert = torch.zeros_like(adv).to(device).detach()

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

        noise = 0

    for j in range(iterations):
        adv.requires_grad = True
        out_adv = model(normalize(adv.clone()))
        loss = criterion(out_adv, label)
        loss.backward()

        adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
        adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
        adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
        adv.grad = adv.grad / adv_mean
        # noise = noise + adv.grad
        pert = eps * 255 * (adv.grad).sign()

        img_R = pert[:, 0, :, :]
        img_G = pert[:, 1, :, :]
        img_B = pert[:, 2, :, :]

        dct_R = dct.dct_2d(img_R)
        dct_G = dct.dct_2d(img_G)
        dct_B = dct.dct_2d(img_B)

        _, rows_R, cols_R = dct_R.shape
        _, rows_G, cols_G = dct_G.shape
        _, rows_B, cols_B = dct_B.shape

        # low-frequency masking
        dct_R[:, 101:rows_R, :] = 0
        dct_R[:, :, 101:cols_R] = 0
        dct_G[:, 101:rows_G, :] = 0
        dct_G[:, :, 101:cols_G] = 0
        dct_B[:, 101:rows_B, :] = 0
        dct_B[:, :, 101:cols_B] = 0

        # high-frequency masking
        #dct_R[:, :21, :21] = 0
        #dct_G[:, :21, :21] = 0
        #dct_B[:, :21, :21] = 0

        adv_pert[:, 0, :, :] = dct.idct_2d(dct_R)
        adv_pert[:, 1, :, :] = dct.idct_2d(dct_G)
        adv_pert[:, 2, :, :] = dct.idct_2d(dct_B)

        adv_pert = adv_pert.detach()
        adv_pert.requires_grad = True
        adv.requires_grad = False

        # out_adv1 = model(normalize(adv_pert.clone()))
        # loss1 = criterion(out_adv1, label)
        # loss1.backward()

        adv2 = adv.detach() + adv_pert
        #adv2 = adv2.detach()
        #adv2.requires_grad = True

        out_adv2 = model(normalize(adv2.clone()))
        loss2 = criterion(out_adv2, label)
        loss2.backward()

        if attack_type == 'mim':
            adv_mean = torch.mean(torch.abs(adv_pert.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv_pert.grad = adv_pert.grad / adv_mean
            noise = noise + adv_pert.grad

        else:
            noise = adv.grad

        # Optimization step
        adv.data = adv.data + step * noise.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()


# %% Attack on ResNet
adv_folder = 'final_Low_200_8_5'  # Change the name of the folder accordingly
if not os.path.isdir(adv_folder):
    os.makedirs(adv_folder)

adv_acc = 0
clean_acc = 0

criterion = nn.CrossEntropyLoss()
eps = 8 / 255  # The value of Epsilon

for i in range(1000):
    img = data[i]
    img = torch.from_numpy(img)[None, :]
    label = labels[i] - 1
    label = torch.tensor([label], dtype=torch.int64)
    img, label = img.to(device), label.to(device)

    clean_acc += torch.sum(resnet(normalize(img.clone().detach())).argmax(dim=-1) == label).item()
    adv = attack(resnet, criterion, img, label, eps=eps, attack_type='mim', iters=5)  # Batch-wise attack
    # options for attack_type are 'fgsm', 'bim', 'mim' and 'pgd'
    # Change the No of Iters according to the attack_type
    # -- eg for attack_type= 'fgsm', iters= 1
    # -- eg for attack_type= 'bim', iters= 10    and so on

    pred = resnet(normalize(adv.clone().detach())).argmax(dim=-1)
    pred_np = pred.detach().cpu().numpy()[0]

    # Saving the Adversarial Files
    ImgID = images.iloc[i, 0]
    adv_im = torch_to_np(adv)
    adv_im = reverse_channels(adv_im)
    plt.imsave(adv_folder + '/' + ImgID + '_adv.png', adv_im)

    adv_acc += torch.sum(pred == label).item()
    if i == 2:  # Save one Image/Adversarial pair as a Sample
        vutils.save_image(vutils.make_grid(torch.cat((img, adv), 0), normalize=False, scale_each=True), 'sample.png')

    print('img: {0}'.format(i))
print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc / 1000, adv_acc / 1000))



