from IQA_pytorch import SSIM, MS_SSIM, VIF, FSIM, utils
from PIL import Image
import torch
import glob
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MS_SSIM(channels=3)

ref_images = glob.glob("Clean/*.png")

#low_images64 = glob.glob("final_Low_64_8_1/*.png")
#low5_images64 = glob.glob("final_Low_64_8_5/*.png")
#low_images128 = glob.glob("final_Low_128_8_1/*.png")
#low5_images128 = glob.glob("final_Low_128_8_5/*.png")
mim_images = glob.glob("final_MIM_8_1/*.png")
mim5_images = glob.glob("final_MIM_8_5/*.png")
#ADV = [low_images64, low5_images64, low_images128, low5_images128, mim_images, mim5_images]

#score1 = 0
#adv = []

#for i in ADV:
    #for img1, img2 in zip(ref_images, i):
        #ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        #dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        #score = model(dist, ref, as_loss=False)
        #score1 += score
    #score1 = score1.item() / 1000
    #adv.append(score1)
    #score1 = 0

#print(adv)

high_images20 = glob.glob("final_High_20_8_1/*.png")
high_images50 = glob.glob("final_High_50_8_1/*.png")
high_images100 = glob.glob("final_High_100_8_1/*.png")
high_images150 = glob.glob("final_High_150_8_1/*.png")
high_images200 = glob.glob("final_High_200_8_1/*.png")
high_images214 = glob.glob("final_High_214_8_1/*.png")
HIGH = [mim_images, high_images20, high_images50, high_images100, high_images150, high_images200, high_images214]

high5_images20 = glob.glob("final_High_20_8_5/*.png")
high5_images50 = glob.glob("final_High_50_8_5/*.png")
high5_images100 = glob.glob("final_High_100_8_5/*.png")
high5_images150 = glob.glob("final_High_150_8_5/*.png")
high5_images200 = glob.glob("final_High_200_8_5/*.png")
high5_images214 = glob.glob("final_High_214_8_5/*.png")
HIGH5 = [mim5_images, high5_images20, high5_images50, high5_images100, high5_images150, high5_images200, high5_images214]

low_images20 = glob.glob("final_Low_20_8_1/*.png")
low_images50 = glob.glob("final_Low_50_8_1/*.png")
low_images100 = glob.glob("final_Low_100_8_1/*.png")
low_images150 = glob.glob("final_Low_150_8_1/*.png")
low_images200 = glob.glob("final_Low_200_8_1/*.png")
low_images214 = glob.glob("final_Low_214_8_1/*.png")
LOW = [mim_images, low_images20, low_images50, low_images100, low_images150, low_images200, low_images214]

low5_images20 = glob.glob("final_Low_20_8_5/*.png")
low5_images50 = glob.glob("final_Low_50_8_5/*.png")
low5_images100 = glob.glob("final_Low_100_8_5/*.png")
low5_images150 = glob.glob("final_Low_150_8_5/*.png")
low5_images200 = glob.glob("final_Low_200_8_5/*.png")
low5_images214 = glob.glob("final_Low_214_8_5/*.png")
LOW5 = [mim5_images, low5_images20, low5_images50, low5_images100, low5_images150, low5_images200, low5_images214]

score2 = 0
high = []

for i in HIGH:
    for img1, img2 in zip(ref_images, i):
        ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        score = model(dist, ref, as_loss=False)
        score2 += score
    score2 = score2.item() / 1000
    high.append(score2)
    score2 = 0

score3 = 0
high5 = []

for i in HIGH5:
    for img1, img2 in zip(ref_images, i):
        ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        score = model(dist, ref, as_loss=False)
        score3 += score
    score3 = score3.item() / 1000
    high5.append(score3)
    score3 = 0

score3 = 0
low = []

for i in LOW:
    for img1, img2 in zip(ref_images, i):
        ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        score = model(dist, ref, as_loss=False)
        score3 += score
    score3 = score3.item() / 1000
    low.append(score3)
    score3 = 0

score4 = 0
low5 = []

for i in LOW5:
    for img1, img2 in zip(ref_images, i):
        ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        score = model(dist, ref, as_loss=False)
        score4 += score
    score4 = score4.item() / 1000
    low5.append(score4)
    score4 = 0


dim = [0, 20, 50, 100, 150, 200, 214]
MS_SSIM_MIM = [0.9539013671875, 0.9539013671875, 0.9539013671875, 0.9539013671875, 0.9539013671875, 0.9539013671875, 0.9539013671875]
MS_SSIM_MIM5 = [0.9656668701171875, 0.9656668701171875, 0.9656668701171875, 0.9656668701171875, 0.9656668701171875, 0.9656668701171875, 0.9656668701171875]

mim_acc = [1-(64.7/93.0), 1-(64.7/93.0), 1-(64.7/93.0), 1-(64.7/93.0), 1-(64.7/93.0), 1-(64.7/93.0), 1-(64.7/93.0)]
mim5_acc = [1-(19.5/93.0), 1-(19.5/93.0), 1-(19.5/93.0), 1-(19.5/93.0), 1-(19.5/93.0), 1-(19.5/93.0), 1-(19.5/93.0)]

high_acc = [1-(64.7/93.0), 1-(66.3/93.0), 1-(66.6/93.0), 1-(73.5/93.0), 1-(73.0/93.0), 1-(80.7/93.0), 1-(82.2/93.0)]
high5_acc = [1-(19.5/93.0), 1-(20.2/93.0), 1-(20.7/93.0), 1-(44.4/93.0), 1-(40.4/93.0), 1-(77.8/93.0), 1-(79.5/93.0)]

low_acc = [1-(64.7/93.0), 1-(66.1/93.0), 1-(65.9/93.0), 1-(66.2/93.0), 1-(67.9/93.0), 1-(67.5/93.0), 1-(67.1/93.0)]
low5_acc = [1-(19.5/93.0), 1-(20.1/93.0), 1-(20.9/93.0), 1-(21.7/93.0), 1-(25.6/93.0), 1-(27.7/93.0), 1-(31.6/93.0)]

plt.plot(dim, low, color='blue', marker='.', label='Low')
plt.plot(dim, MS_SSIM_MIM, color='green', marker='.', label='MIM')
plt.plot(dim, high, color='red', marker='.', label='High')
plt.xlabel('marked dimension')
plt.ylabel('MS-SSIM')
plt.title('MS-SSIM (iter=1)')
plt.legend()
plt.show()

plt.plot(dim, low5, color='blue', marker='.', label='Low')
plt.plot(dim, MS_SSIM_MIM5, color='green', marker='.', label='MIM')
plt.plot(dim, high5, color='red', marker='.', label='High')
plt.xlabel('marked dimension')
plt.ylabel('MS-SSIM')
plt.title('MS-SSIM (iter=5)')
plt.legend()
plt.show()

plt.plot(dim, low_acc, color='blue', marker='.', label='Low')
plt.plot(dim, mim_acc, color='green', marker='.', label='MIM')
plt.plot(dim, high_acc, color='red', marker='.', label='High')
plt.xlabel('marked dimension')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Attack Success Rate (iter=1)')
plt.legend()
plt.show()

plt.plot(dim, low5_acc, color='blue', marker='.', label='Low')
plt.plot(dim, mim5_acc, color='green', marker='.', label='MIM')
plt.plot(dim, high5_acc, color='red', marker='.', label='High')
plt.xlabel('marked dimension')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Attack Success Rate (iter=5)')
plt.legend()
plt.show()