from IQA_pytorch import SSIM, MS_SSIM, VIF, FSIM, utils
from PIL import Image
import torch
import glob
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ref_images = glob.glob("Clean/*.png")

adv_images1 = glob.glob("Adversarial_Folder_1/*.png")
adv_images2 = glob.glob("Adversarial_Folder_2/*.png")
adv_images4 = glob.glob("Adversarial_Folder_4/*.png")
adv_images8 = glob.glob("Adversarial_Folder_8/*.png")
adv = [adv_images1, adv_images2, adv_images4, adv_images8]

high_images1 = glob.glob("High_Folder_1/*.png")
high_images2 = glob.glob("High_Folder_2/*.png")
high_images4 = glob.glob("High_Folder_4/*.png")
high_images8 = glob.glob("High_Folder_8/*.png")
high = [high_images1, high_images2, high_images4, high_images8]

low_images1 = glob.glob("Low_Folder_1/*.png")
low_images2 = glob.glob("Low_Folder_2/*.png")
low_images4 = glob.glob("Low_Folder_4/*.png")
low_images8 = glob.glob("Low_Folder_8/*.png")
low = [low_images1, low_images2, low_images4, low_images8]

epsilon = [1, 2, 4, 8]
ADV = []
HIGH = []
LOW = []

score1 = 0
score2 = 0
score3 = 0

model = FSIM(channels=3)

for i in adv:
    for img1, img2 in zip(ref_images, i):
        ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        score = model(dist, ref, as_loss=False)
        score1 += score

    score1 = score1.item() / 1000
    ADV.append(score1)

for i in high:
    for img1, img2 in zip(ref_images, i):
        ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        score = model(dist, ref, as_loss=False)
        score2 += score

    score2 = score2.item() / 1000
    HIGH.append(score2)

for i in low:
    for img1, img2 in zip(ref_images, i):
        ref = utils.prepare_image(Image.open(img1).convert("RGB")).to(device)
        dist = utils.prepare_image(Image.open(img2).convert("RGB")).to(device)
        score = model(dist, ref, as_loss=False)
        score3 += score

    score3 = score3.item() / 1000
    LOW.append(score3)

print(ADV)
print(HIGH)
print(LOW)

plt.plot(epsilon, ADV, color='yellow', marker='.', label='MIM')
plt.plot(epsilon, HIGH, color='red', marker='.', label='High Frequency')
plt.plot(epsilon, LOW, color='blue', marker='.', label='Low Frequency')
plt.xlabel('epsilon')
plt.ylabel('FSIM')
plt.title('FSIM(iter=1)')
plt.show()

acc1 = [1-(1.2/93.0), 1-(0.5/93.0), 1-(0.4/93.0), 1-(0.1/93.0)]
acc3 = [1-(3.1/93.0), 1-(1.6/93.0), 1-(1.4/93.0), 1-(1.3/93.0)]
acc2 = [1-(10.1/93.0), 1-(6.1/93.0), 1-(5.0/93.0), 1-(4.8/93.0)]

plt.plot(epsilon, acc1, color='yellow', marker='.', label='MIM')
plt.plot(epsilon, acc2, color='red', marker='.', label='High Frequency')
plt.plot(epsilon, acc3, color='blue', marker='.', label='Low Frequency')
plt.xlabel('epsilon')
plt.ylabel('ASR')
plt.title('Attack Success Rate(iter=1)')
plt.show()

plt.plot(ADV, acc1, color='yellow', marker='.', label='MIM')
plt.plot(HIGH, acc2, color='red', marker='.', label='High Frequency')
plt.plot(LOW, acc3, color='blue', marker='.', label='Low Frequency')
plt.xlabel('FSIM')
plt.ylabel('ASR')
plt.title('ASR vs. FSIM(iter=1)')
plt.show()