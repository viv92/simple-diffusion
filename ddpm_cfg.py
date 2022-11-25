### Program implementing Deep Denoising Diffusion Model (DDPM) with Classifier Free Guidance (CFG)

## Features:
# 1. Progressively add gaussian noise (defined noising schedule) to an image as a gaussian process / Markov chain - Forward process
# 2. Progressively denoise the image to obtain original image as a parameterized gaussian process / Markov chain -  Reverse process

# 3. This implementation (ddpm2.py) is different from ddpm.py in the following:
# 3.1. Uses sigma = beta instead of sigma = beta_tilde while sampling
# 3.2. Uses torchvision's dataloader (faster) instead of manual data load function (slower). Also added data-augmentation via random cropping.
# 3.3. Samples n=batch_size instead of n=1

## Todos / Questions:
# 1. img pixels: clamp value in [-1, 1]

# 2. Remember the 4 lines:
## i.   loss = calculate_loss()
## ii.  optimizer.zero_grad()
## iii. loss.backward()
## iv.  optimizer.step()

import os
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from unet import *


def noise_schedule(beta_min, beta_max, max_time_steps):
    return torch.linspace(beta_min, beta_max, max_time_steps)

def noise_img(img_ori, alphas_hat, t, device):
    sqrt_alpha_hat = torch.sqrt(alphas_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alphas_hat[t])[:, None, None, None]
    eps = torch.randn_like(img_ori)
    noised_img = ( sqrt_alpha_hat * img_ori ) + ( sqrt_one_minus_alpha_hat * eps )
    return noised_img, eps


def sample2(net, alphas_hat, betas, alphas, max_time_steps, img_size, device, n, guidance_strength, label):
    net.eval()
    with torch.no_grad():
        x = torch.randn((n, 3, img_size, img_size)).to(device)
        for i in tqdm( reversed(range(1, max_time_steps)), position=0 ):
            t = (torch.ones(n) * i).long().to(device)
            predicted_noise_uncond = net(x, t, None)
            predicted_noise_cond = net(x, t, label)
            # interpolate
            predicted_noise = predicted_noise_cond + guidance_strength * ( predicted_noise_cond - predicted_noise_uncond )
            alpha = alphas[t][:, None, None, None]
            alpha_hat = alphas_hat[t][:, None, None, None]
            beta = betas[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    net.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


# function implementing DDIM sampling (deterministic with sigma = 0)
def sample2_ddim(net, alphas_hat, max_time_steps, subseq_steps, img_size, device, n, guidance_strength, label):
    net.eval()
    with torch.no_grad():
        subseq = torch.linspace(1, max_time_steps-1, subseq_steps, dtype=torch.int)
        x = torch.randn((n, 3, img_size, img_size)).to(device)
        for i in tqdm( reversed(range(1, subseq_steps)), position=0 ):
            tau = (torch.ones(n) * subseq[i]).long().to(device)
            tau_minus1 = (torch.ones(n) * subseq[i-1]).long().to(device)
            predicted_noise_uncond = net(x, tau, None)
            predicted_noise_cond = net(x, tau, label)
            # interpolate
            predicted_noise = predicted_noise_cond + guidance_strength * ( predicted_noise_cond - predicted_noise_uncond )
            alpha_hat = alphas_hat[tau][:, None, None, None]
            alpha_hat_minus1 = alphas_hat[tau_minus1][:, None, None, None]
            if i > 1:
                # sample according to equation 12 from the DDIM paper (with sigma = 0 since deterministic)
                x = torch.sqrt(alpha_hat_minus1) * ((x - torch.sqrt(1-alpha_hat)*predicted_noise)/torch.sqrt(alpha_hat)) + torch.sqrt(1-alpha_hat_minus1) * predicted_noise
            else:
                # just sample the predicted x_0 according to equation 9 in DDIM paper - details in appendix C.1 (equations 56 and 57) in the DDIM paper
                x = (x - torch.sqrt(1-alpha_hat)*predicted_noise)/torch.sqrt(alpha_hat)
    net.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


# note that we sample n=batch_size images for a same label (class label doesn't change between the n samples)
# so we use class label as subfolder name
def save_img2(img, name, label):
    name = 'generated_cfg/' + str(label) + '/' + name
    grid = torchvision.utils.make_grid(img)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(name)

# fetch dataset - using data loader
def get_data(img_size, datapath, batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # equivalent to transforming pixel values from range [0,1] to [-1,1]
    ])
    dataset = torchvision.datasets.ImageFolder(datapath, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


### main
if __name__ == '__main__':
    # hyperparams
    guidance_strength = 3 # w in classifier free guidance paper
    p_uncond = 0.2 # probability for setting class_label = None
    num_classes = 2
    beta_min = 1e-4
    beta_max = 0.02
    max_time_steps = 1000
    subseq_steps = 200
    img_size = 64
    lr = 3e-4
    batch_size = 2
    max_epochs = 30000
    random_seed = 10

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init UNet_conditional
    net = UNet_conditional(num_classes=num_classes, device=device).to(device)

    # calcualate betas and alphas
    betas = noise_schedule(beta_min, beta_max, max_time_steps)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    alphas_hat = alphas_hat.to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)

    # load img dataset
    dataloader = get_data(img_size, './images', batch_size)

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # train
    for ep in tqdm(range(max_epochs)):

        # fetch minibatch
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # set labels = None with prob p_uncond
            if np.random.rand() < p_uncond:
                labels = None

            t = torch.randint(low=1, high=max_time_steps, size=(batch_size,)).to(device) # sample a time step uniformly
            noised_imgs, noise = noise_img(imgs, alphas_hat, t, device) # get noised imgs and the noise

            pred_noise = net(noised_imgs, t, labels)

            loss = mse_loss(noise, pred_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % (max_epochs//100) == 0:
            print('ep:{} \t loss:{}'.format(ep, loss.item()))

            ## sample
            sample_label = np.random.choice(np.arange(num_classes))
            # note that we sample n=batch_size images for a same label (class label doesn't change between the n samples)
            sample_label_tensor = torch.tensor([sample_label]).expand(batch_size).to(device)
            sampled_img = sample2_ddim(net, alphas_hat, max_time_steps, subseq_steps, imgs.shape[-1], device, batch_size, guidance_strength, sample_label_tensor)
            save_img2(sampled_img, str(ep)+'.png', sample_label)
