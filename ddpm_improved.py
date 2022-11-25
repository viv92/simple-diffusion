### Program implementing the Improved Denoising Diffusion Model

## Features (on top of vanilla DDPM with CFG and <optional> DDIM sampling):
# 1. increased diffusion time steps (1000 -> 4000)
# 2. cosine noise schedule
# 3. learnable covariance matrix (actually learns interpolation factor between the two extremem values for sigma: beta and beta_tilde)
# 4. Hybrid loss: L_simple + lambda * L_vlb (with stop gradients on mean factor in L_vlb)
# 5. L_vlb = kl_div for t > 0; L_vlb = - discretized_gaussian_log_likelihood for t = 0
# 6. Strided sampling (in contrast to DDIM sampling)
# 7. In this implementation, UNet predicts [noise, frac_logvar]. And then this predicted noise is used to calculate the mean (for KL loss). But mse loss (L_simple) is between pred_noise and true_noise.

## Todos / Questions:
# 1. Remember the 4 lines.
# 2. Neural net for parameterizing the learnable interpolation factor for sigma
# 3. How to include classifer free guidance in stridded sampling for improved ddpm ?
# 4. Check shapes during interpolation of sigma frac between beta and beta_tilde (do we need broadcasting?)
# 5. alternative: unet predicting [eps, variance] and not [mean, variance] - in openai implementation, they use the predicted eps to calculate predicted x_0 and then use the predicted x_0 to calculate predicted mean. This is unnecessary as we have a formula to calculate predicted mean from predicted eps directly (except if we want to calculate predicted x_0 for some other use - in the calc_bpd_loop function to calculate mse(x_0, predicted_x_0)).
# 6. when calculating q_posterior_mean_variance, the openai implementation is using clipped_log_var instead of log_var - why? Ans: because the formula for beta_tilde is not valid for t = 0. This problem doesn't come up for us as we don't pass t = 0 to the calculate_hybrid_loss function - but then how do we train for t = 0 step ?


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


def linear_noise_schedule(beta_min, beta_max, max_time_steps):
    return torch.linspace(beta_min, beta_max, max_time_steps)

# the function used to calculate cosine factor used in cosine noise schedule
def cosine_func(t, max_time_steps, s=0.008):
    return torch.pow( torch.cos( (((t/max_time_steps)+s) / (1+s)) * torch.tensor(torch.pi/2) ), 2)

def cosine_noise_schedule(max_time_steps):
    betas = []
    # initial values
    f_0 = cosine_func(0, max_time_steps)
    alpha_hat_prev = 1.
    # iterate
    for t in range(1, max_time_steps+1):
        f_t = cosine_func(t, max_time_steps)
        alpha_hat = f_t / f_0
        beta = 1 - (alpha_hat/alpha_hat_prev)
        beta = torch.clamp(beta, min=0., max=0.999)
        betas.append(beta)
        alpha_hat_prev = alpha_hat
    return torch.stack(betas, dim=0)

def noise_img(img_ori, alphas_hat, t, device):
    sqrt_alpha_hat = torch.sqrt(alphas_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alphas_hat[t])[:, None, None, None]
    eps = torch.randn_like(img_ori)
    noised_img = ( sqrt_alpha_hat * img_ori ) + ( sqrt_one_minus_alpha_hat * eps )
    return noised_img, eps

# function to calculate mean and variance of q_posterior: q(x_t-1 | x_t,x_0)
def q_posterior_mean_variance(x_0, x_t, t, t_minus1, alphas_hat, alphas, betas):
    alpha_hat = alphas_hat[t][:, None, None, None]
    alpha_hat_minus1 = alphas_hat[t_minus1][:, None, None, None]
    # beta = betas[t][:, None, None, None] - this is wrong if alphas_hat are strided
    # alpha = alphas[t][:, None, None, None] - this is wrong if alphas_hat are strided
    # so its necessary to calculate beta and alpha from alphas_hat:
    alpha = alpha_hat / alpha_hat_minus1
    beta = 1 - alpha
    mean = ( torch.sqrt(alpha_hat_minus1) * beta * x_0 + torch.sqrt(alpha) * (1 - alpha_hat_minus1) * x_t ) / (1 - alpha_hat)
    var = ( (1 - alpha_hat_minus1) * beta ) / (1 - alpha_hat)
    logvar = torch.log(var)
    return mean, logvar

# function to calculate mean and variance of p(x_t-1 | x_t)
def p_mean_variance(net, x_t, t, t_minus1, labels, alphas_hat):
    out = net(x_t, t, labels) # the unet predicts the concatenated [mean, frac]
    img_channels = x_t.shape[1]
    pred_noise, frac = torch.split(out, img_channels, dim=1)
    # clamp frac values to be in [0, 1]
    # frac = frac.clamp(-1, 1)
    frac = (frac + 1) / 2.0
    # frac = frac * 0
    # get log variance by interpolating between min_log_var (beta_tilde) and max_log_var (beta)
    alpha_hat = alphas_hat[t][:, None, None, None]
    alpha_hat_minus1 = alphas_hat[t_minus1][:, None, None, None]
    # beta = betas[t][:, None, None, None] - this is wrong if alphas_hat are strided
    # so its necessary to calculate beta from alphas_hat:
    beta = 1 - (alpha_hat / alpha_hat_minus1)
    alpha = 1 - beta
    beta_tilde = ( (1 - alpha_hat_minus1) * beta ) / (1 - alpha_hat)
    max_logvar = torch.log(beta)
    min_logvar = torch.log(beta_tilde)
    logvar = frac * max_logvar + (1 - frac) * min_logvar
    # calculate mean from pred_noise
    mean = ( x_t - ((beta * pred_noise) / torch.sqrt(1 - alpha_hat)) ) / torch.sqrt(alpha)
    return mean, logvar, pred_noise

# function to calculate KL div between two gaussians - TODO: check dims for diagonal variance
# used to calculate L_t = KL(q_posterior, p)
def kl_normal(mean_q, logvar_q, mean_p, logvar_p):
    # stop gradient on means
    mean_q, mean_p = mean_q.detach(), mean_p.detach()
    return 0.5 * ( -1.0 + logvar_p - logvar_q + torch.exp(logvar_q - logvar_p) + ((mean_q - mean_p)**2) * torch.exp(-logvar_p) )

### functions to calculate L_0 = -log p(x_0 | x_1) - borrowed from OpenAi implementation of improved DDPM

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

# utility function to take mean of a tensor across all dimensions except the first (batch) dimension
def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.shape))))

# function to calculate hybrid loss: L_hybrid = L_simple + lambda * L_vlb
def calculate_hybrid_loss(net, x_0, t, labels, L_lambda, alphas_hat, alphas, betas, device):
    x_t, true_noise = noise_img(x_0, alphas_hat, t, device)
    q_mean, q_logvar = q_posterior_mean_variance(x_0, x_t, t, t-1, alphas_hat, alphas, betas)
    p_mean, p_logvar, pred_noise = p_mean_variance(net, x_t, t, t-1, labels, alphas_hat)
    # if t == 1:
    p_log_scale = 0.5 * p_logvar
    L_vlb_0 = -1 * discretized_gaussian_log_likelihood(x_0, p_mean, p_log_scale)
    L_vlb_0 = L_vlb_0 / torch.log(torch.tensor(2.0)) # convert loss from nats to bits
    L_vlb_0 = mean_flat(L_vlb_0) # take mean across all dims except batch_dim # shape: [batch_size]
    L_simple_0 = torch.pow(pred_noise - true_noise, 2)
    L_simple_0 = mean_flat(L_simple_0)
    L_hybrid_0 = L_vlb_0
    # L_hybrid_0 = L_simple_0

    # if t > 1:
    L_simple = torch.pow(pred_noise - true_noise, 2) # mse loss but don't want to reduce mean or sum
    L_simple = mean_flat(L_simple) # take mean across all dims except batch_dim # shape: [batch_size]
    L_vlb_t = kl_normal(q_mean, q_logvar, p_mean, p_logvar)
    L_vlb_t = L_vlb_t / torch.log(torch.tensor(2.0)) # convert loss from nats to bits
    L_vlb_t = mean_flat(L_vlb_t) # take mean across all dims except batch_dim # shape: [batch_size]
    L_hybrid_t = L_simple + L_lambda * L_vlb_t
    # populate final loss vector according to t values
    L_hybrid = torch.where((t == 1), L_hybrid_0, L_hybrid_t) # shape: [batch_size]
    L_hybrid = L_hybrid.mean() # final loss scalar
    return L_hybrid


# function to sample x_t-1 ~ p(x_t-1 | x_t)
def p_sample_CFG(i, net, x_t, t, t_minus1, labels, alphas_hat, guidance_strength):
    mean_cond, logvar_cond, pred_noise_cond = p_mean_variance(net, x_t, t, t_minus1, labels, alphas_hat)
    mean_uncond, logvar_uncond, pred_noise_uncond = p_mean_variance(net, x_t, t, t_minus1, None, alphas_hat)
    # calculated interpolated mean (weighted by guidance strength)
    mean_interpolated = mean_cond + guidance_strength * ( mean_cond - mean_uncond )
    # sample
    eps = torch.randn_like(x_t)
    if i == 1:
        eps = eps * 0
    x_t_minus1 = mean_interpolated + torch.exp(0.5 * logvar_cond) * eps
    return x_t_minus1

# function to sample x_t-1 ~ q(x_t-1 | x_t, x_0)
def q_sample(i, net, x_0, x_t, t, t_minus1, alphas_hat, betas):
    mean, logvar = q_posterior_mean_variance(x_0, x_t, t, t_minus1, alphas_hat, alphas, betas)
    eps = torch.randn_like(x_t)
    if i == 1:
        eps = eps * 0
    x_t_minus1 = mean + torch.exp(0.5 * logvar) * eps
    return x_t_minus1

# strided sampling (TODO - adding classifier free guidance) - DONE 
def sample_strided_CFG(net, x_0, alphas_hat, guidance_strength, max_time_steps, subseq_steps, img_size, device, n, label):
    net.eval()
    with torch.no_grad():
        subseq = torch.linspace(0, max_time_steps-1, subseq_steps, dtype=torch.int)
        x = torch.randn((n, 3, img_size, img_size)).to(device)
        x_q = torch.randn((n, 3, img_size, img_size)).to(device)
        for i in tqdm( reversed(range(1, subseq_steps)), position=0 ):
            tau = (torch.ones(n) * subseq[i]).long().to(device)
            tau_minus1 = (torch.ones(n) * subseq[i-1]).long().to(device)
            x = p_sample_CFG(i, net, x, tau, tau_minus1, label, alphas_hat, guidance_strength)
    net.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


def sample2(net, alphas_hat, betas, alphas, max_time_steps, img_size, device, n):
    net.eval()
    with torch.no_grad():
        x = torch.randn((n, 3, img_size, img_size)).to(device)
        for i in tqdm( reversed(range(1, max_time_steps)), position=0 ):
            t = (torch.ones(n) * i).long().to(device)

            mean_cond, logvar_cond, predicted_noise = p_mean_variance(net, x, t, t-1, None, alphas_hat)

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


# function implementing DDIM sampling (using the predicted value of sigma)
def sample_ddim_predictedSigma(net, alphas_hat, max_time_steps, subseq_steps, img_size, device, n, guidance_strength, label):
    net.eval()
    with torch.no_grad():
        subseq = torch.linspace(1, max_time_steps-1, subseq_steps, dtype=torch.int)
        x = torch.randn((n, 3, img_size, img_size)).to(device)
        for i in tqdm( reversed(range(1, subseq_steps)), position=0 ):

            tau = (torch.ones(n) * subseq[i]).long().to(device)
            tau_minus1 = (torch.ones(n) * subseq[i-1]).long().to(device)
            alpha_hat = alphas_hat[tau][:, None, None, None]
            alpha_hat_minus1 = alphas_hat[tau_minus1][:, None, None, None]

            mean_cond, logvar_cond, predicted_noise_cond = p_mean_variance(net, x, tau, tau_minus1, label, alphas_hat)
            mean_uncond, logvar_uncond, predicted_noise_uncond = p_mean_variance(net, x, tau, tau_minus1, None, alphas_hat)
            # interpolate
            predicted_noise = predicted_noise_cond + guidance_strength * ( predicted_noise_cond - predicted_noise_uncond )
            # predicted sigma
            sigma = torch.exp(0.5 * logvar_cond)
            # sigma = sigma * 0

            if i > 1:
                sampling_noise = torch.randn_like(x)
                # sample according to equation 12 from the DDIM paper (with predicted sigma)
                x = torch.sqrt(alpha_hat_minus1) * ( (x - torch.sqrt(1-alpha_hat)*predicted_noise) / torch.sqrt(alpha_hat) ) + torch.sqrt( 1 - alpha_hat_minus1 - torch.pow(sigma,2) ) * predicted_noise + (sigma * sampling_noise)
            else:
                # just sample the predicted x_0 according to equation 9 in DDIM paper - details in appendix C.1 (equations 56 and 57) in the DDIM paper
                x = (x - torch.sqrt(1-alpha_hat)*predicted_noise)/torch.sqrt(alpha_hat)
    net.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


# function implementing DDIM sampling (deterministic with sigma = 0)
def sample_ddim(net, alphas_hat, max_time_steps, subseq_steps, img_size, device, n, guidance_strength, label):
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
def save_img_noCFG(img, name):
    name = 'generated_improved_predNoise_noCFG/' + name
    grid = torchvision.utils.make_grid(img)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(name)

def save_img_ddim_noCFG(img, name):
    name = 'generated_improved_strip4_1_2/' + name
    grid = torchvision.utils.make_grid(img)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(name)

def save_img_CFG(img, name, label):
    name = 'generated_improved_strip4_1_2_CFG/' + str(label) + '/' + name
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
    learnable_sigma = True # for improved ddpm
    L_lambda = 0.001 # for weighing L_vlb
    guidance_strength = 3 # w in classifier free guidance paper
    p_uncond = 0.2 # probability for setting class_label = None
    num_classes = 2
    beta_min = 1e-4 # not needed for cosine noise schedule
    beta_max = 0.02 # not needed for cosine noise schedule
    max_time_steps = 1000 # 4000
    subseq_steps = 200 # used for both strided sampling and ddim sampling - should be less than max_time_steps
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
    out_channels = 3
    if learnable_sigma:
        out_channels *= 2
    net = UNet_conditional(c_out=out_channels, num_classes=num_classes, device=device).to(device)

    # calcualate betas and alphas
    betas = linear_noise_schedule(beta_min, beta_max, max_time_steps)
    # betas = cosine_noise_schedule(max_time_steps)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    alphas_hat = alphas_hat.to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)

    # load img dataset
    dataloader = get_data(img_size, './images', batch_size)

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr)

    # train
    for ep in tqdm(range(max_epochs)):

        # fetch minibatch
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # set labels = None with prob p_uncond
            if np.random.rand() < p_uncond:
                labels = None

            t = torch.randint(low=1, high=max_time_steps, size=(batch_size,)).to(device) # sample a time step uniformly in [1, max_time_steps)

            loss = calculate_hybrid_loss(net, imgs, t, labels, L_lambda, alphas_hat, alphas, betas, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % (max_epochs//100) == 0:
            print('ep:{} \t loss:{}'.format(ep, loss.item()))

            ## sample

            sample_label = np.random.choice(np.arange(num_classes))
            # note that we sample n=batch_size images for a same label (class label doesn't change between the n samples)
            sample_label_tensor = torch.tensor([sample_label]).expand(batch_size).to(device)

            # sampled_img = sample_ddim(net, alphas_hat, max_time_steps, subseq_steps, imgs.shape[-1], device, batch_size, guidance_strength, sample_label_tensor)
            sampled_img = sample_strided_CFG(net, imgs, alphas_hat, guidance_strength, max_time_steps, subseq_steps, imgs.shape[-1], device, batch_size, sample_label_tensor)
            # sampled_img = sample_ddim_predictedSigma(net, alphas_hat, max_time_steps, subseq_steps, imgs.shape[-1], device, batch_size, guidance_strength, sample_label_tensor)
            # sampled_img = sample2(net, alphas_hat, betas, alphas, max_time_steps,  imgs.shape[-1], device, batch_size)
            # save_img_ddim_noCFG(sampled_img, str(ep)+'.png')
            save_img_CFG(sampled_img, str(ep)+'.png', sample_label)
