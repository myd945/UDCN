import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from guide_filter import *
import conf
import time
from image_op import erosion
from typing import Tuple, List
import torch
from model import QA
import os
from filter import guss

esp = 1e-6


def rgb_to_grayscale(s):
    return (0.2989*s[:,0,:,:]+ 0.5870*s[:,1,:,:] + 0.1140*s[:,2,:,:]).unsqueeze(1)

def gradient(input_tensor, direction):
    smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32, device='cuda:0'), [2, 1, 2, 1])
    smooth_kernel_y = smooth_kernel_x.permute(2, 1, 0, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    return torch.abs(F.conv2d(input_tensor, kernel, stride=(1, 1), padding=1))


def smooth(I, R, r):
    # R1 = torch.squeeze(R, 0)
    R = rgb_to_grayscale(R)
    return torch.mean(gradient(I, "x") * torch.exp(-r * gradient(R, "x")) + gradient(I, "y") * torch.exp(-r * gradient(R, "y")))
    # return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.gradient(input_R, "y")))





def lowLightLoss(input_im, R, L, im_eq):
    L_3 = torch.cat((L, L, L), dim=1)
    recon_loss_low = torch.mean(torch.abs(R * L_3 - input_im))
    R_low_max = torch.max(R, dim=1, keepdims=True)
    recon_loss_low_eq = torch.mean(torch.abs(R_low_max[0] - im_eq))
    # R1 = R.detach()
    R1 = torch.squeeze(R, 0)
    # R1 = torch.reshape(R1, (400, 600, 3))
    # R1 = R1.numpy()
    # print(R1.shape)
    a = gradient(rgb_to_grayscale(R1), "x")
    b = gradient(rgb_to_grayscale(R1), "y")

    # print(shp_a)
    # print(shp_b)
    R_low_loss_smooth = torch.mean(torch.abs(a) + torch.abs(b))
    Ismooth_loss_low = smooth(L, R)
    loss_Decom_zhangyu = recon_loss_low + 0.1 * Ismooth_loss_low + 0.1 * recon_loss_low_eq + 0.01 * R_low_loss_smooth
    return loss_Decom_zhangyu


def haze_loss(out, src):
    # I = rgb_to_grayscale(src)
    minx, _ = torch.min(src, dim=1)
    mean = torch.mean(src, dim=1).unsqueeze(1)
    minx = torch.pow(minx.unsqueeze(1), 0.9 * mean + 0.1)
    kernel = torch.ones(9, 9).cuda()
    dark = erosion(minx, kernel)
    # darkI = guidedfilter2d_gray(I, dark, 30, 1e-4)
    u, _ = torch.min(out,dim=1)
    u = erosion(u.unsqueeze(1), kernel)
    loss = F.mse_loss(u, dark)
    return loss


def reconstruction_loss(src, R, I):
    return F.l1_loss(R * I, src, reduction='mean')

# def reconstruction_loss2(output, In, R1, R2):
#     return F.l1_loss(output/(In + esp), R1, reduction='mean') + F.l1_loss(output/(In + esp), R2, reduction='mean')


def noise_loss(image, src, N):
    gray = rgb_to_grayscale(src)
    gray2 = rgb_to_grayscale(image)
    gradient_gray_h, gradient_gray_w = gradient2(gray2)
    R_h = 1/(gradient_gray_h + 1e-5) * gray
    R_w = 1/(gradient_gray_w + 1e-5) * gray
    b,c,w,h = src.shape
    return torch.norm(gray * N, 1)/(b*c*w*h) + (R_h.mean() + R_w.mean()) * 0.00001


def mutual_i_input_loss(input_I_low, input_im):
    input_gray = rgb_to_grayscale(input_im)
    max_rgb, _ = torch.max(input_im, dim=1, keepdim=True)
    low_gradient_x = gradient(input_I_low, "x")
    input_gradient_x = gradient(input_gray, "x")
    x_loss = torch.abs(torch.div(low_gradient_x, torch.clamp(input_gradient_x, 0.01)))
    low_gradient_y = gradient(input_I_low, "y")
    input_gradient_y = gradient(input_gray, "y")
    y_loss = torch.abs(torch.div(low_gradient_y, torch.clamp(input_gradient_y, 0.01)))
    mut_loss = torch.mean(x_loss + y_loss)
    return mut_loss + 0.15 * F.l1_loss(input_I_low, max_rgb)


def mutual_i_loss(input_I_low, input_I_high):
    low_gradient_x = gradient(input_I_low, "x")
    high_gradient_x = gradient(input_I_high, "x")
    x_loss = (low_gradient_x + high_gradient_x) * torch.exp(-10*(low_gradient_x+high_gradient_x))
    low_gradient_y = gradient(input_I_low, "y")
    high_gradient_y = gradient(input_I_high, "y")
    y_loss = (low_gradient_y + high_gradient_y) * torch.exp(-10*(low_gradient_y+high_gradient_y))
    mutual_loss = torch.mean(x_loss + y_loss)
    return mutual_loss

# def normalize01(img):
#     minv = img.min()
#     maxv = img.max()
#     return (img-minv)/(maxv-minv)
#
#
# def reflectance_smooth_loss(image, illumination, reflectance):
#     gray_tensor = rgb_to_grayscale(image)
#     gradient_gray_h, gradient_gray_w = gradient2(gray_tensor)
#     gradient_reflect_h, gradient_reflect_w = gradient2(reflectance)
#     weight = 1/(illumination*gradient_gray_h*gradient_gray_w+0.0001)
#     weight = normalize01(weight)
#     weight.detach()
#     loss_h = weight * gradient_reflect_h
#     loss_w = weight * gradient_reflect_w
#     refrence_reflect = image/illumination
#     refrence_reflect.detach()
#     return loss_h.mean() + loss_w.mean() + torch.norm(refrence_reflect - reflectance, 1)

def entropymax_Loss(image, im_eq):
    R_low_max, _ = torch.max(image, dim=1, keepdim=True)
    recon_loss_low_eq = F.l1_loss(R_low_max, im_eq, reduction='mean')
    return recon_loss_low_eq

def entropymax_Loss2(image, im_eq):
    R_low_max, _ = torch.min(image, dim=1, keepdim=True)
    recon_loss_low_eq = F.l1_loss(R_low_max, im_eq, reduction='mean')
    return recon_loss_low_eq

def gradient2(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:] - img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w


def illumination_smooth_loss(image, illumination):
    gray_tensor = rgb_to_grayscale(image)
    # max_rgb, _ = torch.max(image, 1, keepdim=True)
    gradient_gray_h, gradient_gray_w = gradient2(gray_tensor)
    gradient_illu_h, gradient_illu_w = gradient2(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=conf.gaussian_kernel, padding=conf.g_padding)+esp)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=conf.gaussian_kernel, padding=conf.g_padding)+esp)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    # max_rgb.detach()
    return loss_h.mean() + loss_w.mean() # + 0.1 * F.l1_loss(illumination, max_rgb, reduction='mean')



def illumination_smooth_loss2(image, illumination):
    gray_tensor = rgb_to_grayscale(image)
    min_rgb, _ = torch.min(image, 1, keepdim=True)
    gradient_gray_h, gradient_gray_w = gradient2(gray_tensor)
    gradient_illu_h, gradient_illu_w = gradient2(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=conf.gaussian_kernel, padding=conf.g_padding)+esp)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=conf.gaussian_kernel, padding=conf.g_padding)+esp)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    min_rgb.detach()
    return 0.15 * (loss_h.mean() + loss_w.mean()) + F.l1_loss(illumination, min_rgb, reduction='mean')


def normalize01(img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)


def reflectance_smooth_loss(illumination, reflectance):
    gradient_reflect_h, gradient_reflect_w = gradient2(reflectance)
    weight = 1/(illumination*gradient_reflect_h*gradient_reflect_w+esp)
    weight = normalize01(weight)
    weight.detach()
    loss_h = weight * gradient_reflect_h
    loss_w = weight * gradient_reflect_w
    return loss_h.mean() + loss_w.mean()


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, vgg_choose):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        # if vgg_choose != "no_maxpool":
        #     if opt.vgg_maxpooling:
        #         h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2)
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if vgg_choose == "conv4_3":
            return conv4_3
        elif vgg_choose == "relu4_2":
            return relu4_2
        elif vgg_choose == "relu4_1":
            return relu4_1
        elif vgg_choose == "relu4_3":
            return relu4_3
        elif vgg_choose == "conv5_3":
            return conv5_3
        elif vgg_choose == "relu5_1":
            return relu5_1
        elif vgg_choose == "relu5_2":
            return relu5_2
        elif vgg_choose == "relu5_3" or "maxpool":
            return relu5_3


def vgg_preprocess(batch):
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    return batch


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg, "relu5_1")
        target_fea = vgg(target_vgg, "relu5_1")
        # if self.opt.no_vgg_instance:
        #     return torch.mean((img_fea - target_fea) ** 2)
        # else:
        return F.mse_loss(self.instancenorm(img_fea), self.instancenorm(target_fea))


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    vgg = Vgg16()
    # vgg.cuda()
    vgg.cuda()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg



def L_color(x):
    mean_rgb = torch.mean(x,[2,3],keepdim=True)
    mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
    Drg = torch.pow(mr-mg,2)
    Drb = torch.pow(mr-mb,2)
    Dgb = torch.pow(mb-mg,2)
    k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
    return k.mean()


def hybr_loss(src, src2, I1, I2, R1, R2, T1, T2, output, output2):
    loss_r = reconstruction_loss(src, R1, I1) + reconstruction_loss(src2, R2, I2) + F.l1_loss(R1, R2)
    loss_i = ((mutual_i_input_loss(T1, src) + mutual_i_input_loss(T2, src2)) * 0.05 +
              (mutual_i_input_loss(I1, src) + mutual_i_input_loss(I2, src2)) * 0.15 + 0.2 * mutual_i_loss(I1, I2)) * 0.001
    loss_c = 0
    ###############
    loss_q = (torch.mean(torch.abs(1 - QA(output))) + torch.mean(torch.abs(1 - QA(output2)))) * 0.5
    loss_w = (L_color(output) + L_color(output2)) * 0
    ###############
    loss_total = loss_r + loss_i + loss_q
    k = [loss_i.item(), loss_c, loss_r.item(), loss_w.item(), loss_q.item()]
    return loss_total, k


