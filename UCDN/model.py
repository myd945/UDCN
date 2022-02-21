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
import filter


def rgb_to_grayscale(s):
    return (0.2989*s[:, 0, :, :]+ 0.5870*s[:, 1, :, :] + 0.1140*s[:, 2, :, :]).unsqueeze(1)

class light_conv(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, kernel_size=3, stride=1, padding=1, padding_mode='reflect'):
        super(light_conv, self).__init__()
        self.S = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                                         padding_mode='reflect'),
                               # nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                               )
    def forward(self, x):
        return self.S(x)

class DecomNet(nn.Module):

    def __init__(self, channel=64, kernel_size=3, is_Training=True):
        super(DecomNet, self).__init__()

        self.conv0 = nn.Conv2d(4, int(channel / 2), kernel_size, padding=1)
        self.conv = nn.Conv2d(4, channel, kernel_size * 3, padding=4)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(channel, channel * 2, kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(channel * 2, channel * 2, kernel_size, padding=1)
        self.conv4 = nn.ConvTranspose2d(channel * 2, channel, kernel_size, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1)
        self.conv6 = nn.Conv2d(3 * int(channel / 2), channel, kernel_size, padding=1)
        self.conv7 = nn.Conv2d(channel, 1, kernel_size, padding=1)
        self.upsample = F.interpolate
        # self.S1 = nn.Sequential(light_conv(4, 32, 3, 1, 1),
        #                        # nn.ReLU(inplace=True),
        #                        )
        # self.B = nn.Sequential(light_conv(32, 64, 3, 1, 1),
        #                        # nn.ReLU(inplace=True),
        #                        )
        # self.S2 = nn.Sequential(light_conv(32, 64, 3, 1, 1),
        #                         # nn.ReLU(inplace=True),
        #                         light_conv(64, 128, 3, 1, 1),
        #                         # nn.ReLU(inplace=True),
        #                        )
        # self.M1 = nn.MaxPool2d(2)
        # self.A1 = nn.AvgPool2d(2)
        # self.S3 = nn.Sequential(light_conv(256, 128, 3, 1, 1),
        #                         # nn.ReLU(inplace=True),
        #                         light_conv(128, 64, 3, 1, 1),
        #                         # nn.ReLU(inplace=True),
        #                         # nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
        #                         # nn.ReLU(inplace=True),
        #                         )
        # self.S4 = nn.Sequential(light_conv(128, 64, 3, 1, 1),
        #                         # nn.ReLU(inplace=True),
        #                         light_conv(64, 32, 3, 1, 1),
        #                         # nn.ReLU(inplace=True),
        #                         light_conv(32, 4, 3, 1, 1),
        #                         )

    def forward(self, x):
        x_hist = torch.max(x, dim=1, keepdim=True)
        # x_hist = x_hist.float()
        x = torch.cat((x, x_hist[0]), dim=1)

        x1 = F.relu(self.conv0(x))
        x = self.conv(x)
        x = F.relu(self.conv1(x))
        y = x
        shp = y.data.shape
        shp = shp[2:4]
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.upsample(x, size=shp, mode='nearest')
        x = F.relu(self.conv4_1(x))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.conv5(x))

        x = torch.cat((x, x1), dim=1)
        x = self.conv6(x)
        out = self.conv7(x)

        # x1 = self.S1(x)
        # B = self.B(x1)
        # x2 = self.S2(x1)
        # x3 = F.interpolate(torch.cat([self.M1(x2), self.A1(x2)], dim=1), scale_factor=2, mode='nearest')
        # x4 = self.S3(x3)
        # # print(B.shape, x4.shape)
        # out = self.S4(torch.cat([x4, B], dim=1))

        return out


class LightNet(nn.Module):

    def __init__(self):
        super(LightNet, self).__init__()

        self.conv1 = nn.Sequential(light_conv(3, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.MaxPool2d(2),
                                   light_conv(32, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.MaxPool2d(2),
                                   light_conv(64, 128, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(light_conv(128, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                                   )
        self.conv5 = nn.Sequential(light_conv(128, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(light_conv(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.conv7 = nn.Sequential(light_conv(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   )
        self.conv8 = nn.Sequential(light_conv(32, 3, 3, 1, 1),
                                   nn.Sigmoid())

        self.convT1 = nn.Sequential(light_conv(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True))
        self.convT2 = nn.Sequential(light_conv(64, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True))
        self.convT3 = nn.Sequential(light_conv(32, 1, 3, 1, 1),
                                    nn.Sigmoid())



    def forward(self, x):
        x1 = self.conv1(x)
        t1 = self.convT1(x1)
        x2 = self.conv2(x1)
        x = self.conv3(x2)
        x = self.conv4(x)
        x = self.conv5(torch.cat([x2, x], dim=1))
        x = self.conv6(x)
        x = self.conv7(torch.cat([x1, x], dim=1))
        R = self.conv8(x)
        t2 = self.convT2(torch.cat([t1, x], dim=1))
        I = self.convT3(t2)
        return R, I

class FFI(nn.Module):

    def __init__(self, s):
        super(FFI, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 1, 1),
                                   nn.Sigmoid()
                                   )
        self.size = s

    def forward(self, I, x):
        b, c, w, h = I.shape
        A = F.interpolate(I, size=(w//self.size, h//self.size))
        I = F.interpolate(x, size=(w//self.size, h//self.size))
        B = self.conv1(A)
        C = torch.pow(I, B)
        T = F.interpolate(C, scale_factor=self.size)
        return T


class IP(nn.Module):

    def __init__(self):
        super(IP, self).__init__()
        self.layer1 = FFI(1)
        self.layer2 = FFI(2)
        self.layer3 = FFI(4)

    def forward(self, I, x):
        out = self.layer1(I, x)
        out += self.layer2(I, x)
        out += self.layer3(I, x)
        return out/3


class FuNet(nn.Module):

    def __init__(self):
        super(FuNet, self).__init__()
        self.de1 = DecomNet()
        # self.tune1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 1, 1),
        #                            nn.Sigmoid()
        #                            )
        # self.N = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 32, 3, 1, 1),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(32, 3, 1),
        #                            nn.Tanh()
        #                            )
        self.tune1 = IP()


    def forward(self, x):
        out1 = self.de1(x)
        # R1 = torch.sigmoid(out1[:, 0:3, :, :])
        # I1 = torch.sigmoid(out1[:, 3:4, :, :])
        I1 = torch.sigmoid(out1)
        R1 = x/torch.clamp(out1, 0.1)
        # R1, I1 = self.de1(x)
        T = self.tune1(I1, rgb_to_grayscale(x))
        # R1 = R1
        output = T * R1
        return I1, R1, T, output


def contrast(input, kernal_size):
    gray = rgb_to_grayscale(input)
    K = filter.get_laplacian_kernel2d(kernal_size)
    return torch.abs(filter.filter2D(gray, K.unsqueeze(0)))

def Saturation(input):
    return torch.std(input, dim=1).unsqueeze(1)


def Well_exposedness(input):
    input = torch.exp(-(torch.pow((input-0.5), 2)/0.4))
    W = input[:,0,:,:]*input[:,1,:,:]*input[:,2,:,:]
    return W.unsqueeze(1)

def QA(input):
    C = contrast(input, 9)
    S = Saturation(input)
    W = Well_exposedness(input)
    return C*S*W