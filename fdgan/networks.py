from __future__ import absolute_import
import os, sys
import functools

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler
import torchvision

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net):
    net.apply(weights_init_normal)

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 2 - opt.niter) / float(opt.niter_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def remove_module_key(state_dict):
    for key in list(state_dict.keys()):
        if 'module' in key:
            state_dict[key.replace('module.','')] = state_dict.pop(key)
    return state_dict

def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class CustomPoseGenerator(nn.Module):
    def __init__(self, pose_feature_nc, reid_feature_nc, noise_nc, pose_nc=18, output_nc=3, 
                        dropout=0.0, norm_layer=nn.BatchNorm2d, fuse_mode='cat', connect_layers=0):
        super(CustomPoseGenerator, self).__init__()
        assert (connect_layers>=0 and connect_layers<=5)
        ngf = 64
        self.connect_layers = connect_layers
        self.fuse_mode = fuse_mode
        self.norm_layer = norm_layer
        self.dropout = dropout

        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d

        input_channel = [[8, 8, 4, 2, 1],
                        [16, 8, 4, 2, 1],
                        [16, 16, 4, 2, 1],
                        [16, 16, 8, 2, 1],
                        [16, 16, 8, 4, 1],
                        [16, 16, 8, 4, 2]]

        ##################### Encoder #########################
        self.en_conv1 = nn.Conv2d(pose_nc, ngf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        # N*64*128*64
        self.en_conv2 = self._make_layer_encode(ngf, ngf*2)
        # N*128*64*32
        self.en_conv3 = self._make_layer_encode(ngf*2, ngf*4)
        # N*256*32*16
        self.en_conv4 = self._make_layer_encode(ngf*4, ngf*8)
        # N*512*16*8
        self.en_conv5 = self._make_layer_encode(ngf*8, ngf*8)
        # N*512*8*4
        en_avg = [nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf * 8, pose_feature_nc,
                    kernel_size=(8,4), bias=self.use_bias),
                norm_layer(pose_feature_nc)]
        self.en_avg = nn.Sequential(*en_avg)
        # N*512*1*1

        ##################### Decoder #########################
        if fuse_mode=='cat':
            de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(pose_feature_nc+reid_feature_nc+noise_nc, ngf * 8,
                        kernel_size=(8,4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        elif fuse_mode=='add':
            nc = max(pose_feature_nc, reid_feature_nc, noise_nc)
            self.W_pose = nn.Linear(pose_feature_nc, nc, bias=False)
            self.W_reid = nn.Linear(reid_feature_nc, nc, bias=False)
            self.W_noise = nn.Linear(noise_nc, nc, bias=False)
            de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(nc, ngf * 8,
                        kernel_size=(8,4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        else:
            raise ('Wrong fuse mode, please select from [cat|add]')
        self.de_avg = nn.Sequential(*de_avg)
        # N*512*8*4

        self.de_conv5 = self._make_layer_decode(ngf * input_channel[connect_layers][0],ngf * 8)
        # N*512*16*8
        self.de_conv4 = self._make_layer_decode(ngf * input_channel[connect_layers][1],ngf * 4)
        # N*256*32*16
        self.de_conv3 = self._make_layer_decode(ngf * input_channel[connect_layers][2],ngf * 2)
        # N*128*64*32
        self.de_conv2 = self._make_layer_decode(ngf * input_channel[connect_layers][3],ngf)
        # N*64*128*64
        de_conv1 = [nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * input_channel[connect_layers][4],output_nc,
                        kernel_size=4, stride=2,
                        padding=1, bias=self.use_bias),
                    nn.Tanh()]
        self.de_conv1 = nn.Sequential(*de_conv1)
        # N*3*256*128

    def _make_layer_encode(self, in_nc, out_nc):
        block = [nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, out_nc,
                        kernel_size=4, stride=2,
                        padding=1, bias=self.use_bias),
                self.norm_layer(out_nc)]
        return nn.Sequential(*block)

    def _make_layer_decode(self, in_nc, out_nc):
        block = [nn.ReLU(True),
                nn.ConvTranspose2d(in_nc, out_nc,
                    kernel_size=4, stride=2,
                    padding=1, bias=self.use_bias),
                self.norm_layer(out_nc),
                nn.Dropout(self.dropout)]
        return nn.Sequential(*block)

    def decode(self, model, fake_feature, pose_feature, cnlayers):
        if cnlayers>0:
            return model(torch.cat((fake_feature,pose_feature),dim=1)), cnlayers-1
        else:
            return model(fake_feature), cnlayers

    def forward(self, posemap, reid_feature, noise):
        batch_size = posemap.data.size(0)

        pose_feature_1 = self.en_conv1(posemap)
        pose_feature_2 = self.en_conv2(pose_feature_1)
        pose_feature_3 = self.en_conv3(pose_feature_2)
        pose_feature_4 = self.en_conv4(pose_feature_3)
        pose_feature_5 = self.en_conv5(pose_feature_4)
        pose_feature = self.en_avg(pose_feature_5)

        if self.fuse_mode=='cat':
            feature = torch.cat((reid_feature, pose_feature, noise),dim=1)
        elif self.fuse_mode=='add':
            feature = self.W_reid(reid_feature.view(batch_size, -1)) + \
                    self.W_pose(pose_feature.view(batch_size, -1)) + \
                    self.W_noise(noise.view(batch_size,-1))
            feature = feature.view(batch_size,-1,1,1)

        fake_feature = self.de_avg(feature)

        cnlayers = self.connect_layers
        fake_feature_5, cnlayers = self.decode(self.de_conv5, fake_feature, pose_feature_5, cnlayers)
        fake_feature_4, cnlayers = self.decode(self.de_conv4, fake_feature_5, pose_feature_4, cnlayers)
        fake_feature_3, cnlayers = self.decode(self.de_conv3, fake_feature_4, pose_feature_3, cnlayers)
        fake_feature_2, cnlayers = self.decode(self.de_conv2, fake_feature_3, pose_feature_2, cnlayers)
        fake_feature_1, cnlayers = self.decode(self.de_conv1, fake_feature_2, pose_feature_1, cnlayers)

        fake_imgs = fake_feature_1
        return fake_imgs

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        ndf = 64
        n_layers = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
