import os,sys
import itertools
import numpy as np
import math
import random
import copy
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.nn import functional as F

import fdgan.utils.util as util
from fdgan.networks import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, \
                            remove_module_key, set_bn_fix, get_scheduler, print_network
from fdgan.losses import GANLoss
from reid.models import create
from reid.models.embedding import EltwiseSubEmbed
from reid.models.multi_branch import SiameseNet

class FDGANModel(object):

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints, opt.name)
        self.norm_layer = get_norm_layer(norm_type=opt.norm)

        self._init_models()
        self._init_losses()
        self._init_optimizers()

        print('---------- Networks initialized -------------')
        print_network(self.net_E)
        print_network(self.net_G)
        print_network(self.net_Di)
        print_network(self.net_Dp)
        print('-----------------------------------------------')

    def _init_models(self):
        self.net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, self.opt.noise_feature_size,
                                dropout=self.opt.drop, norm_layer=self.norm_layer, fuse_mode=self.opt.fuse_mode, connect_layers=self.opt.connect_layers)
        e_base_model = create(self.opt.arch, cut_at_pooling=True)
        e_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=2)
        self.net_E = SiameseNet(e_base_model, e_embed_model)

        di_base_model = create(self.opt.arch, cut_at_pooling=True)
        di_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=1)
        self.net_Di = SiameseNet(di_base_model, di_embed_model)
        self.net_Dp = NLayerDiscriminator(3+18, norm_layer=self.norm_layer)

        if self.opt.stage==1:
            init_weights(self.net_G)
            init_weights(self.net_Dp)
            state_dict = remove_module_key(torch.load(self.opt.netE_pretrain))
            self.net_E.load_state_dict(state_dict)
            state_dict['embed_model.classifier.weight'] = state_dict['embed_model.classifier.weight'][1]
            state_dict['embed_model.classifier.bias'] = torch.FloatTensor([state_dict['embed_model.classifier.bias'][1]])
            self.net_Di.load_state_dict(state_dict)
        elif self.opt.stage==2:
            self._load_state_dict(self.net_E, self.opt.netE_pretrain)
            self._load_state_dict(self.net_G, self.opt.netG_pretrain)
            self._load_state_dict(self.net_Di, self.opt.netDi_pretrain)
            self._load_state_dict(self.net_Dp, self.opt.netDp_pretrain)
        else:
            assert('unknown training stage')

        self.net_E = torch.nn.DataParallel(self.net_E).cuda()
        self.net_G = torch.nn.DataParallel(self.net_G).cuda()
        self.net_Di = torch.nn.DataParallel(self.net_Di).cuda()
        self.net_Dp = torch.nn.DataParallel(self.net_Dp).cuda()

    def reset_model_status(self):
        if self.opt.stage==1:
            self.net_G.train()
            self.net_Dp.train()
            self.net_E.eval()
            self.net_Di.train()
            self.net_Di.apply(set_bn_fix)
        elif self.opt.stage==2:
            self.net_E.train()
            self.net_G.train()
            self.net_Di.train()
            self.net_Dp.train()
            self.net_E.apply(set_bn_fix)
            self.net_Di.apply(set_bn_fix)

    def _load_state_dict(self, net, path):
        state_dict = remove_module_key(torch.load(path))
        net.load_state_dict(state_dict)

    def _init_losses(self):
        if self.opt.smooth_label:
            self.criterionGAN_D = GANLoss(smooth=True).cuda()
            self.rand_list = [True] * 1 + [False] * 10000
        else:
            self.criterionGAN_D = GANLoss(smooth=False).cuda()
            self.rand_list = [False]
        self.criterionGAN_G = GANLoss(smooth=False).cuda()

    def _init_optimizers(self):
        if self.opt.stage==1:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.opt.lr*0.01, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opt.stage==2:
            param_groups = [{'params': self.net_E.module.base_model.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_E.module.embed_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.net_G.parameters(), 'lr_mult': 0.1}]
            self.optimizer_G = torch.optim.Adam(param_groups,
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)

        self.schedulers = []
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_Di)
        self.optimizers.append(self.optimizer_Dp)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.opt))

    def set_input(self, input):
        input1, input2 = input
        labels = (input1['pid']==input2['pid']).long()
        noise = torch.randn(labels.size(0), self.opt.noise_feature_size)

        # keep the same pose map for persons with the same identity
        mask = labels.view(-1,1,1,1).expand_as(input1['posemap'])
        input2['posemap'] = input1['posemap']*mask.float() + input2['posemap']*(1-mask.float())
        mask = labels.view(-1,1,1,1).expand_as(input1['target'])
        input2['target'] = input1['target']*mask.float() + input2['target']*(1-mask.float())

        origin = torch.cat([input1['origin'], input2['origin']])
        target = torch.cat([input1['target'], input2['target']])
        posemap = torch.cat([input1['posemap'], input2['posemap']])
        noise = torch.cat((noise, noise))

        self.origin = origin.cuda()
        self.target = target.cuda()
        self.posemap = posemap.cuda()
        self.labels = labels.cuda()
        self.noise = noise.cuda()

    def forward(self):
        A = Variable(self.origin)
        B_map = Variable(self.posemap)
        z = Variable(self.noise)
        bs = A.size(0)

        A_id1, A_id2, self.id_score = self.net_E(A[:bs//2], A[bs//2:])
        A_id = torch.cat((A_id1, A_id2))
        self.fake = self.net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))

    def backward_Dp(self):
        real_pose = torch.cat((Variable(self.posemap), Variable(self.target)),dim=1)
        fake_pose = torch.cat((Variable(self.posemap), self.fake.detach()),dim=1)
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(fake_pose)

        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.loss_Dp = loss_D.data[0]

    def backward_Di(self):
        _, _, pred_real = self.net_Di(Variable(self.origin), Variable(self.target))
        _, _, pred_fake = self.net_Di(Variable(self.origin), self.fake.detach())
        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.loss_Di = loss_D.data[0]

    def backward_G(self):
        loss_v = F.cross_entropy(self.id_score, Variable(self.labels).view(-1))
        loss_r = F.l1_loss(self.fake, Variable(self.target))
        fake_1 = self.fake[:self.fake.size(0)//2]
        fake_2 = self.fake[self.fake.size(0)//2:]
        loss_sp = F.l1_loss(fake_1[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1], 
                            fake_2[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1])

        _, _, pred_fake_Di = self.net_Di(Variable(self.origin), self.fake)
        pred_fake_Dp = self.net_Dp(torch.cat((Variable(self.posemap),self.fake),dim=1))
        loss_G_GAN_Di = self.criterionGAN_G(pred_fake_Di, True)
        loss_G_GAN_Dp = self.criterionGAN_G(pred_fake_Dp, True)

        loss_G = loss_G_GAN_Di + loss_G_GAN_Dp + \
                loss_r * self.opt.lambda_recon + \
                loss_v * self.opt.lambda_veri + \
                loss_sp * self.opt.lambda_sp
        loss_G.backward()

        del self.id_score
        self.loss_G = loss_G.data[0]
        self.loss_v = loss_v.data[0]
        self.loss_sp = loss_sp.data[0]
        self.loss_r = loss_r.data[0]
        self.loss_G_GAN_Di = loss_G_GAN_Di.data[0]
        self.loss_G_GAN_Dp = loss_G_GAN_Dp.data[0]
        self.fake = self.fake.data

    def optimize_parameters(self):
        self.forward()

        self.optimizer_Di.zero_grad()
        self.backward_Di()
        self.optimizer_Di.step()

        self.optimizer_Dp.zero_grad()
        self.backward_Dp()
        self.optimizer_Dp.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_v', self.loss_v),
                            ('G_r', self.loss_r),
                            ('G_sp', self.loss_sp),
                            ('G_gan_Di', self.loss_G_GAN_Di),
                            ('G_gan_Dp', self.loss_G_GAN_Dp),
                            ('D_i', self.loss_Di),
                            ('D_p', self.loss_Dp)
                            ])

    def get_current_visuals(self):
        input = util.tensor2im(self.origin)
        target = util.tensor2im(self.target)
        fake = util.tensor2im(self.fake)
        map = self.posemap.sum(1)
        map[map>1]=1
        map = util.tensor2im(torch.unsqueeze(map,1))
        return OrderedDict([('input', input), ('posemap', map), ('fake', fake), ('target', target)])

    def save(self, epoch):
        self.save_network(self.net_E, 'E', epoch)
        self.save_network(self.net_G, 'G', epoch)
        self.save_network(self.net_Di, 'Di', epoch)
        self.save_network(self.net_Dp, 'Dp', epoch)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']