import torch
import os
import torch.nn as nn
from Models.layers import *


def mobileNet_V2(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        # if 'transform_input' not in kwargs:
        #     kwargs['transform_input'] = True
        model = MobileNetV2(**kwargs)
        state_dict = torch.load('/home/hdc/stomach/ckpt/mobileNet.pth.tar')
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "classifier" not in k}
        model.load_state_dict(state_dict, False)
        return model

    return MobileNetV2(**kwargs)


class MobileNetV2(nn.Module):
    def __init__(self,num_classes=2,downsampling=256,num_channels=3,kernel_size=3,img_width=512,img_height=512,dropout_prob=0.0,width_multiplier=1):
        super(MobileNetV2,self).__init__()

        s1,s2=2,2
        if downsampling==16:
            s1,s2=2,1
        elif downsampling==8:
            s1,s2=1,1
        '''
        network_settings中:
        't'表示Inverted Residuals的扩征系数
        'c'表示该block输出的通道数
        ‘n’表示当前block由几个残差单元组成
        's'表示当前block的stride
        '''
        self.network_settings=[{'t':-1,'c':32,'n':1,'s':s1},# Expected input batch_size (1600) to match target batch_size (64)
                               {'t': 1, 'c': 16, 'n': 8, 's': 1},# Expected input batch_size (256) to match target batch_size (64).
                               {'t': 6, 'c': 24, 'n': 2, 's': s2},
                               {'t': 6, 'c': 32, 'n': 3, 's': 2},
                               {'t': 6, 'c': 64, 'n': 4, 's': 2},
                               {'t': 6, 'c': 96, 'n': 24, 's': 1},
                               {'t': 6, 'c': 160, 'n': 3, 's': 2},
                               {'t': 6, 'c': 320, 'n': 24, 's': 1},
                               {'t': 6, 'c': 320, 'n': 2, 's': 2},
                               {'t': None, 'c': 1280, 'n': 1, 's': 1}]
        self.num_classes=num_classes
        # width_multiplier网络的通道"瘦身"系数
        self.network=[conv2d_bn_relu6(num_channels,int(self.network_settings[0]['c']*width_multiplier),kernel_size,
                                      self.network_settings[0]['s'],dropout_prob)]
        for i in range(1,len(self.network_settings)-1):
            self.network.extend(
                inverted_residual_sequence(
                    int(self.network_settings[i-1]['c']*width_multiplier),
                    int(self.network_settings[i]['c']*width_multiplier),
                    self.network_settings[i]['n'],self.network_settings[i]['t'],
                    kernel_size,self.network_settings[i]['s']
                )
            )
        self.network.append(
            conv2d_bn_relu6(int(self.network_settings[7]['c']*width_multiplier),
                            int(self.network_settings[8]['c']*width_multiplier),
                            1,self.network_settings[8]['s'],dropout_prob)
        )
        self.network.append(nn.Dropout2d(dropout_prob,inplace=False))
        self.network.append(nn.AvgPool2d((img_height//downsampling,img_width//downsampling)))
        # self.network.append(nn.AvgPool2d((2,2)))
        self.network.append(nn.Dropout2d(dropout_prob,inplace=False))
        self.network.append(nn.Conv2d(int(self.network_settings[8]['c']*width_multiplier),self.num_classes,1,bias=True))
        self.network=nn.Sequential(*self.network)
        self.initialize()


    def forward(self,x):
        x=self.network(x)
        x=x.view(-1,self.num_classes)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)