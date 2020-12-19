import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import torch.autograd  
import os

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

class DepthCompletionNet(nn.Module):
    def __init__(self, args):
        assert (args.layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(DepthCompletionNet, self).__init__()
        self.modality = args.input

        # pre-trained DI2DI
        checkpoint = None
        if os.path.isfile(args.DI2DI):
            print("load DI2DI pre-train model", args.DI2DI)
            DI_model = realDICompletionNet(args).cuda()
            checkpoint = torch.load(args.DI2DI)
            DI_model.load_state_dict(checkpoint['model'])
            self.conv1_d = DI_model._modules['conv1_d']
            self.conv1_i = DI_model._modules['conv1_i']
            self.conv2 = DI_model._modules['conv2']
            self.conv3 = DI_model._modules['conv3']
            self.conv4 = DI_model._modules['conv4']
            self.conv5 = DI_model._modules['conv5']
            self.conv6 = DI_model._modules['conv6']
            self.conv2I = DI_model._modules['conv2I']
            self.conv3I = DI_model._modules['conv3I']
            self.conv4I = DI_model._modules['conv4I']
            self.conv5I = DI_model._modules['conv5I']
            self.encodef = DI_model._modules['encodef']
            self.convt5 = DI_model._modules['convt5']
            self.convt4 = DI_model._modules['convt4']
            self.convt3 = DI_model._modules['convt3']
            self.convt2 = DI_model._modules['convt2']
            self.convt1 = DI_model._modules['convt1']
            self.convtf = DI_model._modules['convtf']
            print("load done")
            del DI_model
        else:
            print("use new model")
            if 'd' in self.modality:
                #TODO
                channels = 64  // len(self.modality) #64 if d,  16 if rgbd, 32 if gd
                # to check
                self.conv1_d = conv_bn_relu(2, channels, kernel_size=3, stride=1, padding=1)
                self.conv1_i = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
                # self.conv1_d = conv_bn_relu(2, channels, kernel_size=3, stride=1, padding=1)
                # self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
            if 'rgb' in self.modality:
                channels = 64 * 3 // len(self.modality) #64 if rgb, 48 if rgbd
                self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
            elif 'g' in self.modality:
                channels = 64 // len(self.modality) # 32 if gd
                self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

            pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
            if not args.pretrained:
                pretrained_model.apply(init_weights)
            #self.maxpool = pretrained_model._modules['maxpool']
            self.conv2 = pretrained_model._modules['layer1']
            self.conv3 = pretrained_model._modules['layer2']
            self.conv4 = pretrained_model._modules['layer3']
            self.conv5 = pretrained_model._modules['layer4']
            del pretrained_model # clear memory

            pretrained_modelI = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
            if not args.pretrained:
                pretrained_modelI.apply(init_weights)
            self.conv2I = pretrained_modelI._modules['layer1']
            self.conv3I = pretrained_modelI._modules['layer2']
            self.conv4I = pretrained_modelI._modules['layer3']
            self.conv5I = pretrained_modelI._modules['layer4']
            del pretrained_modelI # clear memory

            # define number of intermediate channels
            if args.layers <= 34:
                num_channels = 512
            elif args.layers >= 50:
                num_channels = 2048
            self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)
            self.encodef = nn.Sequential(
                conv_bn_relu(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
            )
            # self.conv6I = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

            # decoding layers
            kernel_size = 3
            stride = 2

            self.convt5 = convt_bn_relu(in_channels=512, out_channels=256,
                kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
            self.convt4 = convt_bn_relu(in_channels=(512+256), out_channels=128,
                kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
            self.convt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
                kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
            self.convt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
                kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
            self.convt1 = convt_bn_relu(in_channels=(64+64), out_channels=64,
                kernel_size=kernel_size, stride=1, padding=1)
            # self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)
            self.convtf = conv_bn_relu(in_channels=(64+64), out_channels=2, kernel_size=1, stride=1, bn=False, relu=False)

        # inverse normalization model
        self.conv_normal1 = conv_bn_relu(1, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
        self.conv_normal2 = conv_bn_relu(1, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
        self.conv1_norm = conv_bn_relu(3, 16, kernel_size=3, stride=1, padding=1)
        # self.conv1_norm = conv_bn_relu(2, 16, kernel_size=3, stride=1, padding=1)
        # self.conv1_norm = conv_bn_relu(2, 16, kernel_size=5, stride=1, padding=2)
        self.conv2_norm = conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1)
        self.convfinal_norm = conv_bn_relu(in_channels=32, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, x):
        # first layer
        conv0_merge = torch.cat((x['d'], x['intensity']), 1)
        conv1 = self.conv1_d(conv0_merge)   # input 2
        conv1I = self.conv1_i(x['intensity'])

        conv1_sum = torch.add(conv1, conv1I)
        conv2 = self.conv2(conv1_sum)
        conv2I = self.conv2I(conv1I)

        conv2_sum = torch.add(conv2, conv2I)
        conv3 = self.conv3(conv2_sum) # batchsize * ? * 176 * 608
        conv3I = self.conv3I(conv2I)

        conv3_sum = torch.add(conv3, conv3I)
        conv4 = self.conv4(conv3_sum) # batchsize * ? * 176 * 608
        conv4I = self.conv4I(conv3I)

        conv4_sum = torch.add(conv4, conv4I)
        conv5 = self.conv5(conv4_sum) # batchsize * ? * 176 * 608
        conv5I = self.conv5I(conv4I)

        conv5_sum = torch.add(conv5, conv5I)
        conv6 = self.conv6(conv5_sum) # batchsize * ? * 176 * 608
        # conv6I = self.conv6I(conv5I)

        # conv_merge = torch.cat((conv6, conv6I), 1)
        # conv_sum = torch.add(conv6, conv6I)
        conv_enfinal = self.encodef(conv6)

        # decoder
        # convt5 = self.convt5(conv_merge)    #1024->256
        convt5 = self.convt5(conv_enfinal)
        y = torch.cat((convt5, conv5_sum), 1)   # 256+512+512

        convt4 = self.convt4(y) 
        y = torch.cat((convt4, conv4_sum), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3_sum), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2_sum), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1_sum), 1)

        y = self.convtf(y)  # without BN, whthout relu
        y = 100 * y

        y_normal = self.conv_normal1( ((y[:, 0, :, :]).unsqueeze(1)) )
        y_normal = self.conv_normal2(y_normal)

        # y_norm = self.conv1_norm(y)
        y_norm = self.conv1_norm(torch.cat((y_normal, y), 1))
        y_norm = self.conv2_norm(y_norm)
        y_norm = self.convfinal_norm(y_norm)

        if self.training:
            return y, y_norm
            # return 100 * y[:,0, :, :]
        else:
            min_distance = 0.9
            y = F.relu(y - min_distance) + min_distance
            return y, F.relu(y_norm)
            # return F.relu(100 * y - min_distance) + min_distance # the minimum range of Velodyne is around 3 feet ~= 0.9m


class realDICompletionNet(nn.Module):
    def __init__(self, args):
        assert (args.layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(realDICompletionNet, self).__init__()
        self.modality = args.input

        if 'd' in self.modality:
            #TODO
            channels = 64  // len(self.modality) #64 if d,  16 if rgbd, 32 if gd
            # to check
            self.conv1_d = conv_bn_relu(2, channels, kernel_size=3, stride=1, padding=1)
            self.conv1_i = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
            # self.conv1_d = conv_bn_relu(2, channels, kernel_size=3, stride=1, padding=1)
            # self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality) #64 if rgb, 48 if rgbd
            self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality) # 32 if gd
            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model # clear memory

        pretrained_modelI = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_modelI.apply(init_weights)
        self.conv2I = pretrained_modelI._modules['layer1']
        self.conv3I = pretrained_modelI._modules['layer2']
        self.conv4I = pretrained_modelI._modules['layer3']
        self.conv5I = pretrained_modelI._modules['layer4']
        del pretrained_modelI # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)
        self.encodef = nn.Sequential(
            conv_bn_relu(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        )
        # self.conv6I = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2

        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=(512+256), out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=(64+64), out_channels=64,
            kernel_size=kernel_size, stride=1, padding=1)
        # self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)
        self.convtf = conv_bn_relu(in_channels=(64+64), out_channels=2, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, x):
        # first layer
        conv0_merge = torch.cat((x['d'], x['intensity']), 1)
        conv1 = self.conv1_d(conv0_merge)   # input 2
        conv1I = self.conv1_i(x['intensity'])

        conv1_sum = torch.add(conv1, conv1I)
        conv2 = self.conv2(conv1_sum)
        conv2I = self.conv2I(conv1I)

        conv2_sum = torch.add(conv2, conv2I)
        conv3 = self.conv3(conv2_sum) # batchsize * ? * 176 * 608
        conv3I = self.conv3I(conv2I)

        conv3_sum = torch.add(conv3, conv3I)
        conv4 = self.conv4(conv3_sum) # batchsize * ? * 176 * 608
        conv4I = self.conv4I(conv3I)

        conv4_sum = torch.add(conv4, conv4I)
        conv5 = self.conv5(conv4_sum) # batchsize * ? * 176 * 608
        conv5I = self.conv5I(conv4I)

        conv5_sum = torch.add(conv5, conv5I)
        conv6 = self.conv6(conv5_sum) # batchsize * ? * 176 * 608
        # conv6I = self.conv6I(conv5I)

        # conv_merge = torch.cat((conv6, conv6I), 1)
        # conv_sum = torch.add(conv6, conv6I)
        conv_enfinal = self.encodef(conv6)

        # decoder
        # convt5 = self.convt5(conv_merge)    #1024->256
        convt5 = self.convt5(conv_enfinal)
        y = torch.cat((convt5, conv5_sum), 1)   # 256+512+512

        convt4 = self.convt4(y) 
        y = torch.cat((convt4, conv4_sum), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3_sum), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2_sum), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1_sum), 1)

        y = self.convtf(y)  # without BN, whthout relu

        if self.training:
            # y_out_depth = 100 * y[:, 0, :, :]
            # y_out_intensity = 100 * y[:, 1, :, :]
            # y_out_normal = y[:, 2:5, :, :]
            # y_out = torch.cat((y_out_depth.unsqueeze(1), y_out_intensity.unsqueeze(1), y_out_normal), 1)
            # return y_out
            y = 100* y
            return y
            # return 100 * y[:,0, :, :]
        else:
            min_distance = 0.9
            return F.relu(100 * y - min_distance) + min_distance # the minimum range of Velodyne is around 3 feet ~= 0.9m
            # y_out_depth = F.relu(100 * y[:, 0, :, :] - min_distance) + min_distance # the minimum range of Velodyne is around 3 feet ~= 0.9m
            # y_out_intensity = F.relu(100 * y[:, 1, :, :])
            # y_out_normal = y[:, 2:5, :, :]
            # y_out = torch.cat((y_out_depth.unsqueeze(1), y_out_intensity.unsqueeze(1), y_out_normal), 1)
            # return y_out
