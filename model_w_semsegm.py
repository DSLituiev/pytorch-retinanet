import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors

from unet import DownConv, conv1x1, conv3x3, upconv2x2
from coordconv import CoordConv

from lib.nms.pth_nms import pth_nms

def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    return pth_nms(dets, thresh)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def subsample_features(x, pyramid_levels):
    pyramid_features = []
    for ii in range(max(pyramid_levels)+1):
        #print(ii, (ii in pyramid_levels), pyramid_levels)
        if (ii in pyramid_levels):
            pyramid_features.append(x)
        x = nn.MaxPool2d(2)(x)
    return pyramid_features


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = CoordConv(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = CoordConv(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = CoordConv(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = CoordConv(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        #self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        #print("shape", out.shape)
        # out is B x C x W x H, with C = 4*num_anchors
        batch_size, channels, width, height = out.shape
        #out = out.permute(0, 2, 3, 1)
        #batch_size, width, height, channels = out.shape
        #out2 = out.contiguous().view(batch_size, self.num_anchors, 4, width, height,)
        out2 = out.contiguous().view(batch_size, 4, self.num_anchors, width, height,)

        return out2.contiguous()#.view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)
        batch_size, channels, width, height = out.shape

        # out is B x C x W x H, with C = n_classes + n_anchors
        #out1 = out.permute(0, 2, 3, 1)
        #batch_size, width, height, channels = out1.shape

        out2 = out.contiguous().view(batch_size, self.num_classes, self.num_anchors, width, height,)

        return out2.contiguous()#.view(x.shape[0], -1, self.num_classes)

#############################################################
from warnings import warn
class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, input):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        if isinstance(input, (list, tuple)):
            from_up, from_down = input
        else:
            from_up, from_down = input, None

        from_up = self.upconv(from_up)
        if from_down is not None:
            if self.merge_mode == 'concat':
                x = torch.cat((from_up, from_down), 1)
            else:
                x = from_up + from_down
        else:
            warn("no from_down branch")
            x = from_up
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
#############################################################
def UpsampleBlock(in_channels = 256, out_channels=None, steps=3):
    in_channels_ = []
    out_channels_ = []
    for ii in range(steps):
        in_ = in_channels//(2**ii)
        out_ = int(in_channels//(2**(ii+1)))
        in_channels_.append(in_)
        out_channels_.append(out_)
    if out_channels is not None:
        out_channels_[-1] = out_channels
        
    uc = []
    for ii in range(steps):
#         print(in_, out_)
        uc.append(UpConv(in_channels_[ii], out_channels_[ii], merge_mode=''))  
    return torch.nn.Sequential(*uc)
#############################################################
class TemplateUnet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
            
            
class UNetEncode(TemplateUnet):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, in_channels=3, depth=5, hid_channels = None,
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNetEncode, self).__init__()
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []

        # create the encoder pathway and add to a list
        if hid_channels is None:
            ins = [self.start_filts*(2**(i-1)) for i  in range(depth)]
            ins[0] = self.in_channels
            outs = [self.start_filts*(2**i) for i  in range(depth)]
        else:
            ins = [self.in_channels] + hid_channels[:-1]
            outs = hid_channels
        #print("ins", *ins, sep='\t')
        #print("outs", *outs, sep='\t')
            
        for i in range(depth):
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins[i], outs[i], pooling=pooling)
            self.down_convs.append(down_conv)
        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.reset_params()

    def forward(self, x):
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        return encoder_outs

class UNetDecode(TemplateUnet):
    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, hid_channels = None,
                 up_mode='transpose', 
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNetDecode, self).__init__()
        if hid_channels is not None:
            depth = len(hid_channels)

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")
        self.start_filts = start_filts
        self.num_classes = num_classes
        self.depth = depth

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        
        if hid_channels is None:
            self.hid_channels = [int(self.start_filts*(2**(i-1))) for i  in range(depth,0,-1)]
        else:
            self.hid_channels = hid_channels
#             ins = [int(self.start_filts*(2**(i-1))) for i  in range(depth,1,-1)]
#             outs = [int(self.start_filts*(2**(i-2))) for i  in range(depth,1,-1)]
            
#         else: 
        self.ins = self.hid_channels[::-1][:-1]
        self.outs = self.hid_channels[::-1][1:]
        print("ins", *self.ins, sep='\t')
        print("outs", *self.outs, sep='\t')
                
            
        self.up_convs = []
        for i in range(depth-1):
#             ins = outs
#             outs = ins // 2
            up_conv = UpConv(self.ins[i], self.outs[i], up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(self.outs[-1], self.num_classes)
        # add the list of modules to current module
        self.up_convs = nn.ModuleList(self.up_convs)
        self.reset_params()

    def forward(self, encoder_outs):
        x = encoder_outs[-1]
        for i, module in enumerate(self.up_convs):
            try:
                before_pool = encoder_outs[-(i+2)]
                x = module([x, before_pool])
            except Exception as ee:
                print("Exception on upconv #{:d},\tin-channels: {}\tout-channels: {}"
                      .format(i, self.ins[i], self.outs[i])
                      )
                print("input #1: {}".format(before_pool.shape))
                print("input #2: {}".format(x.shape))
                raise ee

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block = Bottleneck, num_classes=None, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = self._make_layer(block, 128, 1, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, img_batch):

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #x5 = self.layer5(x4)
        return [x2, x3, x4,]

#############################################################
class EncToLogits(nn.Module):
    def __init__(self, in_channels = 512, num_classes = 3):
        super(EncToLogits, self).__init__()
        intermediate_channels = 2*(int(math.sqrt(in_channels * num_classes))//2)
        self.seq = nn.Sequential(
                                nn.Conv2d(in_channels,intermediate_channels, 1),
                                nn.ELU(),
                                nn.Conv2d(intermediate_channels,num_classes, 1),
                               )
    def forward(self, inputs):
        return self.seq(inputs)

#############################################################

class RetinaNet(nn.Module):
    def __init__(self, num_classes, block=Bottleneck, layers=[3, 4, 6, 3],
                 prior = 0.01,
                 no_rpn = False,
                 no_semantic=False,
                 squeeze=True,
                 ):
        super(RetinaNet, self).__init__()
        self.squeeze = squeeze
        self.pyramid_levels = [3,4,5]
        self.no_rpn = no_rpn
        self.no_semantic = no_semantic
        self.encoder = ResNet(block=block, layers=layers)
        self.fpn_sizes = [self.get_out_channels(getattr(self.encoder,"layer%d"%nn)) for nn in [2,3,4]]
        #self.fpn_sizes.append([sz[-1]//2 for sz in self.fpn_sizes[-1]])
        print("fpn_sizes")
        print(*self.fpn_sizes, sep='\t')
#         if block == BasicBlock:
#             fpn_sizes = [self.layer2[-1].conv2.out_channels, 
#                          self.layer3[-1].conv2.out_channels, 
#                          self.layer4[-1].conv2.out_channels]
#             print
#         elif block == Bottleneck:
#             fpn_sizes = [self.layer2[-1].conv3.out_channels, 
#                          self.layer3[-1].conv3.out_channels, 
#                          self.layer4[-1].conv3.out_channels]

#         self.decoder = UNetDecode(num_classes, hid_channels=fpn_sizes)
        self.decoder = nn.Sequential(
                        UNetDecode(256, hid_channels=self.fpn_sizes[:-1]),
                        UpsampleBlock(in_channels = 256, out_channels=1+num_classes, steps=3)
                        )

        self.enc_to_logits  = nn.ModuleList([EncToLogits(n, num_classes+1) for n in self.fpn_sizes])
        #self.fpn = PyramidFeatures(self.fpn_sizes[0], self.fpn_sizes[1], self.fpn_sizes[2])
        #self.regressionModel = RegressionModel(256)
        #self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.fpn = PyramidFeatures(*([num_classes+1]*3))

        self.regressionModel = RegressionModel(num_classes+1)
        self.classificationModel = ClassificationModel(num_classes+1, num_classes=num_classes)

        self.anchors = Anchors(pyramid_levels=self.pyramid_levels, squeeze=squeeze)

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()
        
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    @classmethod
    def collect_rpn_scores(cls, rpn_model, features):
        classification = []
        for feature in features:
            res = rpn_model(feature)
            res = res.reshape(res.shape[:3]+ (-1,))
            classification.append(res)
        num_channels_ = classification[0].shape[1]
        classification = torch.cat(classification, dim=-1).\
                            permute((0,3,2,1,)).\
                            reshape(classification[0].shape[0],-1, num_channels_)
        return classification

    @classmethod
    def get_out_channels(cls, layer):
        out_ch = None
        for la in layer[-1].children():
            if hasattr(la,"out_channels"):
                out_ch = la.out_channels
        return out_ch

    def forward(self, img_batch):
        [x2, x3, x4] = self.encoder(img_batch)
#        import ipdb
#        ipdb.set_trace()
        if not self.no_semantic:
            sem_segm = self.decoder([x2, x3, x4])
            features = subsample_features(sem_segm, self.pyramid_levels)
        else:
            sem_segm = None
            features = [ e2l(x) for e2l, x in zip(self.enc_to_logits, [x2, x3, x4]) ]
            #features = self.fpn([x2, x3, x4])
            # features.append(nn.MaxPool2d(2)(features[-1]))

        if not self.no_rpn:
            if self.squeeze:
                regression = self.collect_rpn_scores(self.regressionModel, features)
                classification = self.collect_rpn_scores(self.classificationModel, features)
            else:
                regression = [self.regressionModel(f) for f in features]
                classification = [self.classificationModel(f) for f in features]
            #regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
            #classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        else:
            regression = None
            classification = None
        
        anchors = self.anchors(img_batch)
        if img_batch.type().startswith('torch.cuda'):
            anchors = anchors.cuda()

        if self.training:
            return [classification, regression, anchors, sem_segm,]
        else:
            if not self.no_rpn:
                transformed_anchors = self.regressBoxes(anchors, regression)
                transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

                scores = torch.max(classification, dim=2, keepdim=True)[0]

                scores_over_thresh = (scores>0.05)[0, :, 0]

                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just return
                    return [classification, regression, anchors, sem_segm ] +\
                        [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4),]

                classification_selected = classification[:, scores_over_thresh, :]
                transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                scores = scores[:, scores_over_thresh, :]

                # the code here uses [x0, y0, h, w] format
                # the NMS module accepts [x0, y0, x1, y1] format
                transformed_anchors_coord = transformed_anchors.clone()
                transformed_anchors_coord[...,2:] = transformed_anchors_coord[...,2:] + transformed_anchors[...,:2]

                anchors_nms_idx = nms(torch.cat([transformed_anchors_coord, scores], dim=2)[0, :, :], 0.5)

                nms_scores, nms_class = classification_selected[0, anchors_nms_idx, :].max(dim=1)

                return [classification, regression, anchors, sem_segm] + \
                       [nms_scores,
                        nms_class,
                        transformed_anchors[0, anchors_nms_idx, :], 
                        ]
            else:
                return [classification, regression, anchors, sem_segm]


#############################################################


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
