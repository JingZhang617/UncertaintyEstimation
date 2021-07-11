import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class FCDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(ndf)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x, pred, depth=None):
        if depth != None:
            x = torch.cat((x,pred,depth),1)
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.leaky_relu(x)
        else:
            x = torch.cat((x, pred), 1)
            x = self.conv1_2(x)
            x = self.bn1_2(x)
            x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class Encoder_x(nn.Module):
    def __init__(self, input_channels, latent_size, channels=32):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels


        self.fc1 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2 = nn.Linear(channels * 8 * 11 * 11, latent_size)


        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 11 * 11)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar


class Encoder_xy(nn.Module):
    def __init__(self, input_channels, latent_size, channels=32):
        super(Encoder_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2 = nn.Linear(channels * 8 * 11 * 11, latent_size)


        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        output = self.leakyrelu(self.bn1(self.layer1(x)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 11 * 11)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar

class Feature_encoder(nn.Module):
    def __init__(self,channel):
        super(Feature_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.conv4 = BasicConv2d(2048, channel, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(1024, channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(512, channel, kernel_size=3, padding=1)
        self.conv1 = BasicConv2d(256, channel, kernel_size=3, padding=1)

        if self.training:
            self.initialize_weights()


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)

        return conv1_feat,conv2_feat,conv3_feat,conv4_feat

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

class Feature_decoder(nn.Module):
    def __init__(self,latent_dim,channel):
        super(Feature_decoder, self).__init__()
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.noise_conv = nn.Conv2d(channel + latent_dim, channel, kernel_size=1, padding=0)
        self.racb_4321 = RCAB(channel * 4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)



    def forward(self, x1,x2,x3,x4,z_noise):
        x4 = torch.cat((x4, z_noise), 1)
        x4 = self.noise_conv(x4)
        feat_cat = torch.cat(
            (x1, self.upsample2(x2), self.upsample2(x3), self.upsample2(x4)), 1)
        feat_cat = self.racb_4321(feat_cat)
        x5 = self.layer5(feat_cat)

        return self.upsample4(x5)


class ResNet_Baseline(nn.Module):
    # resnet based encoder decoder
    def __init__(self, latent_dim=8,channel=32):
        super(ResNet_Baseline, self).__init__()
        self.resnet = B2_ResNet()
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)

        self.xy_encoder = Encoder_xy(4, latent_dim)
        self.x_encoder = Encoder_x(3, latent_dim)

        self.spatial_axes = [2, 3]
        self.feature_enc = Feature_encoder(channel)

        self.xy_encoder = Encoder_xy(4, latent_dim)
        self.x_encoder = Encoder_x(3, latent_dim)

        self.prior_feat_dec = Feature_decoder(latent_dim, channel)
        self.post_feat_dec = Feature_decoder(latent_dim, channel)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def forward(self, x, y=None,training = True):
        x1, x2, x3, x4 = self.feature_enc(x)
        if training:
            self.posterior, muxy, logvarxy = self.xy_encoder(torch.cat((x, y), 1))
            self.prior, mux, logvarx = self.x_encoder(x)
            latent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
            z_noise_post = self.reparametrize(muxy, logvarxy)
            z_noise_prior = self.reparametrize(mux, logvarx)

            z_noise_prior = torch.unsqueeze(z_noise_prior, 2)
            z_noise_prior = self.tile(z_noise_prior, 2, x4.shape[self.spatial_axes[0]])
            z_noise_prior = torch.unsqueeze(z_noise_prior, 3)
            z_noise_prior = self.tile(z_noise_prior, 3, x4.shape[self.spatial_axes[1]])

            z_noise_post = torch.unsqueeze(z_noise_post, 2)
            z_noise_post = self.tile(z_noise_post, 2, x4.shape[self.spatial_axes[0]])
            z_noise_post = torch.unsqueeze(z_noise_post, 3)
            z_noise_post = self.tile(z_noise_post, 3, x4.shape[self.spatial_axes[1]])

            post_pred = self.post_feat_dec(x1,x2,x3,x4,z_noise_post)
            prior_pred = self.prior_feat_dec(x1, x2, x3, x4, z_noise_prior)
            return post_pred, prior_pred, latent_loss
        else:
            self.prior, mux, logvarx = self.x_encoder(x)
            z_noise_prior = self.reparametrize(mux, logvarx)
            z_noise_prior = torch.unsqueeze(z_noise_prior, 2)
            z_noise_prior = self.tile(z_noise_prior, 2, x4.shape[self.spatial_axes[0]])
            z_noise_prior = torch.unsqueeze(z_noise_prior, 3)
            z_noise_prior = self.tile(z_noise_prior, 3, x4.shape[self.spatial_axes[1]])
            prior_pred = self.prior_feat_dec(x1, x2, x3, x4, z_noise_prior)
            return prior_pred


