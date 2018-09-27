import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(27, 32, 32), res_blocks=4, c_dim=9):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model1 = [nn.Conv2d(channels + c_dim, 64, 5, stride=1, padding=2, bias=False),
                 nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                 nn.ReLU(inplace=True)]

        # Downsampling
        curr_dim = 64

        model1 += [nn.Conv2d(curr_dim, curr_dim * 2, 3, stride=2, padding=1, bias=False),
                   nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                   nn.ReLU(inplace=True)]
        curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model1 += [ResidualBlock(curr_dim)]
            model1 += [Self_Attn(curr_dim, "relu")]

        # Upsampling
        model2 = []
        for _ in range(2):
            model2 += [SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False)),
                      nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True)]
            curr_dim = curr_dim // 2
        model2 += [Self_Attn(curr_dim, "relu")]
        # Output layer
        model2 += [nn.Conv2d(curr_dim, 3, 7, stride=1, padding=3),
                  nn.Tanh()]
        self.downmodel = nn.Sequential(*model1)
        self.upmodel = nn.Sequential(*model2)

    def forward(self, img, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, img.size(2), img.size(3))
        x = torch.cat([img, c], 1)
        x = self.downmodel(x)
        return self.upmodel(x)


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, img_shape=(27, 32, 32), c_dim=9, n_strided=4):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                      nn.LeakyReLU(0.01)]
            layers += [Self_Attn(out_filters, "relu")]
            return layers

        layers = discriminator_block(channels+c_dim, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim*2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2**n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img, gen_img):
        input_ = torch.cat([img, gen_img], 1)
        feature_repr = self.model(input_)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)

