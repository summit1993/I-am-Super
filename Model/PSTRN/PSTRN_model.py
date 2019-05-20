import torch
import torch.nn as nn

class FSTRN_Model(nn.Module):
    def __init__(self, rfb_num):
        super(FSTRN_Model, self).__init__()
        self.lfe = self.create_bottle_net(has_relu=False)
        self.RFB_blocks = nn.Sequential()
        self.rfb_num = rfb_num
        for i in range(rfb_num):
            self.RFB_blocks.add_module('rfb_' + str(i + 1), self.create_bottle_net())
        self.lrl = nn.Sequential()
        self.lrl.add_module('PReLU', nn.PReLU())
        self.lsr = nn.Sequential()
        self.lsr.add_module('conv1', nn.Conv2d(3, 3, 3, 1, 1))
        # H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
        self.lsr.add_module('dconv1', nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.lsr.add_module('dconv2', nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.lsr.add_module('conv2', nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        # x[0]: LR Volume; x[1]: up-sampled LR image
        f_0 = self.lfe(x[0])
        f = f_0 + self.RFB_blocks.__getattr__('rfb_1')(f_0)
        for i in range(1, self.rfb_num):
            f = f + self.RFB_blocks.__getattr__('rfb_' + str(i + 1))(f)
        f += f_0
        f = torch.sum(f, 2)
        f = self.lrl(f)
        f = self.lsr(f)
        f += x[1]
        return f

    def create_bottle_net(self, has_relu=True):
        bottle_net = nn.Sequential()
        if has_relu:
            bottle_net.add_module('PReLU', nn.PReLU())
        bottle_net.add_module('SptioConv', nn.Conv3d(3, 3, (1, 3, 3), 1, (0, 1, 1)))
        bottle_net.add_module('TemporalConv', nn.Conv3d(3, 3, (3, 1, 1), 1, (1, 0, 0)))
        return bottle_net