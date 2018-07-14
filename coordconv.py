import torch
import torch.nn as nn
# from https://raw.githubusercontent.com/mkocabas/CoordConv-pytorch/master/CoordConv.py


class AddCoordsTh(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoordsTh, self).__init__()
        #self.x_dim = x_dim
        #self.y_dim = y_dim
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """

        use_gpu = input_tensor.type().startswith('torch.cuda')

        batch_size_tensor = input_tensor.shape[0]
        self.x_dim = input_tensor.shape[-2] 
        self.y_dim = input_tensor.shape[-1]

        xx_ones = torch.ones([1, self.x_dim], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.y_dim], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)

        xx_channel = xx_channel.float() / (self.x_dim - 1)
        yy_channel = yy_channel.float() / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        if use_gpu:
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            if use_gpu:
                rr = rr.cuda()
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self,# x_dim, y_dim, 
                 in_channels, *args, 
                 dimensions = 2,
                 with_r=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoordsTh(with_r=with_r,) #x_dim=x_dim, y_dim=y_dim,)
        self.dimensions = dimensions
        in_channels += self.dimensions
        self.conv = nn.Conv2d(in_channels, *args, **kwargs)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret
