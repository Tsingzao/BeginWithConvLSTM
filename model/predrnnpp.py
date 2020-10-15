# froked from https://github.com/zhangyanbiao/predrnn-_pytorch

import torch
import torch.nn as nn


def tensor_layer_norm(num_features):
    return nn.BatchNorm2d(num_features)


class CausalLSTMCell(nn.Module):
    def __init__(self, layer_name, num_hidden_in, num_hidden_out,
                 seq_shape, forget_bias, tln=True):
        super(CausalLSTMCell, self).__init__()
        """Initialize the Causal LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden_in: number of units for input tensor.
            num_hidden_out: number of units for output tensor.
            seq_shape: shape of a sequence.
            forget_bias: float, The bias added to forget gates.
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name
        self.num_hidden_in = num_hidden_in
        self.num_hidden_out = num_hidden_out
        self.batch = seq_shape[0]
        self.height = seq_shape[3]
        self.width = seq_shape[2]
        self.layer_norm = tln
        self._forget_bias = forget_bias

        self.bn_h_cc = tensor_layer_norm(self.num_hidden_out * 4)
        self.bn_c_cc = tensor_layer_norm(self.num_hidden_out * 3)
        self.bn_m_cc = tensor_layer_norm(self.num_hidden_out * 3)
        self.bn_x_cc = tensor_layer_norm(self.num_hidden_out * 7)
        self.bn_c2m = tensor_layer_norm(self.num_hidden_out * 4)
        self.bn_o_m = tensor_layer_norm(self.num_hidden_out)

        self.h_cc_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 4, 5, 1, 2)
        self.c_cc_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 3, 5, 1, 2)
        self.m_cc_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 3, 5, 1, 2)
        self.x_cc_conv = nn.Conv2d(self.num_hidden_in, self.num_hidden_out * 7, 5, 1, 2)
        self.c2m_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 4, 5, 1, 2)
        self.o_m_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out, 5, 1, 2)
        self.o_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out, 5, 1, 2)
        self.cell_conv = nn.Conv2d(self.num_hidden_out * 2, self.num_hidden_out, 1, 1, 0)

    def init_state(self):
        return torch.zeros((self.batch, self.num_hidden_out, self.width, self.height), dtype=torch.float32).to(self.cell_conv.weight.device)

    def forward(self, x, h, c, m):
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()
        h_cc = self.h_cc_conv(h)
        c_cc = self.c_cc_conv(c)
        m_cc = self.m_cc_conv(m)
        if self.layer_norm:
            h_cc = self.bn_h_cc(h_cc)
            c_cc = self.bn_c_cc(c_cc)
            m_cc = self.bn_m_cc(m_cc)

        i_h, g_h, f_h, o_h = torch.split(h_cc, self.num_hidden_out, 1)
        i_c, g_c, f_c = torch.split(c_cc, self.num_hidden_out, 1)
        i_m, f_m, m_m = torch.split(m_cc, self.num_hidden_out, 1)
        if x is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.x_cc_conv(x)
            if self.layer_norm:
                x_cc = self.bn_x_cc(x_cc)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc, self.num_hidden_out, 1)
            i = torch.sigmoid(i_x + i_h + i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)
        c_new = f * c + i * g
        c2m = self.c2m_conv(c_new)
        if self.layer_norm:
            c2m = self.bn_c2m(c2m)

        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden_out, 1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self._forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)
        m_new = ff * torch.tanh(m_m) + ii * gg
        o_m = self.o_m_conv(m_new)
        if self.layer_norm:
            o_m = self.bn_o_m(o_m)
        if x is None:
            o = torch.tanh(o_c + o_m)

        else:
            o = torch.tanh(o_x + o_c + o_m)
        o = self.o_conv(o)
        # 此时c_new以及m_new的格式均为[b,c,w,h]
        cell = torch.cat([c_new, m_new], 1)
        cell = self.cell_conv(cell)
        h_new = o * torch.tanh(cell)
        return h_new, c_new, m_new


class GHU(nn.Module):
    def __init__(self, layer_name, inputs_shape, num_features, tln=False):
        super(GHU, self).__init__()
        """Initialize the Gradient Highway Unit.
        """
        self.layer_name = layer_name
        self.num_features = num_features
        self.layer_norm = tln
        self.batch = inputs_shape[0]
        self.height = inputs_shape[3]
        self.width = inputs_shape[2]

        self.bn_z_concat = tensor_layer_norm(self.num_features * 2)
        self.bn_x_concat = tensor_layer_norm(self.num_features * 2)

        self.z_concat_conv = nn.Conv2d(self.num_features, self.num_features * 2, 5, 1, 2)
        self.x_concat_conv = nn.Conv2d(self.num_features, self.num_features * 2, 5, 1, 2)

    def init_state(self):
        return torch.zeros((self.batch, self.num_features, self.width, self.height), dtype=torch.float32).to(self.z_concat_conv.weight.device)

    def forward(self, x, z):
        if z is None:
            z = self.init_state()
        z_concat = self.z_concat_conv(z)
        if self.layer_norm:
            z_concat = self.bn_z_concat(z_concat)

        x_concat = self.x_concat_conv(x)
        if self.layer_norm:
            x_concat = self.bn_x_concat(x_concat)

        gates = torch.add(x_concat, z_concat)
        p, u = torch.split(gates, self.num_features, 1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new


class RNN(nn.Module):
    def __init__(self, shape, batch, num_layers, num_hidden, seq_length, tln=True):
        super(RNN, self).__init__()

        self.img_width = shape[-2]
        self.img_height = shape[-1]
        self.total_length = shape[1]
        self.input_length = shape[0]
        self.shape = [batch, shape[2], shape[3], shape[4]]
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        ghu_list = []

        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = 1
            else:
                num_hidden_in = self.num_hidden[i - 1]
            cell_list.append(CausalLSTMCell('lstm_' + str(i + 1),
                                    num_hidden_in,
                                    num_hidden[i],
                                    self.shape, 1.0, tln=tln))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[-1], 1, 1, 1, 0)
        ghu_list.append(GHU('highway', self.shape, self.num_hidden[1], tln=tln))
        self.ghu_list = nn.ModuleList(ghu_list)

    def forward(self, images):
        # [batch, length, channel, width, height]

        batch = images.shape[0]
        height = images.shape[3]
        width = images.shape[4]

        next_images = []
        h_t = []
        c_t = []
        z_t = None
        m_t = None

        for i in range(self.num_layers):
            h_t.append(None)
            c_t.append(None)

        for t in range(self.total_length):
            if t < self.input_length:
                net = images[:, t]
            h_t[0], c_t[0], m_t = self.cell_list[0](net, h_t[0], c_t[0], m_t)
            z_t = self.ghu_list[0](h_t[0], z_t)
            h_t[1], c_t[1], m_t = self.cell_list[1](z_t, h_t[1], c_t[1], m_t)

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], m_t = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_images.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_images = torch.stack(next_images, dim=1)
        out = next_images
        next_images = []

        return out[:,-(self.total_length-self.input_length):]


if __name__ == '__main__':
    device = torch.device('cuda:6')
    shape = [6, 9, 1, 64, 64]
    a = torch.randn((1, 6, 1, 64, 64)).float().to(device)
    numlayers = 3
    predrnn = RNN(shape, 1, numlayers, [64, 64, 64], 6, True).float().to(device)
    predict = predrnn(a)
    print(predict.shape)
