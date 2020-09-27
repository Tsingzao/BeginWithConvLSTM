# forked from https://github.com/coolsunxu/MIM_Pytorch


import torch
import torch.nn as nn


def tensor_layer_norm(num_features):
    return nn.BatchNorm2d(num_features)


class SpatioTemporalLSTMCell(nn.Module):  # stlstm
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, x_shape_in, tln=False, initializer=None):
        super(SpatioTemporalLSTMCell, self).__init__()

        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            forget_bias: float, The bias added to forget gates (see above).
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name  # 当前网络层名
        self.filter_size = filter_size  # 卷积核大小
        self.num_hidden_in = num_hidden_in  # 隐藏层输入大小
        self.num_hidden = num_hidden  # 隐藏层数量
        self.batch = seq_shape[0]  # batch_size
        self.height = seq_shape[3]  # 图片高度
        self.width = seq_shape[4]  # 图片宽度
        self.x_shape_in = x_shape_in  # 通道数
        self.layer_norm = tln  # 是否归一化
        self._forget_bias = 1.0  # 遗忘参数

        # 建立网络层
        # h
        self.t_cc = nn.Conv2d(self.num_hidden_in,
                              self.num_hidden * 4,  # 网络输入 输出通道数
                              self.filter_size, 1, padding=2  # 滤波器大小 步长 填充方式
                              )

        # m
        self.s_cc = nn.Conv2d(self.num_hidden_in,
                              self.num_hidden * 4,  # 网络输入 输出通道数
                              self.filter_size, 1, padding=2  # 滤波器大小 步长 填充方式
                              )

        # x
        self.x_cc = nn.Conv2d(self.x_shape_in,
                              self.num_hidden * 4,  # 网络输入 输出通道数
                              self.filter_size, 1, padding=2  # 滤波器大小 步长 填充方式
                              )

        # c
        self.c_cc = nn.Conv2d(self.num_hidden * 2,
                              self.num_hidden,  # 网络输入 输出通道数
                              1, 1, padding=0  # 滤波器大小 步长 填充方式
                              )

        # bn
        self.bn_t_cc = tensor_layer_norm(self.num_hidden * 4)
        self.bn_s_cc = tensor_layer_norm(self.num_hidden * 4)
        self.bn_x_cc = tensor_layer_norm(self.num_hidden * 4)

    def init_state(self):  # 初始化lstm 隐藏层状态
        return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
                           dtype=torch.float32).to(self.c_cc.weight.device)

    def forward(self, x, h, c, m):
        # x [batch, in_channels, in_height, in_width]
        # h c m [batch, num_hidden, in_height, in_width]

        # 初始化隐藏层 记忆 空间
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()

        # 计算网络输出
        t_cc = self.t_cc(h)
        s_cc = self.s_cc(m)
        x_cc = self.x_cc(x)

        if self.layer_norm:
            # 计算均值 标准差 归一化
            t_cc = self.bn_t_cc(t_cc)
            s_cc = self.bn_s_cc(s_cc)
            x_cc = self.bn_x_cc(x_cc)

        # 在第3维度上切分为4份 因为隐藏层是4*num_hidden
        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1)  # [batch, num_hidden, in_height, in_width]
        i_t, g_t, f_t, o_t = torch.split(t_cc, self.num_hidden, 1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f = torch.sigmoid(f_x + f_t + self._forget_bias)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        new_c = f * c + i * g
        cell = torch.cat((new_c, new_m), 1)  # [batch, 2*num_hidden, in_height, in_width]

        cell = self.c_cc(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m  # 大小均为 [batch, num_hidden, in_height, in_width]


class MIMBlock(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, x_shape_in, tln=False, initializer=None):
        super(MIMBlock, self).__init__()

        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            forget_bias: float, The bias added to forget gates (see above).
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name  # 当前网络层名
        self.filter_size = filter_size  # 卷积核大小
        self.num_hidden_in = num_hidden_in  # 隐藏层输入
        self.num_hidden = num_hidden  # 隐藏层大小
        self.convlstm_c = None  #
        self.batch = seq_shape[0]  # batch_size
        self.height = seq_shape[3]  # 图片高度
        self.width = seq_shape[4]  # 图片宽度
        self.x_shape_in = x_shape_in  # 通道数
        self.layer_norm = tln  # 是否归一化
        self._forget_bias = 1.0  # 遗忘参数

        # MIMS

        # h_t
        self.mims_h_t = nn.Conv2d(self.num_hidden,
                                  self.num_hidden * 4,
                                  self.filter_size, 1, padding=2
                                  )

        # c_t
        self.ct_weight = nn.Parameter(torch.randn((self.num_hidden * 2, self.height, self.width)))

        # x
        self.mims_x = nn.Conv2d(self.num_hidden,
                                self.num_hidden * 4,
                                self.filter_size, 1, padding=2
                                )

        # oc
        self.oc_weight = nn.Parameter(torch.randn((self.num_hidden, self.height, self.width)))

        # bn
        self.bn_h_concat = tensor_layer_norm(self.num_hidden * 4)
        self.bn_x_concat = tensor_layer_norm(self.num_hidden * 4)

        # MIMBLOCK
        # h
        self.t_cc = nn.Conv2d(self.num_hidden_in,
                              self.num_hidden * 3,  # 网络输入 输出通道数
                              self.filter_size, 1, padding=2  # 滤波器大小 步长 填充方式
                              )

        # m
        self.s_cc = nn.Conv2d(self.num_hidden_in,
                              self.num_hidden * 4,  # 网络输入 输出通道数
                              self.filter_size, 1, padding=2  # 滤波器大小 步长 填充方式
                              )

        # x
        self.x_cc = nn.Conv2d(self.x_shape_in,
                              self.num_hidden * 4,  # 网络输入 输出通道数
                              self.filter_size, 1, padding=2  # 滤波器大小 步长 填充方式
                              )

        # c
        self.c_cc = nn.Conv2d(self.num_hidden * 2,
                              self.num_hidden,  # 网络输入 输出通道数
                              1, 1, padding=0  # 滤波器大小 步长 填充方式
                              )

        # bn
        self.bn_t_cc = tensor_layer_norm(self.num_hidden * 3)
        self.bn_s_cc = tensor_layer_norm(self.num_hidden * 4)
        self.bn_x_cc = tensor_layer_norm(self.num_hidden * 4)

    def init_state(self):  # 初始化lstm 隐藏层状态
        return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
                           dtype=torch.float32).to(self.c_cc.weight.device)

    def MIMS(self, x, h_t, c_t):  # MIMS

        # h_t c_t[batch, in_height, in_width, num_hidden]
        # 初始化隐藏层 记忆 空间

        if h_t is None:
            h_t = self.init_state()
        if c_t is None:
            c_t = self.init_state()

        # h_t
        h_concat = self.mims_h_t(h_t)

        if self.layer_norm:  # 是否归一化
            h_concat = self.bn_h_concat(h_concat)

        # 在第3维度上切分为4份 因为隐藏层是4*num_hidden
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

        # ct_weight
        ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            # x
            x_concat = self.mims_x(x)

            if self.layer_norm:
                x_concat = self.bn_x_concat(x_concat)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        # oc_weight
        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, diff_h, h, c, m):
        # 初始化隐藏层 记忆 空间

        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()
        if diff_h is None:
            diff_h = torch.zeros_like(h)

        # h
        t_cc = self.t_cc(h)

        # m
        s_cc = self.s_cc(m)

        # x
        x_cc = self.x_cc(x)

        if self.layer_norm:
            t_cc = self.bn_t_cc(t_cc)
            s_cc = self.bn_s_cc(s_cc)
            x_cc = self.bn_x_cc(x_cc)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, 1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_

        # MIMS
        c, self.convlstm_c = self.MIMS(diff_h, c, self.convlstm_c)

        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)

        # c
        cell = self.c_cc(cell)

        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m  # 大小均为 [batch, in_height, in_width, num_hidden]


class MIMN(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden, seq_shape, tln=True, initializer=0.001):
        super(MIMN, self).__init__()
        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            tln: whether to apply tensor layer normalization.
        """
        self.layer_name = layer_name  # 当前网络层名
        self.filter_size = filter_size  # 卷积核大小
        self.num_hidden = num_hidden  # 隐藏层大小
        self.layer_norm = tln  # 是否归一化
        self.batch = seq_shape[0]  # batch_size
        self.height = seq_shape[3]  # 图片高度
        self.width = seq_shape[4]  # 图片宽度
        self._forget_bias = 1.0  # 遗忘参数

        # h_t
        self.h_t = nn.Conv2d(self.num_hidden,
                             self.num_hidden * 4,
                             self.filter_size, 1, padding=2
                             )

        # c_t
        self.ct_weight = nn.Parameter(torch.randn((self.num_hidden * 2, self.height, self.width)))

        # x
        self.x = nn.Conv2d(self.num_hidden,
                           self.num_hidden * 4,
                           self.filter_size, 1, padding=2
                           )

        # oc
        self.oc_weight = nn.Parameter(torch.randn((self.num_hidden, self.height, self.width)))

        # bn
        self.bn_h_concat = tensor_layer_norm(self.num_hidden * 4)
        self.bn_x_concat = tensor_layer_norm(self.num_hidden * 4)

    def init_state(self):  # 初始化lstm 隐藏层状态
        shape = [self.batch, self.num_hidden, self.height, self.width]
        return torch.zeros(shape, dtype=torch.float32).to(self.oc_weight.device)

    def forward(self, x, h_t, c_t):

        # h c [batch, num_hidden, in_height, in_width]

        # 初始化隐藏层 记忆 空间

        if h_t is None:
            h_t = self.init_state()
        if c_t is None:
            c_t = self.init_state()

        # 1
        h_concat = self.h_t(h_t)

        if self.layer_norm:
            h_concat = self.bn_h_concat(h_concat)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

        # 2 变量 可训练
        ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            # 3 x
            x_concat = self.x(x)

            if self.layer_norm:
                x_concat = self.bn_x_concat(x_concat)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        # 4 变量 可训练
        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new  # 大小均为 [batch, in_height, in_width, num_hidden]


class MIM(nn.Module):  # stlstm
    def __init__(self, shape, num_layers, num_hidden, filter_size,
                 total_length=20, input_length=10, tln=True):
        super(MIM, self).__init__()

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.filter_size = filter_size
        self.total_length = total_length
        self.input_length = input_length
        self.tln = tln

        self.gen_images = []  # 存储生成的图片
        self.stlstm_layer = nn.ModuleList()  # 存储 stlstm 和 mimblock
        self.stlstm_layer_diff = nn.ModuleList()  # 存储 mimn
        self.cell_state = []  # 存储 stlstm_layer 的记忆
        self.hidden_state = []  # 存储 stlstm_layer 的隐藏层输出
        self.cell_state_diff = []  # 存储 stlstm_layer_diff 的记忆
        self.hidden_state_diff = []  # 存储 stlstm_layer_diff 的隐藏层输出
        self.shape = shape  # 输入形状
        self.output_channels = shape[-3]  # 输出的通道数

        for i in range(self.num_layers):  # 隐藏层数目
            if i == 0:
                num_hidden_in = self.num_hidden[self.num_layers - 1]  # 隐藏层的输入 前一时间段最后一层的输出为后一时间段第一层的输入
            else:
                num_hidden_in = self.num_hidden[i - 1]  # 隐藏层的输入
            if i < 1:  # 初始层 使用 stlstm
                new_stlstm_layer = SpatioTemporalLSTMCell('stlstm_' + str(i + 1),
                                          self.filter_size,
                                          num_hidden_in,
                                          self.num_hidden[i],
                                          self.shape,
                                          self.output_channels,
                                          tln=self.tln)
            else:  # 后续层 使用 mimblock
                new_stlstm_layer = MIMBlock('stlstm_' + str(i + 1),
                                            self.filter_size,
                                            num_hidden_in,
                                            self.num_hidden[i],
                                            self.shape,
                                            self.num_hidden[i - 1],
                                            tln=self.tln)
            self.stlstm_layer.append(new_stlstm_layer)  # 列表
            self.cell_state.append(None)  # 记忆
            self.hidden_state.append(None)  # 状态

        for i in range(self.num_layers - 1):  # 添加 MIMN
            new_stlstm_layer = MIMN('stlstm_diff' + str(i + 1),
                                    self.filter_size,
                                    self.num_hidden[i + 1],
                                    self.shape,
                                    tln=self.tln)
            self.stlstm_layer_diff.append(new_stlstm_layer)  # 列表
            self.cell_state_diff.append(None)  # 记忆
            self.hidden_state_diff.append(None)  # 状态

        self.st_memory = None  # 空间存储

        # 生成图片
        self.x_gen = nn.Conv2d(self.num_hidden[self.num_layers - 1],
                               self.output_channels, 1, 1, padding=0
                               )

    def forward(self, images, schedual_sampling_bool):

        for time_step in range(self.total_length - 1):  # 时间步长
            print('time_step: ' + str(time_step))

            if time_step < self.input_length:  # 小于输入步长
                x_gen = images[:, time_step]  # 输入大小为 [batch, in_channel,in_height, in_width]
            else:
                # 掩模 mask
                x_gen = schedual_sampling_bool[:, time_step - self.input_length] * images[:, time_step] + \
                        (1 - schedual_sampling_bool[:, time_step - self.input_length]) * x_gen

            preh = self.hidden_state[0]  # 初始化状态
            self.hidden_state[0], self.cell_state[0], self.st_memory = self.stlstm_layer[0](
                # 使用的是 stlstm 输出 hidden_state[0], cell_state[0], st_memory
                x_gen, self.hidden_state[0], self.cell_state[0], self.st_memory)

            # 对于 mimblock
            for i in range(1, self.num_layers):
                print('i: ' + str(i))
                if time_step > 0:
                    if i == 1:
                        self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            # 先求出 mimn
                            self.hidden_state[i - 1] - preh, self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
                    else:
                        self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            # 先求出 mimn
                            self.hidden_state_diff[i - 2], self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(self.hidden_state[i - 1]), None, None)

                # 接下来计算 mimblock
                preh = self.hidden_state[i]
                self.hidden_state[i], self.cell_state[i], self.st_memory = self.stlstm_layer[i](  # mimblock
                    self.hidden_state[i - 1], self.hidden_state_diff[i - 1], self.hidden_state[i], self.cell_state[i],
                    self.st_memory)

            # 生成图像 取最后一层的隐藏层状态
            x_gen = self.x_gen(self.hidden_state[self.num_layers - 1])

            self.gen_images.append(x_gen)

        self.gen_images = torch.stack(self.gen_images, dim=1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(self.gen_images, images[:, 1:])
        return [self.gen_images, loss]


if __name__ == '__main__':
    device = torch.device('cuda:2')
    a = torch.randn((5, 6, 1, 64, 64)).float().to(device)
    b = torch.randn((5, 2, 1, 64, 64)).float().to(device)

    num_layers = 3
    num_hidden = [64, 64, 64]
    filter_size = 5
    total_length = a.shape[1]
    input_length = a.shape[1]
    shape = [5, 6, 1, 64, 64]

    stlstm = MIM(shape, num_layers, num_hidden, filter_size, total_length, input_length).float().to(device)

    new = stlstm(a, b)
    print(new[0].shape)
    print(new[1])