# forked from https://github.com/metrofun/E3D-LSTM


from functools import reduce
import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        # print(self.conv3d(input).shape, input.shape)
        # return self.conv_transpose3d(self.conv3d(input))
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


class E3DLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_size, kernel_size):
        super().__init__()

        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size

        # memory gates: input, cell(input modulation), forget
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)

        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)

        self.weight_xr = copy.deepcopy(self.weight_xi)
        self.weight_hr = copy.deepcopy(self.weight_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)

        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)

        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)

        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        r_flatten = r.view(batch_size, -1, channels)
        # BxtaoTHWxC
        c_history_flatten = c_history.view(batch_size, -1, channels)

        # Attention mechanism
        # BxTHWxC x BxtaoTHWxC' = B x THW x taoTHW
        scores = torch.einsum("bxc,byc->bxy", r_flatten, c_history_flatten)
        attention = F.softmax(scores, dim=2)

        return torch.einsum("bxy,byc->bxc", attention, c_history_flatten).view(*r.shape)

    def self_attention_fast(self, r, c_history):
        # Scaled Dot-Product but for tensors
        # instead of dot-product we do matrix contraction on twh dimensions
        scaling_factor = 1 / (reduce(operator.mul, r.shape[-3:], 1) ** 0.5)
        scores = torch.einsum("bctwh,lbctwh->bl", r, c_history) * scaling_factor

        attention = F.softmax(scores, dim=0)
        return torch.einsum("bl,lbctwh->bctwh", attention, c_history)

    def forward(self, x, c_history, m, h):
        # Normalized shape for LayerNorm is CxT×H×W
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        # R is CxT×H×W
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))

        recall = self.self_attention_fast(r, c_history)
        # nice_print(**locals())
        # mem_report()
        # cpu_stats()

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)
        # nice_print(**locals())

        return (c_history, m, h)

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)


class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau):
        super().__init__()

        self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    c_history, m, h = self._cells[cell_idx].init_hidden(batch_size, self._tau, input.device)
                    c_history_states.append(c_history)
                    h_states.append(h)

                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h = cell(x, c_history_states[cell_idx], m, h_states[cell_idx])
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h

            outputs.append(h)

        # NOTE Concat along the channels
        return torch.cat(outputs, dim=1)

class E3DLSTMNet(nn.Module):
    def __init__(self, time_steps=6, timeLen=6, hidden=64, lstm_layers=3, kernel=(1,5,5)):
        input_shape = (1, timeLen, 64, 64)
        super(E3DLSTMNet, self).__init__()
        self.encoder = E3DLSTM(input_shape, hidden, lstm_layers, kernel, input_shape[1])
        self.decoder = nn.Conv3d(hidden * time_steps, input_shape[0], kernel, padding=(0, 2, 2))
    def forward(self, input):
        return self.decoder(self.encoder(input))

if __name__ == '__main__':
    device = torch.device("cuda:0")
    dtype = torch.float

    h, w, c = 64, 64, 1
    timeLen = 3
    time_steps = 6


    # encoder = E3DLSTM(input_shape, hidden_size, lstm_layers, kernel, tau).type(dtype).to(device)
    # decoder = nn.Conv3d(hidden_size * time_steps, c, kernel, padding=(0, 2, 2)).type(dtype).to(device)
    #
    # input = torch.randn((time_steps, 1, c, temporal_frames, h, w)).float().to(device)
    # output = encoder(input)
    # print(output.shape)
    # output = decoder(output)
    # print(output.shape)

    input = torch.randn((time_steps, 1, 1, timeLen, 64, 64)).float().to(device)
    model = E3DLSTMNet().float().to(device)
    output = model(input)
    print(output.shape)
