import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, inChannel, hidden, kernel):
        super(ConvGRUCell, self).__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels=inChannel+hidden,
                              out_channels=2*hidden,
                              kernel_size=kernel,
                              padding=kernel // 2)
        self.rconv = nn.Conv2d(in_channels=inChannel+hidden,
                               out_channels=hidden,
                               kernel_size=kernel,
                               padding=kernel // 2)

    def forward(self, input, state):
        h = state
        inp = torch.cat([input, h], dim=1)
        comb = self.conv(inp)
        z, r = torch.split(comb, self.hidden, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_= torch.tanh(self.rconv(torch.cat([input, r*h], dim=1)))
        h = (1-z)*h+z*h_
        return h

    def init_state(self, batch, size):
        h, w = size
        return torch.zeros((batch, self.hidden, h, w)).to(self.conv.weight.device)


class ConvGRU(nn.Module):
    def __init__(self, inChannel, hiddenList, kernelList, return_all=False, timeLen=None, outConv=None, useLast=True, shape=None):
        super(ConvGRU, self).__init__()
        self.nlayer = len(hiddenList)
        self.return_all = return_all
        ConvGRUList = []
        for i in range(self.nlayer):
            ConvGRUList.append(ConvGRUCell(inChannel, hiddenList[i], kernelList[i]))
            inChannel = hiddenList[i]
        self.ConvGRUList = nn.ModuleList(ConvGRUList)
        self.timeLen = timeLen
        self.outConv = outConv
        self.useLast = useLast
        self.shape = shape

    def forward(self, input, state=None):
        if self.outConv:
            return self.Decoder(input, state)
        else:
            return self.Encoder(input, state)

    def Encoder(self, input, state=None):
        b, t, c, h, w = input.shape
        if state is None:
            state = []
            for i in range(self.nlayer):
                state.append(self.ConvGRUList[i].init_state(b, (h, w)))
        outputList = []
        for l in range(self.nlayer):
            h = state[l]
            output = []
            for tt in range(t):
                h = self.ConvGRUList[l](input[:,tt], h)
                output.append(h)
            input = torch.stack(output, dim=1)
            outputList.append(input)
            state[l] = h
        if not self.return_all:
            outputList = outputList[-1:]
        return outputList, state

    def Decoder(self, input, state=None):
        if not self.useLast:
            input = torch.zeros(self.shape).to(self.ConvGRUList[0].conv.weight.device)
        b, c, h, w = input.shape
        if state is None:
            state = []
            print('State == None is not recommend for decoding')
            for i in range(self.nlayer):
                state.append(self.ConvGRUList[i].init_state(b, (h, w)))
        t = self.timeLen

        outputList = []
        for tt in range(t):
            for l in range(self.nlayer):
                h = state[l]
                h = self.ConvGRUList[l](input, h)
                input = h
                state[l] = h
            input = self.outConv(h)
            outputList.append(input)
        outputList = torch.stack(outputList, dim=1)
        return outputList, h


if __name__ == '__main__':
    import numpy as np
    device = torch.device('cuda:6')
    input = torch.from_numpy(np.random.random((1, 6, 1, 64, 64))).float().to(device)
    model1 = ConvGRU(1, [32, 64, 192], [3, 3, 3], 1).float().to(device)
    model2 = ConvGRU(1, [32, 64, 192], [3, 3, 3], timeLen=3, outConv=nn.Conv2d(in_channels=192, out_channels=1, kernel_size=3, padding=1), useLast=True).float().to(device)
    output = model1(input)
    print(output[1][0].shape, output[0][0].shape)
    output = model2(input[:,0], output[1])
    print(output[1].shape, output[0].shape)

