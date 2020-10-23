import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, inChannel, hidden, kernel):
        super(ConvLSTMCell, self).__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels=inChannel+hidden,
                              out_channels=4*hidden,
                              kernel_size=kernel,
                              padding=kernel // 2)

    def forward(self, input, state):
        h, c = state
        input = torch.cat([input, h], dim=1)
        comb = self.conv(input)
        i, f, o, g = torch.split(comb, self.hidden, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c+i*g
        h = o*torch.tanh(c)
        return h, c

    def init_state(self, batch, size):
        h, w = size
        return (torch.zeros((batch, self.hidden, h, w)).to(self.conv.weight.device),
                torch.zeros((batch, self.hidden, h, w)).to(self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, inChannel, hiddenList, kernelList, return_all=False, timeLen=None, outConv=None, useLast=True, shape=None):
        super(ConvLSTM, self).__init__()
        self.nlayer = len(hiddenList)
        self.return_all = return_all
        LSTMList = []
        for i in range(self.nlayer):
            LSTMList.append(ConvLSTMCell(inChannel, hiddenList[i], kernelList[i]))
            inChannel = hiddenList[i]
        self.LSTMList = nn.ModuleList(LSTMList)
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
                state.append(self.LSTMList[i].init_state(b, (h, w)))
        outputList = []
        for l in range(self.nlayer):
            h, c = state[l]
            output = []
            for tt in range(t):
                h, c = self.LSTMList[l](input[:,tt], (h, c))
                output.append(h)
            input = torch.stack(output, dim=1)
            outputList.append(input)
            state[l] = (h, c)
        if not self.return_all:
            outputList = outputList[-1:]
        return outputList, state

    def Decoder(self, input, state=None):
        if not self.useLast:
            input = torch.zeros(self.shape).to(self.LSTMList[0].conv.weight.device)
        b, c, h, w = input.shape
        if state is None:
            state = []
            print('State == None is not recommend for decoding')
            for i in range(self.nlayer):
                state.append(self.LSTMList[i].init_state(b, (h, w)))
        t = self.timeLen

        outputList = []
        for tt in range(t):
            for l in range(self.nlayer):
                h, c = state[l]
                h, c = self.LSTMList[l](input, (h, c))
                input = h
                state[l] = h, c
            input = self.outConv(h)
            outputList.append(input)
        outputList = torch.stack(outputList, dim=1)
        return outputList, (h, c)


class ConvLSTMNet(nn.Module):
    def __init__(self, inChannel=1, hidden=[64, 64, 64], kernel=[3, 3, 3], outLen=6):
        super(ConvLSTMNet, self).__init__()
        outConv = nn.Conv2d(in_channels=hidden[-1], out_channels=inChannel, kernel_size=kernel[0], padding=kernel[0]//2)
        self.encoder = ConvLSTM(inChannel, hidden, kernel, True)
        self.decoder = ConvLSTM(inChannel, hidden, kernel, timeLen=outLen, outConv=outConv, useLast=True)
    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(input[:,-1], output[1])
        return output[0]


if __name__ == '__main__':
    import numpy as np
    device = torch.device('cuda:2')
    input = torch.from_numpy(np.random.random((1, 6, 1, 64, 64))).float().to(device)
    # model1 = ConvLSTM(1, [32, 64, 192], [3, 3, 3], 1).float().to(device)
    # model2 = ConvLSTM(1, [32, 64, 192], [3, 3, 3], timeLen=3, outConv=nn.Conv2d(in_channels=192, out_channels=1, kernel_size=3, padding=1), useLast=True).float().to(device)
    # output = model1(input)
    # output = model2(input[:,0], output[1])
    # # print(Predictor)
    # print(output[1][0].shape, output[1][1].shape, output[0].shape)


    model = ConvLSTMNet().float().to(device)
    output = model(input)
    print(output.shape)
