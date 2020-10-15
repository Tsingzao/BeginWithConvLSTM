import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    device = torch.device('cuda: 0')
    gru = TrajGRUCell(4, 32, 3).to(device).float()
    input = torch.randn((1, 4, 256, 256)).float().to(device)
    state = torch.randn((1, 32, 256, 256)).float().to(device)
    output = gru(input, state)
    print(output.shape)
'''
class TrajGRUCell(nn.Module):
    def __init__(self, inChannel, hidden, kernel, L=13):
        super(TrajGRUCell, self).__init__()
        self.convUV = nn.Sequential(*[nn.Conv2d(in_channels=inChannel+hidden, out_channels=32, kernel_size=(5,5), stride=1, padding=(2,2), dilation=(1,1)),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(in_channels=32, out_channels=L*2, kernel_size=(5,5), stride=1, padding=(2,2), dilation=(1,1))])
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels=inChannel+hidden*L, out_channels=2*hidden, kernel_size=kernel, padding=kernel // 2)
        self.rconvI = nn.Conv2d(in_channels=inChannel, out_channels=hidden, kernel_size=kernel, padding=kernel // 2)
        self.rconvH = nn.Conv2d(in_channels=hidden*L, out_channels=hidden, kernel_size=kernel, padding=kernel // 2)

    def wrap(self, input, flow, device=torch.device('cpu')):
        B, C, H, W = input.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if not grid.cuda:
            print(1)
        if not flow.cuda:
            print(2)
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output

    def forward(self, input, state):
        h = state
        inp = torch.cat([input, h], dim=1)
        flows = self.convUV(inp)
        flows = torch.split(flows, 2, dim=1)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(self.wrap(h, -flow, device=self.conv.weight.device))
        wrapped_data = torch.cat(wrapped_data, dim=1)
        inp = torch.cat([input, wrapped_data], dim=1)
        comb = self.conv(inp)
        z, r = torch.split(comb, self.hidden, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_ = F.leaky_relu(self.rconvI(input)+r*self.rconvH(wrapped_data), negative_slope=0.2, inplace=True)
        h = (1-z)*h_+z*h
        return h

    def init_state(self, batch, size):
        h, w = size
        return torch.zeros((batch, self.hidden, h, w)).to(self.conv.weight.device)


class TrajGRU(nn.Module):
    def __init__(self, inChannel, hiddenList, kernelList, return_all=False, timeLen=None, useLast=True, shape=None, enConvList=None, forConvList=None):
        super(TrajGRU, self).__init__()
        self.nlayer = len(hiddenList)
        self.return_all = return_all
        TrajGRUList = []
        if forConvList:
            for i in range(self.nlayer):
                TrajGRUList.append(TrajGRUCell(inChannel, hiddenList[i], kernelList[i]))
                inChannel = forConvList[i].out_channels
        else:
            for i in range(self.nlayer):
                inChannel = enConvList[i].out_channels if enConvList else inChannel
                TrajGRUList.append(TrajGRUCell(inChannel, hiddenList[i], kernelList[i]))
                inChannel = None if enConvList else hiddenList[i]
        self.TrajGRUList = nn.ModuleList(TrajGRUList)
        self.timeLen = timeLen
        self.useLast = useLast
        self.shape = shape
        self.enConvList = enConvList
        self.forConvList = forConvList

    def forward(self, input, state=None):
        if self.forConvList:
            return self.Decoder(input, state)
        else:
            return self.Encoder(input, state)

    def Encoder(self, input, state=None):
        b, t, c, hh, ww = input.shape
        if state is None:
            state = []
            for i in range(self.nlayer):
                hh, ww = hh//self.enConvList[i].stride[0], ww//self.enConvList[i].stride[1]
                state.append(self.TrajGRUList[i].init_state(b, (hh, ww)))
        outputList = []
        for l in range(self.nlayer):
            h = state[l]
            output = []
            for tt in range(t):
                inp = self.enConvList[l](input[:,tt]) if self.enConvList else input[:,tt]
                h = self.TrajGRUList[l](inp, h)
                output.append(h)
            input = torch.stack(output, dim=1)
            outputList.append(input)
            state[l] = h
        if not self.return_all:
            outputList = outputList[-1:]
        return outputList, state

    def Decoder(self, input=None, state=None):
        b, c, hh, ww = self.shape
        if not input:
            input = []
            for i in range(self.timeLen):
                input.append(torch.zeros(b, c, hh, ww).to(self.TrajGRUList[0].conv.weight.device))
        if state is None:
            state = []
            h0, w0 = hh, ww
            print('State == None is not recommend for decoding')
            for i in range(self.nlayer):
                state.append(self.TrajGRUList[i].init_state(b, (h0, w0)))
                h0, w0 = h0*self.forConvList[i].stride, w0*self.forConvList[i].stride

        t = self.timeLen
        outputList = []
        for tt in range(t):
            inp = input[tt]
            for l in range(self.nlayer):
                h = state[l]
                h = self.TrajGRUList[l](inp, h)
                state[l] = h
                inp = self.forConvList[l](h) if self.forConvList else h
            outputList.append(inp)
        outputList = torch.stack(outputList, dim=1)
        return outputList, h


if __name__ == '__main__':
    device = torch.device('cuda:6')
    convList = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1).float().to(device),
                              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2).float().to(device),
                              nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=2).float().to(device)])
    forList = nn.ModuleList([nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=4, padding=1, stride=2).float().to(device),
                             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2).float().to(device),
                             nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1).float().to(device)])
    model1 = TrajGRU(1, [32, 64, 192], [3, 3, 3], enConvList=convList).float().to(device)
    model2 = TrajGRU(192, [192, 64, 32], [3, 3, 3], forConvList=forList, shape=(1, 192, 16, 16), timeLen=3).float().to(device)
    input = torch.randn((1, 6, 1, 64, 64)).float().to(device)
    print(input.shape)
    output = model1(input)
    print(output[0][0].shape, output[1][0].shape, output[1][1].shape)
    # output = model2(input=None, state=output[1][::-1])
    output = model2(input=None, state=[item.to(device) for item in output[1][::-1]])
    print(output[0].shape)
