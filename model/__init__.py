from model.atmconvgru import ATMGRUNet
from model.convgru import ConvGRUNet
from model.convlstm import ConvLSTMNet
from model.DeformGRU import DeformConvGRUNet
from model.e3d import E3DLSTMNet
from model.MIM import MIMNet
from model.trajGRU import TrajGRUNet
from model.predrnnpp import PredRNNPP
from model.predrnn import PredRNNNet
from model.ablationGRU import ConvGRUNet as ABModel
import os
import torch
import shutil
import torch.optim as optim
from torch.autograd import Variable


class Solver(object):
    # def __init__(self, model='ConvLSTMNet', device=None, loader=None, log=None, lr=None, comment=None):
    def __init__(self, args):
        if args.model == 'ConvLSTMNet':
            print('Build model ConvLSTMNet...')
            self.model = ConvLSTMNet()
        elif args.model == 'ConvGRUNet':
            print('Build model ConvGRUNet...')
            self.model = ConvGRUNet()
        elif args.model == 'DeformConvGRUNet':
            print('Build model DeformConvGRUNet...')
            self.model = DeformConvGRUNet()
        elif args.model == 'TrajGRUNet':
            print('Build model TrajGRUNet...')
            self.model = TrajGRUNet()
        elif args.model == 'PredRNNNet':
            print('Build model PredRNNNet...')
            self.model = PredRNNNet()
        elif args.model == 'PredRNNPP':
            print('Build model PredRNNPP...')
            self.model = PredRNNPP()
        elif args.model == 'E3DLSTMNet':
            print('Build model E3DLSTMNet...')
            self.model = E3DLSTMNet()
        elif args.model == 'MIMNet':
            print('Build model MIMNet...')
            self.model = MIMNet()
        elif args.model == 'ATMGRUNet':
            print('Build model ATMGRUNet...')
            self.model = ATMGRUNet()
        elif args.model == 'ABModel':
            print('Build model ABModel...')
            self.model = ABModel(outLen=args.outLen)
        else:
            Exception()

        loader, log = args.loader, args.log
        self.model = self.model.float().to(args.device)
        self.trainLoader, self.validLoader, self.testLoader = loader[0], loader[1], loader[2]
        self.log, self.writer = log[0], log[1]
        self.device = args.device
        self.epoch = 0
        self.comment = args.comment
        self.inLen = args.inLen
        self.outLen = args.outLen
        self.maskType = args.maskType

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = args.criterion.to(args.device)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5,10], gamma=0.5)

    def train(self, epoch):
        loader = self.trainLoader
        self.log.info('Start train epoch %s...' % epoch)
        print('Start train...')
        self.model.train()
        trainLoss, maskLoss = 0, 0
        for batch, data in enumerate(loader):
            # forward pass
            input, label = Variable(data[:, :self.inLen]).float().to(self.device), Variable(
                data[:, self.inLen:self.inLen + self.outLen]).float().to(self.device)
            # input, label = input.permute((1, 0, 2, 3, 4)), label.permute((1, 0, 2, 3, 4))
            # mask = torch.zeros_like(input)
            output = self.model(input)  # , mask)
            # label = label.permute((1, 0, 2, 3, 4))

            if self.maskType.upper() == 'MASKONLY':
                mask = label.ge(0.01)
                loss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                trainLoss += loss.item()
                printStr = 'MaskIterLoss={:.4f}\t MaskAvgLoss={:.4f}'.format(loss, trainLoss / (batch + 1))
            elif self.maskType.upper() == 'MIX':
                mask = label.ge(0.01)
                maskedloss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                totalloss = self.criterion(output, label)
                loss = maskedloss + totalloss * 5
                maskLoss += maskedloss.item()
                trainLoss += totalloss.item() * 5
                printStr = 'MaskIterLoss={:.4f}\t MaskAvgLoss={:.4f} IterLoss*5={:.4f}\t AvgLoss={:.4f}\t TotalIterLoss*5={:.4f}'.format(
                    maskedloss, maskLoss / (batch + 1), totalloss*5, trainLoss / (batch + 1), loss)
            elif self.maskType.upper() == 'WITHOUTMASK':
                totalloss = self.criterion(output, label)
                loss = totalloss #* 5
                trainLoss += totalloss.item() #* 5
                printStr = 'IterLoss={:.4f}\t AvgLoss={:.4f}'.format(loss, trainLoss / (batch + 1))
            else:
                raise NotImplementedError

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record
            self.log.info('Epoch:[{}/{}]\t Iteration:[{}/{}]\t {}'.format(epoch, 15, batch, len(loader), printStr))

        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        self.saveCheckpoint(epoch, state)
        return trainLoss / len(loader)

    def valid(self, epoch):
        loader = self.validLoader
        self.log.info('Start valid epoch %s...' % epoch)
        print('Start valid...')
        self.model.eval()
        validLoss, maskLoss = 0, 0
        with torch.no_grad():
            for batch, data in enumerate(loader):
                # forward pass
                input, label = Variable(data[:, :self.inLen]).float().to(self.device), Variable(
                    data[:, self.inLen:self.inLen + self.outLen]).float().to(self.device)
                # input, label = input.permute((1, 0, 2, 3, 4)), label.permute((1, 0, 2, 3, 4))
                # mask = torch.zeros_like(input)
                output = self.model(input)  # , mask)
                # label = label.permute((1, 0, 2, 3, 4))

                if self.maskType.upper() == 'MASKONLY':
                    mask = label.ge(0.01)
                    loss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                    validLoss += loss.item()
                    printStr = 'MaskIterLoss={:.4f}\t MaskAvgLoss={:.4f}'.format(loss, validLoss / (batch + 1))
                elif self.maskType.upper() == 'MIX':
                    mask = label.ge(0.01)
                    maskedloss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                    totalloss = self.criterion(output, label)
                    loss = maskedloss + totalloss * 5
                    maskLoss += maskedloss.item()
                    validLoss += totalloss.item() * 5
                    printStr = 'MaskIterLoss={:.4f}\t MaskAvgLoss={:.4f} IterLoss*5={:.4f}\t AvgLoss={:.4f}\t TotalIterLoss*5={:.4f}'.format(
                        maskedloss, maskLoss / (batch + 1), totalloss * 5, validLoss / (batch + 1), loss)
                elif self.maskType.upper() == 'WITHOUTMASK':
                    totalloss = self.criterion(output, label)
                    loss = totalloss * 5
                    validLoss += totalloss.item() * 5
                    printStr = 'IterLoss={:.4f}\t AvgLoss={:.4f}'.format(loss, validLoss / (batch + 1))
                else:
                    raise NotImplementedError

                # record
                self.log.info('Epoch:[{}/{}]\t Iteration:[{}/{}]\t {}'.format(epoch, 15, batch, len(loader), printStr))

        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        self.saveCheckpoint(epoch, state, isBest=True)
        return validLoss / len(loader)

    def test(self, epoch):
        loader = self.testLoader
        self.log.info('Start testing %s...' % epoch)
        modelPath = os.path.join('./checkpoint_%s' % self.comment, 'checkpoint_%03d.pth.tar' % (epoch))
        checkpoint = torch.load(modelPath, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Start test...')
        self.model.eval()
        maskLoss, testLoss = 0, 0
        with torch.no_grad():
            for batch, data in enumerate(loader):
                # forward pass
                input, label = Variable(data[:, :self.inLen]).float().to(self.device), Variable(
                    data[:, self.inLen:self.inLen + self.outLen]).float().to(self.device)
                # input, label = input.permute((1, 0, 2, 3, 4)), label.permute((1, 0, 2, 3, 4))
                # mask = torch.zeros_like(input)
                output = self.model(input)
                # output = self.model(input, mask)
                # label = label.permute((1, 0, 2, 3, 4))

                if self.maskType.upper() == 'MASKONLY':
                    mask = label.ge(0.01)
                    loss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                    testLoss += loss.item()
                    printStr = 'MaskIterLoss={:.4f}\t MaskAvgLoss={:.4f}'.format(loss, testLoss / (batch + 1))
                elif self.maskType.upper() == 'MIX':
                    mask = label.ge(0.01)
                    maskedloss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                    totalloss = self.criterion(output, label)
                    loss = maskedloss + totalloss * 5
                    maskLoss += maskedloss.item()
                    testLoss += totalloss.item() * 5
                    printStr = 'MaskIterLoss={:.4f}\t MaskAvgLoss={:.4f} IterLoss*5={:.4f}\t AvgLoss={:.4f}\t TotalIterLoss*5={:.4f}'.format(
                        maskedloss, maskLoss / (batch + 1), totalloss * 5, testLoss / (batch + 1), loss)
                elif self.maskType.upper() == 'WITHOUTMASK':
                    totalloss = self.criterion(output, label)
                    loss = totalloss * 5
                    testLoss += totalloss.item() * 5
                    printStr = 'IterLoss={:.4f}\t AvgLoss={:.4f}'.format(loss, testLoss / (batch + 1))
                else:
                    raise NotImplementedError

                # record
                self.log.info('Epoch:[{}/{}]\t Iteration:[{}/{}]\t {}'.format(epoch, 15, batch, len(loader), printStr))
        return testLoss / len(loader)

    def saveCheckpoint(self, epoch, state, isBest=False):
        cpkRoot = './checkpoint_%s' % self.comment
        fileName = 'checkpoint_%03d.pth.tar' % (epoch)
        os.makedirs(cpkRoot, exist_ok=True)
        filePath = os.path.join(cpkRoot, fileName)
        bestPath = os.path.join(cpkRoot, 'model_best.pth.tar')
        if isBest:
            shutil.copyfile(filePath, bestPath)
            return
        torch.save(state, filePath)

    def adjust_learning_rate(self, epoch, lr):
        lr = lr * ((1. - float(epoch) / 10) ** 0.9)
        self.log.info('changing learnging rate to {:.4e}'.format(lr))
        self.optimizer.param_groups[0]['lr'] = lr

    def showTensor(self, input, label, output):
        import matplotlib.pyplot as plt
        for i in range(self.outLen):
            plt.subplot(3, self.outLen, i + 1)
            plt.imshow(input[0, i, 0].data.cpu().numpy(), vmax=1, vmin=0)
            plt.subplot(3, self.outLen, i + 1 + self.outLen)
            plt.imshow(label[0, i, 0].data.cpu().numpy(), vmax=1, vmin=0)
            plt.subplot(3, self.outLen, i + 1 + self.outLen*2)
            plt.imshow(output[0, i, 0].data.cpu().numpy(), vmax=1, vmin=0)
        plt.show()
