from model.atmconvgru import ATMGRUNet
from model.convgru import ConvGRUNet
from model.convlstm import ConvLSTMNet
from model.DeformGRU import DeformConvGRUNet
from model.e3d import E3DLSTMNet
from model.MIM import MIMNet
from model.trajGRU import TrajGRUNet
from model.predrnnpp import PredRNNPP
from model.predrnn import PredRNNNet
from torch.autograd import Variable
import os
import torch
import shutil
import torch.optim as optim
from torch.autograd import Variable


class Solver(object):
    def __init__(self, model='ConvLSTMNet', device=None, loader=None, log=None, lr=None, comment=None):
        if model == 'ConvLSTMNet':
            print('Build model ConvLSTMNet...')
            self.model = ConvLSTMNet()
        elif model == 'ConvGRUNet':
            print('Build model ConvGRUNet...')
            self.model = ConvGRUNet()
        elif model == 'DeformConvGRUNet':
            print('Build model DeformConvGRUNet...')
            self.model = DeformConvGRUNet()
        elif model == 'TrajGRUNet':
            print('Build model TrajGRUNet...')
            self.model = TrajGRUNet()
        elif model == 'PredRNNNet':
            print('Build model PredRNNNet...')
            self.model = PredRNNNet()
        elif model == 'PredRNNPP':
            print('Build model PredRNNPP...')
            self.model = PredRNNPP()
        elif model == 'E3DLSTMNet':
            print('Build model E3DLSTMNet...')
            self.model = E3DLSTMNet()
        elif model == 'MIMNet':
            print('Build model MIMNet...')
            self.model = MIMNet()
        elif model == 'ATMGRUNet':
            print('Build model ATMGRUNet...')
            self.model = ATMGRUNet()

        self.model = self.model.float().to(device)
        self.trainLoader, self.validLoader, self.testLoader = loader[0], loader[1], loader[2]
        self.log, self.writer = log[0], log[1]
        self.device = device
        self.epoch = 0
        self.comment = comment

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.L1Loss().to(device)

    def train(self, epoch):
        loader = self.trainLoader
        self.log.info('Start train epoch %s...' % epoch)
        print('Start train...')
        self.model.train()
        trainLoss = 0
        for batch, data in enumerate(loader):
            # forward pass
            input, label = Variable(data[:,:6]).float().to(self.device), Variable(data[:,6:]).float().to(self.device)
            # input, label = input.permute((1, 0, 2, 3, 4)), label.permute((1, 0, 2, 3, 4))
            # mask = torch.zeros_like(input)
            output = self.model(input)#, mask)
            # label = label.permute((1, 0, 2, 3, 4))
            mask = label.ge(0.01)
            loss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
            trainLoss += loss.item()

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record
            self.log.info(
                'Epoch:[{}/{}]\t Iteration:[{}/{}]\t IterLoss={:.4f}\t AvgLoss={:.4f}'.format(epoch, 15, batch,
                                                                                              len(loader), loss,
                                                                                              trainLoss / (batch + 1)))
            self.writer.add_scalar('Train/TrainIterLoss_TrajGRU', loss, epoch * len(loader) + batch)
            self.writer.add_scalar('Train/TrainAvgLoss_TrajGRU', trainLoss / (batch + 1), epoch * len(loader) + batch)

        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        self.saveCheckpoint(epoch, state)
        return trainLoss / len(loader)

    def valid(self, epoch):
        loader = self.validLoader
        self.log.info('Start valid epoch %s...' % epoch)
        print('Start valid...')
        self.model.eval()
        validLoss = 0
        with torch.no_grad():
            for batch, data in enumerate(loader):
                # forward pass
                input, label = Variable(data[:,:6]).float().to(self.device), Variable(data[:,6:]).float().to(self.device)
                # input, label = input.permute((1, 0, 2, 3, 4)), label.permute((1, 0, 2, 3, 4))
                # mask = torch.zeros_like(input)
                output = self.model(input)#, mask)
                # label = label.permute((1, 0, 2, 3, 4))
                mask = label.ge(0.01)
                loss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                validLoss += loss.item()

                # record
                self.log.info(
                    'Epoch:[{}/{}]\t Iteration:[{}/{}]\t IterLoss={:.4f}\t AvgLoss={:.4f}'.format(epoch, 15, batch,
                                                                                                  len(loader), loss,
                                                                                                  validLoss / (batch + 1)))

        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        self.saveCheckpoint(epoch, state, isBest=True)
        return validLoss / len(loader)

    def test(self, epoch):
        loader = self.testLoader
        self.log.info('Start testing %s...' % epoch)
        print('Start test...')
        self.model.eval()
        testLoss = 0
        with torch.no_grad():
            for batch, data in enumerate(loader):
                # forward pass
                input, label = Variable(data[:,:6]).float().to(self.device), Variable(data[:,6:]).float().to(self.device)
                # input, label = input.permute((1, 0, 2, 3, 4)), label.permute((1, 0, 2, 3, 4))
                # mask = torch.zeros_like(input)
                output = self.model(input)#, mask)
                # label = label.permute((1, 0, 2, 3, 4))
                mask = label.ge(0.01)
                loss = self.criterion(torch.masked_select(output, mask), torch.masked_select(label, mask))
                testLoss += loss.item()

                # record
                self.log.info(
                    'Epoch:[{}/{}]\t Iteration:[{}/{}]\t IterLoss={:.4f}\t AvgLoss={:.4f}'.format(epoch, 15, batch,
                                                                                                  len(loader), loss,
                                                                                                  testLoss / (batch + 1)))
        return testLoss / len(loader)


    def saveCheckpoint(self, epoch, state, isBest=False):
        cpkRoot = './checkpoint_%s'%self.comment
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