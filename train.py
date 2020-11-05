import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset.dataloader import OwnLoader, RadarLoader
import argparse
import torch
import torch.nn as nn
from model import Solver
import logging
import os
from datetime import datetime


def parse():
    parser = argparse.ArgumentParser(description='It is interesting.')
    parser.add_argument('--gpuID', default=0, type=int)
    parser.add_argument('--inLen', default=6, type=int)
    parser.add_argument('--outLen', default=6, type=int)
    parser.add_argument('--patchSize', default=64, type=int)
    parser.add_argument('--batchSize', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--type', default='Radar', type=str)
    parser.add_argument('--testOnly', action='store_true')
    parser.add_argument('--maskType', default='Mix', type=str, choices=['MaskOnly', 'WithoutMask', 'Mix'])
    parser.add_argument('--model', default='ConvLSTMNet', type=str,
                        choices=['ATMGRUNet', 'ConvGRUNet', 'ConvLSTMNet', 'DeformConvGRUNet', 'TrajGRUNet', 'MIMNet',
                                 'E3DLSTMNet', 'PredRNNPP', 'PredRNNNet', 'ABModel'])
    parser.add_argument('--abChoice', default='', type=str)
    args = parser.parse_args()
    return args


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=1e-3):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(self, input, target):
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum((input - target) ** 2, (2, 3, 4))
        mae = torch.sum(torch.abs((input - target)), (2, 3, 4))

        mse = torch.sum(torch.mean(mse, dim=1))
        mae = torch.sum(torch.mean(mae, dim=1))
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight * mse + self.mae_weight * mae)

if __name__ == '__main__':
    args = parse()
    device = torch.device('cuda:%s' % args.gpuID)
    inLen, outLen, patchSize, batchSize, lr = args.inLen, args.outLen, args.patchSize, args.batchSize, args.lr
    comment = '%s%s%s%s_%s-in_%s-out_lr-%s' % (args.type, args.maskType, args.model, args.abChoice, inLen, outLen, lr)
    args.device, args.comment, args.criterion = device, comment, Weighted_mse_mae()

    writer = None #SummaryWriter(comment=comment, logdir=args.type+args.maskType) if not args.testOnly else None
    print('Use GPU:%s' % device)

    # trainLoader = DataLoader(OwnLoader(type=args.type, phase='train'), batch_size=batchSize, shuffle=True)
    # validLoader = DataLoader(OwnLoader(type=args.type, phase='valid'), batch_size=batchSize, shuffle=True)
    # testLoader = DataLoader(OwnLoader(type=args.type, phase='test'), batch_size=batchSize, shuffle=True)
    trainLoader = DataLoader(RadarLoader(mode='train'), batch_size=batchSize, shuffle=True)
    validLoader = DataLoader(RadarLoader(mode='valid'), batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(RadarLoader(mode='test'), batch_size=batchSize, shuffle=True)
    args.loader = [trainLoader, validLoader, testLoader]

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log = get_logger('./runs%s/%s_%s.log' % (args.type+args.maskType, 'Train' if not args.testOnly else 'Test', current_time + '_' + comment))
    args.log = [log, writer]
    solver = Solver(args)

    # print('===>Load checkpoint {}'.format(cfg.modelPath))
    # checkpoint = torch.load(cfg.modelPath, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'], strict=False)

    baseline = 999999

    for epoch in range(15):
        solver.scheduler.step(epoch)
        if not args.testOnly:
            trainLoss = solver.train(epoch)
            print("===>Epoch {}, Average Loss: {:.4f}".format(epoch, trainLoss / len(trainLoader)))
            validLoss = solver.valid(epoch)
            print("===>Epoch {}, Average Loss: {:.4f}".format(epoch, validLoss / len(validLoader)))
        testLoss = solver.test(epoch)
        print("===>Epoch {}, Average Loss: {:.4f}".format(epoch, testLoss / len(validLoader)))
