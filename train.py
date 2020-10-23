import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset.dataloader import OwnLoader
import argparse
import torch
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
    parser.add_argument('--model', default='ConvLSTMNet', type=str,
                        choices=['ATMGRUNet', 'ConvGRUNet', 'ConvLSTMNet', 'DeformConvGRUNet', 'TrajGRUNet', 'MIMNet',
                                 'E3DLSTMNet', 'PredRNNPP', 'PredRNNNet'])
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


if __name__ == '__main__':
    args = parse()
    inLen, outLen, patchSize, batchSize, lr = args.inLen, args.outLen, args.patchSize, args.batchSize, args.lr
    comment = '%s_24-12-6-3-2-1-in_%s-out_lr-%s' % (args.model, outLen, lr)

    device = torch.device('cuda:%s' % args.gpuID)
    writer = SummaryWriter(comment=comment)
    print('Use GPU:%s' % device)

    trainLoader = DataLoader(OwnLoader(phase='train'), batch_size=batchSize, shuffle=True)
    validLoader = DataLoader(OwnLoader(phase='valid'), batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(OwnLoader(phase='test'), batch_size=batchSize, shuffle=True)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log = get_logger('./runs/%s.log' % current_time + '_' + comment)
    solver = Solver(model=args.model, device=device, loader=[trainLoader, validLoader, testLoader], log=[log, writer],
                    lr=lr, comment=comment)

    # print('===>Load checkpoint {}'.format(cfg.modelPath))
    # checkpoint = torch.load(cfg.modelPath, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'], strict=False)

    baseline = 999999

    for epoch in range(15):
        solver.adjust_learning_rate(epoch, lr)
        trainLoss = solver.train(epoch)
        print("===>Epoch {}, Average Loss: {:.4f}".format(epoch, trainLoss / len(trainLoader)))
        validLoss = solver.valid(epoch)
        print("===>Epoch {}, Average Loss: {:.4f}".format(epoch, validLoss / len(validLoader)))
