import os
import random
import numpy as np
import skimage.io as io
from torch.utils.data import Dataset, DataLoader


class OwnLoader(Dataset):
    def __init__(self, type='Radar', phase='train', per=0.6):
        super(OwnLoader, self).__init__()
        self.type = type
        self.basePath = '/home/tsingzao/Dataset/OwnData/%sSeg/'%type
        folderList = os.listdir(self.basePath)
        random.seed(20201021)
        random.shuffle(folderList)
        trainList = folderList[:int(len(folderList)*per)]
        validList = folderList[int(len(folderList)*per):int(len(folderList)*(per+(1-per)*0.5))]
        testList = folderList[int(len(folderList)*(per+(1-per)*0.5)):]
        self.fileList = {'train':trainList, 'valid':validList, 'test':testList}[phase]

    def __getitem__(self, item):
        folder = self.fileList[item]
        filePath = [os.path.join(self.basePath, folder, path) for path in os.listdir(os.path.join(self.basePath, folder))]
        filePath.sort()
        imageList = []
        for file in filePath:
            if self.type == 'Radar':
                img = 1-io.imread(file, as_gray=True)
            else:
                img = io.imread(file)/255.0
            imageList.append(img)
        return np.expand_dims(np.array(imageList), axis=1)

    def __len__(self):
        return len(self.fileList)


if __name__ == '__main__':
    loader = iter(DataLoader(OwnLoader(type='Radar'), shuffle=True))
    img = next(loader)
    print(img.shape)
    import matplotlib.pyplot as plt
    timeLen = img.shape[1]
    for i in range(timeLen):
        plt.subplot(1, timeLen, i+1)
        plt.imshow(img.numpy()[0,i,0], vmax=1, vmin=0)
    plt.show()