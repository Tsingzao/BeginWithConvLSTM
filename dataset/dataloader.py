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

class RadarLoader(Dataset):
    def __init__(self, mode='train'):
        super(RadarLoader, self).__init__()
        with open('./dataset/radarList', 'r') as fp:
            lines = fp.readlines()
        if mode == 'train':
            self.fileList = lines[:8000]
        elif mode == 'valid':
            self.fileList = lines[8000:9000]
        else:
            self.fileList = lines[9000:]

    def __getitem__(self, index):
        feature = []
        fileIdx = self.fileList[index].strip()
        pathFormat = '/home/tsingzao/Dataset/Radar/train/%s_%03d.png'
        for i in range(21):
            path = pathFormat % (fileIdx, i)
            image = io.imread(path)
            image[image == 255] = 0
            image = image.astype('float')
            image /= 80
            image[image > 1] = 1
            feature.append(image[::4,::4])
        data = np.expand_dims(np.asarray(feature), axis=1)
        return data

    def __len__(self):
        return len(self.fileList)

if __name__ == '__main__':
    # loader = iter(DataLoader(OwnLoader(type='Temperature'), shuffle=True))
    loader = iter(DataLoader(RadarLoader(), shuffle=True))
    img = next(loader)
    print(img.shape)
    import matplotlib.pyplot as plt
    timeLen = img.shape[1]
    for i in range(timeLen):
        plt.subplot(1, timeLen, i+1)
        plt.imshow(img.numpy()[0,i,0], vmax=1, vmin=0)
    plt.show()