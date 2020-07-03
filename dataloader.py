import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
# import visdom
import time

all_item = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M',' N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ,' Z',
    'a' ,'b', 'c' ,'d', 'e', 'f', 'g', 'h', 'i', 'j' ,'k', 'l', 'm','n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def to_str(index):
    return all_item[index]

class MyDataSet(Dataset):
    def __init__(self, root, resize, mode):
        super(MyDataSet, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.images, self.labels = load_csv(root, 'data.csv')

        length = len(self.images)
        if mode == 'train':
            self.images = self.images[:int(0.6*length)]
            self.labels = self.labels[:int(0.6*length)]
        elif mode == 'test':
            self.images = self.images[int(0.6*length):int(0.8*length)]
            self.labels = self.labels[int(0.6*length):int(0.8*length)]
        else:
            self.images = self.images[int(0.8*length):int(length)]
            self.labels = self.labels[int(0.8*length):int(length)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img, label = self.images[item], self.labels[item]
        
        transform = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
        ])
        img = transform(img)
        label = torch.tensor(label)
        return img, label

def load_csv(root, filename):
    name_to_label = {} # dirname : label
    for name in sorted(os.listdir(os.path.join(root)), key=lambda x: int(x[-3:])):
        name_to_label[name] = len(name_to_label.keys())

    images, labels = [], []

    for name in name_to_label.keys():
        images += glob.glob(os.path.join(root, name, '*.png'))
    random.shuffle(images)

    for img in images:
        name = img.split(os.sep)[-2] # 得到所在的目录
        label = name_to_label[name] # 得到目录对应的标签
        labels.append(label)
    return images,labels

if __name__ == '__main__':
    images, labels = load_csv("Img", "data.csv")
    if len(images) != len(labels):
        assert "图片数量不等于标签数量"


    db = MyDataSet('Img', 224, 'train') # db.__getitem__(0)
    vis = visdom.Visdom()
    # x, y = next(iter(db))
    # print('sample:', x.shape, y.shape)
    # print(all_item[int(y.numpy())])

    # to_image = transforms.ToPILImage()
    # img = to_image(x)
    # img.show()

    # to_numpy = np.transpose(x.detach().float().numpy(), (1, 2, 0))
    # plt.imshow(to_numpy)
    # plt.show()

    loaders = DataLoader(db, batch_size=16, shuffle=True)
    for x, y in loaders:
        vis.images(x, nrow=4, win='batch', opts=dict(title='batch'))

        str_ = [all_item[int(i.numpy())] for i in y]
        vis.text(str_, win='label', opts=dict(title='batch-y'))
        time.sleep(10)


