from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from config import *

class MyDataSet(Dataset):
    def __init__(self, filename, labels, resize):
        self.filename = filename
        self.labels = labels
        self.resize = resize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.filename[item]
        transform = transforms.Compose([
            lambda x:Image.open(image).convert('RGB'),
            transforms.Resize(int(self.resize*1.25)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor()
        ])
        image = transform(image)
        return image, torch.tensor(self.labels[item])


def split(dir, ratio, batch_size, resize):
    dataset = ImageFolder(dir)
    # print(dataset.class_to_idx)
    # print(dataset.classes) # 所有文件夹的名字
    # print(dataset.samples) # 文件的路径及对应的标签

    img_to_label = [[] for _ in range(len(dataset.classes))] # 标签及对应的图片路径
    for path, lab in dataset.samples:
        img_to_label[lab].append(path)

    train_x, val_x, test_x = [], [], []
    train_y, val_y, test_y = [], [], []
    for i, db in enumerate(img_to_label):
        length = len(db)
        train_x.extend(db[:int(ratio[0]*length)])
        train_y.extend([i for _ in range(0, int(ratio[0]*length))])

        val_x.extend(db[int(ratio[0]*length):int(int((ratio[0]+ratio[1])*length))])
        val_y.extend([i for _ in range(int(ratio[0]*length), int(int((ratio[0]+ratio[1])*length)))])

        test_x.extend(db[int((ratio[0]+ratio[1])*length):length])
        test_y.extend([i for _ in range(int((ratio[0]+ratio[1])*length), length)])
    train_loader = DataLoader(MyDataSet(train_x, train_y, resize), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MyDataSet(val_x, val_y, resize), batch_size=batch_size)
    test_loader = DataLoader(MyDataSet(test_x, test_y, resize), batch_size=batch_size)

    return train_loader, val_loader, test_loader

    # print(type(train_x[0]), type(train_y[0]))
    # print(len(train_x), len(train_y))
    # print(len(val_x), len(val_y))
    # print(len(test_x), len(test_y))


def main():
    dir = 'Img'
    train_loader, val_loader, test_loader = split(dir, [0.8, 0.1, 0.1], batch_size=32, resize=224)
if __name__ == '__main__':
    main()