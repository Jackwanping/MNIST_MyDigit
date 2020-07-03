import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary

# import visdom

from utils  import Flatten
from dataloader import MyDataSet
from resnet import *

batch_size = 128
lr = 1e-3
epochs = 40
device = torch.device('cuda:0')
torch.manual_seed(1234)


train_db = MyDataSet('Img', 224, 'train')
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=4)

val_db = MyDataSet('Img', 224, 'val')
val_loader = DataLoader(val_db, batch_size=batch_size, num_workers=2)

test_db = MyDataSet('Img', 224, 'test')
test_loader = DataLoader(test_db, batch_size=batch_size,num_workers=2)

def evalue(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

if __name__ == '__main__':
    # train_model = torchvision.models.resnet18(pretrained=True)
    # model = nn.Sequential(*list(train_model.children())[:-1],
    #                       Flatten(),
    #                       nn.Linear(512, 62)).to(device)
    model = ResNet18(62).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss().to(device)

    # viz = visdom.Visdom()
    best_acc, best_epoch = 0, 0
    # viz.line([0], [-1], win='train-loss', opts=dict(title='train-loss'))
    # viz.line([0], [-1], win='val-acc', opts=dict(title='val-acc'))

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            model.train()
            x ,y = x.to(device), y.to(device)
            logits = model(x)

            loss = criteon(logits, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        print("train loss:{}".format(loss.item()))
        # viz.line([loss.item()], [epoch], win='train-loss', update='append')

        if epoch % 3 == 0:
            val_acc = evalue(model, val_loader)
            print("test acc:{}".format(val_acc))
            # viz.line([val_acc], [epoch], win='val-acc', update='append')
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

        torch.save(model.state_dict(), 'save_model/best.ckpt')

        print('best-acc', best_acc, 'best-epoch', best_epoch)

    model.load_state_dict(torch.load('save_model/best.ckpt'))
    print('loaded from best_gray model')


    test_acc = evalue(model, test_loader)
    print('test-acc', test_acc)