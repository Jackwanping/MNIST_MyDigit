import torch
from torch import nn
from torch import optim
import torchvision
from newdata import MyDataSet, split
# import visdom
from utils  import Flatten

batch_size = BATCH_SIZE
lr = LEARNING_RATE
epochs = TOTAL_EPOCH
device = DEVICE
torch.manual_seed(1234)
train_loader, val_loader, test_loader = split('Img', [0.7, 0.2, 0.1], batch_size=batch_size, resize=224)

print("dataset loaded!")

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

def main():
    train_model = torchvision.models.resnet34(pretrained=True)
    ffc = train_model.fc.in_features
    model = nn.Sequential(*list(train_model.children())[:-1],
                          Flatten(),
                          nn.Linear(ffc, 62)).to(device)
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

        print("epoch :{}, train loss:{}".format(epoch, loss.item()))
        # viz.line([loss.item()], [epoch], win='train-loss', update='append')

        if epoch % 3 == 0:
            val_acc = evalue(model, val_loader)

            print("epoch:{}, val acc:{}".format(epoch, val_acc))
            # viz.line([val_acc], [epoch], win='val-acc', update='append')
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

        torch.save(model.state_dict(), 'best.ckpt')

        print('epoch:',epoch,'best-acc:', best_acc, 'best-epoch:', best_epoch)

    model.load_state_dict(torch.load('best.ckpt'))
    print('loaded from best_gray model')
    test_acc = evalue(model, test_loader)
    print('test-acc', test_acc)

if __name__ == '__main__':
    main()