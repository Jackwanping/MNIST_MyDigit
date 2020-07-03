import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torchvision import transforms
from utils  import Flatten
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from resnet import *
lr = 1e-3

class PredictDataset(Dataset):
    def __init__(self, root):
        super(PredictDataset, self).__init__()
        self.root = root
        self.images = self.load(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        transform = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        img = transform(img)
        return img
    def load(self, root):
        images = []
        for name in sorted(os.listdir(root)):
            images.append(os.path.join(root, name))
        return images

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


test = PredictDataset("Predict")
test_loader = DataLoader(test, batch_size=1, num_workers=2)

model = ResNet18(62)
model.load_state_dict(torch.load("save_model/best.ckpt"))
model.eval()

for img in test_loader:
    img = img.to(device)
    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(dim=1)
    print(pred[0].detach().float().cpu().numpy())





# sample = next(iter(test_loader))
# print(sample.shape)

# train_model = torchvision.models.resnet18(pretrained=True)

# model = nn.Sequential(*list(train_model.children())[:-1],
#                           Flatten(),
#                           nn.Linear(512, 62)).to(device)
    