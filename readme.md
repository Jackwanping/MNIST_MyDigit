train.py 训练，测试，验证的

dataloader.py 加载数据集

resnet.py 手写的ResNet18

utils.py 一些工具类

Img是存放手写数字和字母的文件夹，可以在[这里](https://drive.google.com/open?id=1cjk8yk21-__-wz3FyVVeQpcqft6qmP39)下载

## 运行步骤：

#### 1、下载EnglishHnd数据集，并改名为`Img`，在`config.py`中调整配置项

#### 2、训练

如果用ResNet模型,运行

```sh
python train_without_visdom.py
```

如果用EfficientNet模型，运行

```sh
python train_efficient.py
```





