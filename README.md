train.py 训练，测试，验证的

dataloader.py 加载数据集

resnet.py 手写的ResNet18

utils.py 一些工具类



### 运行环境：

pytorch 1.x

NVIDIA GPU

### 依赖

visdom

pillow

### 运行方式

命令行进入项目根目录，输入`python -m visdom.server`（前提已经安装visdom），浏览器输入url`http://localhost:8097/`可以看到visdom界面，再用命令行或者其他方式运行`train.py`文件



### 实现思路

1、自定义数据集

`Img`目录下的 `Sample001-Sample062` 对应数字0到小写字母z，然后通过 `glob`和 `os`库找到图片的路径以及对应的标签，并且存储到csv文件中，例如 `H:\Project\MNIST_MyDigit\remote\MNIST_MyDigit\Img\Sample001\img001-001.png, 0`

注意实现自定义数据集类时需要继承 `torch.utils.data.Dataset`类并且覆写 `__len__`与`__getitem__`方法，可以在`__getitem__`方法中使用`transforms`根据路径打开图片，调整大小，旋转，裁剪，转化为tensor类型



2、模型搭建

可以调用`torchvision.models`里预训练好的模型，也可以自己手写，本项目调用预训练好的Resnet18，取得除最后一层外的所有层，打平后送到全连接层



3、训练测试

batch_size设置为128，学习率设为0.001，训练40个epoch，每3个epoch在验证集上验证，同时记录最好的效果，并且保存参数。训练结束后，加载在验证集上表现最好的模型，并且在测试集上进行测试。





