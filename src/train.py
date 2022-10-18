import torch.utils.data
import torchvision.datasets
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from LeNet.src.model import LeNet

device = torch.device("cuda:0")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# 数据集
# 准备训练集
# 第一次把download设为True
train_set = torchvision.datasets.CIFAR10('../data',train=True,
              download=False,transform=transform)
#准备测试集
test_set = torchvision.datasets.CIFAR10('../data',train=False,
              download=False,transform=transform)
test_size = len(test_set)
# dataloader加载
train_loader = torch.utils.data.DataLoader(train_set,64,shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,10000,shuffle=True)

test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

# 导入分类标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 数据集图片展示
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))

# 定义网络
lenet = LeNet()
lenet.to(device)

train_step = 0
val_step = 0
# 定义损失函数
loss_function = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(lenet.parameters(), lr=0.001)

# 引入tensorboard
writer = SummaryWriter("../logs")
for epoch in range(10):
    lenet.train()
    running_loss = 0.0

    for step, data in enumerate(train_loader,start=0):
        images, labels = data
        # 梯度置为零
        optimizer.zero_grad()
        outputs = lenet(images.to(device))

        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        train_step += 1
        running_loss += loss.item()

        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        writer.add_scalar("train_loss",loss.item(),train_step)

        # lenet.eval()
        # if step % 500 == 499:
        #     with torch.no_grad():
        #         outputs = lenet(test_image.to(device))
        #         predict = torch.max(outputs, dim=1)[1]
        #         acc = (predict == test_label.to(device)).sum().item() / test_size
        #
        #         # print("\n[epoch {}]  train_loss: {}  test_accuracy: {}".
        #         #       format(epoch + 1, running_loss / step, acc / test_size))
        #         print('\r[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
        #               (epoch + 1, step + 1, running_loss / 500, acc))
        #         # val_step += 1
        #         running_loss = 0

    lenet.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in test_loader:
            test_images, test_labels = data_test
            outputs = lenet(test_images.to(device))

            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()

        # 得到测试集准确率
        accurate_test = acc / test_size


        print("[epoch {}]  train_loss: {}  test_accuracy: {}".
              format(epoch + 1, running_loss / step, acc / test_size))
        val_step += 1
        writer.add_scalar("val_acc", accurate_test, val_step)

        # if accurate_test > best_acc:
        #     best_acc = accurate_test
        #     torch.save()
writer.close()
print('Finished Training')