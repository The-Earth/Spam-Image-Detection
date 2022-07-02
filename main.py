import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import PILToTensor
from tqdm import tqdm
import os
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
device = 'cuda'


class Images(Dataset):
    def __init__(self, path: str, device: str = 'cpu'):
        self.images = []
        self.labels = []
        self.device = device
        self.pil2tensor = PILToTensor()
        for file in os.listdir(f'{path}/negative/'):
            self.images.append(Image.open(f'{path}/negative/{file}'))
            self.labels.append(0.)
        for file in os.listdir(f'{path}/positive/'):
            self.images.append(Image.open(f'{path}/positive/{file}'))
            self.labels.append(1.)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        # 144, 256
        if image.size[0] > image.size[1]:
            image = image.rotate(90)
        image = image.resize((180, 320))

        image_tensor = self.pil2tensor(image) / 255.
        label_tensor = torch.tensor([label])
        return image_tensor.to(self.device), label_tensor.to(self.device)

    def __len__(self):
        return len(self.images)


class Net(nn.Module):
    def __init__(self):
        """
        Input: (180, 320)
        """
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 18, 3)
        self.conv4 = nn.Conv2d(18, 24, 5)

        self.conv5 = nn.Conv2d(3, 6, 11)
        self.conv6 = nn.Conv2d(6, 12, 9)
        self.conv7 = nn.Conv2d(12, 18, 9)
        self.conv8 = nn.Conv2d(18, 24, 9)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(4128, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = torch.relu(self.pool(self.conv1(x)))
        x1 = torch.relu(self.pool(self.conv2(x1)))
        x1 = torch.relu(self.pool(self.conv3(x1)))
        x1 = torch.relu(self.pool(self.conv4(x1)))
        x1 = torch.flatten(x1, start_dim=1)

        x2 = torch.relu(self.pool(self.conv5(x)))
        x2 = torch.relu(self.pool(self.conv6(x2)))
        x2 = torch.relu(self.pool(self.conv7(x2)))
        x2 = torch.relu(self.pool(self.conv8(x2)))
        x2 = torch.flatten(x2, start_dim=1)

        x = torch.cat((x1, x2), 1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


def model_train(train_set, test_set, epochs, learning_rate, batch_size, test_while_train=True, verbose=False):
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    bce = nn.BCELoss().to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(1, epochs + 1)):
        net.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_pred = net(batch_x)
            batch_loss = bce(batch_pred, batch_y) # + 0.00001 * L1(net)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss * len(batch_x)

        epoch_loss /= len(train_set)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        if verbose:
            print(f'Epoch {epoch} Loss/Train {epoch_loss}')
        if test_while_train:
            net.eval()
            train_acc, _ = model_test(train_set, net)
            test_acc, test_loss = model_test(test_set, net)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('F1 score/train', train_acc, epoch)
            writer.add_scalar('F1 score/test', test_acc, epoch)
            if verbose:
                print(f'Epoch {epoch} Loss/test {test_loss}')
                print(f'Epoch {epoch} Accuracy/train {train_acc}')
                print(f'Epoch {epoch} Accuracy/test {test_acc}')

    return net


def model_test(test_set: Dataset, net: nn.Module):
    net.eval()
    test_x, test_y = next(iter(DataLoader(test_set, batch_size=20, shuffle=True)))
    bce = nn.BCELoss().to(device)

    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        test_pred = net(test_x)
        loss = bce(test_pred, test_y)

        for i, item in enumerate(test_pred):
            if item > 0.5 and test_y[i] > 0.5:
                tp += 1
            elif item > 0.5 and test_y[i] < 0.5:
                fp += 1
            elif item < 0.5 and test_y[i] > 0.5:
                fn += 1

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0, loss
    else:
        return 2 * precision * recall / (precision + recall), loss


def L1(net: nn.Module):
    s = 0
    for param in net.parameters():
        s += torch.sum(torch.abs(param))
    return s


if __name__ == '__main__':
    lr = 0.000015
    batch = 5

    writer = SummaryWriter(comment=f'L2_lr_{lr}_batch_{batch}')
    dataset = Images('data', device)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    model = model_train(train, test, epochs=250, batch_size=batch, learning_rate=lr)

    torch.save(model.state_dict(), f'saved_models/lr_{lr}_batch_{batch}.pt')
