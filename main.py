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

        image_tensor = self.pil2tensor(image) / 255.
        label_tensor = torch.tensor([label])

        # 64x64 sub-tensor
        sub_list = []

        for i in range(25):
            h_start = torch.randint(0, image_tensor.shape[1] - 64, (1,))
            w_start = torch.randint(0, image_tensor.shape[2] - 64, (1,))
            sub_list.append(image_tensor[:, h_start:h_start + 64, w_start:w_start + 64])

        sub_tensors = torch.cat(sub_list, dim=1)
        return sub_tensors.to(self.device), label_tensor.to(self.device)

    def __len__(self):
        return len(self.images)


class Net(nn.Module):
    def __init__(self):
        """
        Input: (1600, 64)
        """
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 18, 3)
        self.conv4 = nn.Conv2d(18, 24, 3)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.ModuleList([nn.Linear(384, 128) for _ in range(25)])
        self.fc2 = nn.ModuleList([nn.Linear(128, 16) for _ in range(25)])
        self.fc3 = nn.ModuleList([nn.Linear(16, 1) for _ in range(25)])

        self.fc4 = nn.Linear(25, 12)
        self.fc5 = nn.Linear(12, 1)

    def forward(self, x):
        sub_tensors = [x[:, :, i * 64:((i + 1) * 64)] for i in range(25)]
        sub_output = []

        for i, item in enumerate(sub_tensors):
            x1 = torch.relu(self.pool(self.conv1(item)))
            x1 = torch.relu(self.pool(self.conv2(x1)))
            x1 = torch.relu(self.pool(self.conv3(x1)))
            x1 = torch.relu(self.conv4(x1))
            x1 = torch.flatten(x1, start_dim=1)

            x1 = torch.relu(self.fc1[i](x1))
            x1 = torch.relu(self.fc2[i](x1))
            x1 = torch.relu(self.fc3[i](x1))

            sub_output.append(x1)

        x = torch.cat(sub_output, dim=1)
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))

        return x


def model_train(train_set, test_set, epochs, learning_rate, batch_size, test_while_train=True, verbose=False):
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    bce = nn.BCELoss().to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(1, epochs + 1)):
        net.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_pred = net(batch_x)
            batch_loss = bce(batch_pred, batch_y)  # + 0.000001 * L1(net)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss * len(batch_x)

        epoch_loss /= len(train_set)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        train_acc, _ = model_test(train_set, net)
        writer.add_scalar('F1 score/train', train_acc, epoch)

        if verbose:
            print(f'Epoch {epoch} Loss/Train {epoch_loss}')
            print(f'Epoch {epoch} F1 score/train {train_acc}')

        if test_while_train:
            net.eval()
            test_acc, test_loss = model_test(test_set, net)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('F1 score/test', test_acc, epoch)
            if verbose:
                print(f'Epoch {epoch} Loss/test {test_loss}')
                print(f'Epoch {epoch} F1/test {test_acc}')

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
    lr = 0.00005
    batch = 50

    writer = SummaryWriter(comment=f'lr_{lr}_batch_{batch}_64x64')
    dataset = Images('data', device)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    model = model_train(train, test, epochs=1000, batch_size=batch, learning_rate=lr, test_while_train=True)

    torch.save(model.state_dict(), f'saved_models/lr_{lr}_batch_{batch}_64x64.pt')
