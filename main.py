import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
from PIL import Image

# torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
device = 'cuda'


class Images(Dataset):
    def __init__(self, path: str, device: str = 'cpu'):
        self.images = []
        self.labels = []
        self.device = device
        self.pil2tensor = ToTensor()
        for file in os.listdir(f'{path}/negative/'):
            self.images.append(Image.open(f'{path}/negative/{file}'))
            self.labels.append(0.)
        for file in os.listdir(f'{path}/positive/'):
            self.images.append(Image.open(f'{path}/positive/{file}'))
            self.labels.append(1.)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]

        image_tensor = self.pil2tensor(image).to(self.device)
        label_tensor = torch.tensor([label], device=self.device)

        # build up 128x128 sub-tensors
        sub_list = []

        for i in range(25):
            h_start = torch.randint(0, image_tensor.shape[1] - 128, (1,))
            w_start = torch.randint(0, image_tensor.shape[2] - 128, (1,))
            sub_list.append(image_tensor[:, h_start:h_start + 128, w_start:w_start + 128])

        sub_tensors = torch.cat(sub_list, dim=1)
        return sub_tensors, label_tensor

    def __len__(self):
        return len(self.images)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 18, 3)
        self.conv4 = nn.Conv2d(18, 24, 3)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.ModuleList([nn.Linear(3456, 256) for _ in range(25)])
        self.fc2 = nn.ModuleList([nn.Linear(256, 64) for _ in range(25)])
        self.fc3 = nn.ModuleList([nn.Linear(64, 16) for _ in range(25)])

        self.fc4 = nn.Linear(400, 80)
        self.fc5 = nn.Linear(80, 10)
        self.fc6 = nn.Linear(10, 1)

    def forward(self, x):
        sub_tensors = [x[:, :, i * 128:((i + 1) * 128)] for i in range(25)]
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

            sub_output.append(torch.flatten(x1, start_dim=1))

        x = torch.cat(sub_output, dim=1)
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))

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
        writer.add_scalar('Precision/train', train_acc, epoch)

        if verbose:
            print(f'Epoch {epoch} Loss/Train {epoch_loss}')
            print(f'Epoch {epoch} Precision/train {train_acc}')

        if test_while_train:
            net.eval()
            test_acc, test_loss = model_test(test_set, net)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Precision/test', test_acc, epoch)
            if verbose:
                print(f'Epoch {epoch} Loss/test {test_loss}')
                print(f'Epoch {epoch} Precision/test {test_acc}')

    return net


def model_test(test_set: Dataset, net: nn.Module):
    net.eval()
    test_x, test_y = next(iter(DataLoader(test_set, batch_size=50, shuffle=True)))
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

    # return f1(tp, fp, fn), loss
    return precision(tp, fp), loss


def f1(tp, fp, fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)

    if prec + rec == 0:
        return 0
    else:
        return 2 * prec * rec / (prec + rec)


def precision(tp, fp):
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)


def recall(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)


def L1(net: nn.Module):
    s = 0
    for param in net.parameters():
        s += torch.sum(torch.abs(param))
    return s


if __name__ == '__main__':
    lr = 0.0002
    batch = 25

    writer = SummaryWriter(comment=f'lr_{lr}_batch_{batch}_128')
    dataset = Images('data', device)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    # model = model_train(train, test, epochs=500, batch_size=batch, learning_rate=lr, test_while_train=False)
    model = model_train(dataset, test, epochs=500, batch_size=batch, learning_rate=lr, test_while_train=False)

    torch.save(model.state_dict(), f'saved_models/lr_{lr}_batch_{batch}_128.pt')
