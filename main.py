import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
import os

# torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
device = 'cuda'


class Images(Dataset):
    def __init__(self, path: str, device: str = 'cpu'):
        self.images = []
        self.labels = []
        self.device = device
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
        ])
        for file in os.listdir(f'{path}/negative/'):
            img = read_image(f'{path}/negative/{file}')
            image_tensor = self.transform(img).to(self.device)
            self.images.append(image_tensor)
            self.labels.append(torch.tensor([0.], device=self.device))
        for file in os.listdir(f'{path}/positive/'):
            img = read_image(f'{path}/positive/{file}')
            image_tensor = self.transform(img).to(self.device)
            self.images.append(image_tensor)
            self.labels.append(torch.tensor([1.], device=self.device))

    def __getitem__(self, item):
        image_tensor = self.images[item]
        label_tensor = self.labels[item]

        image_tensor = self._crop(image_tensor)

        return image_tensor, label_tensor

    @staticmethod
    def _crop(t: torch.Tensor) -> torch.Tensor:
        """
        crop 256x256 from center
        :param t: input image tensor of shape (3, h, w)
        :return: 256x256 cropped tensor from the center of t
        """
        h_start = torch.randint(0, t.shape[1] - 256, (1,)) if t.shape[1] > 256 else 0
        w_start = torch.randint(0, t.shape[2] - 256, (1,)) if t.shape[2] > 256 else 0
        out = t[:, h_start: h_start + 256, w_start: w_start + 256]
        return out

    def __len__(self):
        return len(self.images)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.conv2_1 = nn.Conv2d(16, 16, 3, padding='same')
        self.conv2_2 = nn.Conv2d(16, 16, 3, padding='same')
        self.conv2_3 = nn.Conv2d(16, 16, 3, padding='same')
        self.conv2_4 = nn.Conv2d(16, 16, 3, padding='same')

        self.conv3 = nn.Conv2d(16, 32, 3, padding='same')
        self.conv3_1 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv3_res = nn.Conv2d(16, 32, 1, stride=2)
        self.conv3_2 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv3_3 = nn.Conv2d(32, 32, 3, padding='same')

        self.conv4 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4_1 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv4_res = nn.Conv2d(32, 64, 1, stride=2)
        self.conv4_2 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv4_3 = nn.Conv2d(64, 64, 3, padding='same')

        self.conv5 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv5_1 = nn.Conv2d(128, 128, 3, padding='same')
        self.conv5_res = nn.Conv2d(64, 128, 1, stride=2)
        self.conv5_2 = nn.Conv2d(128, 128, 3, padding='same')
        self.conv5_3 = nn.Conv2d(128, 128, 3, padding='same')

        self.conv6 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv6_1 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv6_res = nn.Conv2d(128, 256, 1, stride=2)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv6_3 = nn.Conv2d(256, 256, 3, padding='same')

        self.max_pool = nn.MaxPool2d(2)
        self.avg_pool = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = self.max_pool(torch.relu(self.conv1(x)))
        x2 = torch.relu(self.conv2_1(x1))
        x2 = torch.relu(self.conv2_2(x2))
        x1 = torch.relu(x1 + x2)
        x2 = torch.relu(self.conv2_3(x1))
        x2 = torch.relu(self.conv2_4(x2))
        x1 = torch.relu(x1 + x2)

        x2 = self.max_pool(torch.relu(self.conv3(x1)))
        x2 = torch.relu(self.conv3_1(x2))
        x3 = torch.relu(self.conv3_res(x1))
        x1 = torch.relu(x2 + x3)
        x2 = torch.relu(self.conv3_2(x1))
        x2 = torch.relu(self.conv3_3(x2))
        x1 = torch.relu(x1 + x2)

        x2 = self.max_pool(torch.relu(self.conv4(x1)))
        x2 = torch.relu(self.conv4_1(x2))
        x3 = torch.relu(self.conv4_res(x1))
        x1 = torch.relu(x2 + x3)
        x2 = torch.relu(self.conv4_2(x1))
        x2 = torch.relu(self.conv4_3(x2))
        x1 = torch.relu(x1 + x2)

        x2 = self.max_pool(torch.relu(self.conv5(x1)))
        x2 = torch.relu(self.conv5_1(x2))
        x3 = torch.relu(self.conv5_res(x1))
        x1 = torch.relu(x2 + x3)
        x2 = torch.relu(self.conv5_2(x1))
        x2 = torch.relu(self.conv5_3(x2))
        x1 = torch.relu(x1 + x2)

        x2 = self.max_pool(torch.relu(self.conv6(x1)))
        x2 = torch.relu(self.conv6_1(x2))
        x3 = torch.relu(self.conv6_res(x1))
        x1 = torch.relu(x2 + x3)
        x2 = torch.relu(self.conv6_2(x1))
        x2 = torch.relu(self.conv6_3(x2))
        x1 = torch.relu(x1 + x2)

        x1 = self.avg_pool(x1)

        x = torch.flatten(x1, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
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
            batch_loss = bce(batch_pred, batch_y) + 0.000001 * L1(net)
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
        loss = bce(test_pred, test_y) + 0.000001 * L1(net)

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
    lr = 5e-5
    batch = 50

    writer = SummaryWriter(comment=f'lr_{lr}_batch_{batch}_256')
    dataset = Images('data', device)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    # model = model_train(train, test, epochs=800, batch_size=batch, learning_rate=lr, test_while_train=True)
    model = model_train(dataset, test, epochs=700, batch_size=batch, learning_rate=lr, test_while_train=False)

    torch.save(model.state_dict(), f'saved_models/lr_{lr}_batch_{batch}.pt')
