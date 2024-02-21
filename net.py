import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

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


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        

