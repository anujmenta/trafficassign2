import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(500, 50)
#         self.fc2 = nn.Linear(50, nclasses)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 500)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 200, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200, 250, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm2d(250)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(250*3*3, 150)
        self.fc2 = nn.Linear(150, nclasses)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        bt_size = x.size(0)

        x = self.stn(x)
        x = self.batchnorm1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        x = self.dropout(x)
        x = self.batchnorm2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        x = self.dropout(x)
        x = self.batchnorm3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        x = self.dropout(x)

        x = x.view(-1, 250*3*3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
