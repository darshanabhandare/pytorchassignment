import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm
from pytsrc import utils


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_train_test_loaders() -> tuple:
    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    batch_size = 512

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

    return train_loader, test_loader


class ModelBuilder():
    device = None
    model = None
    name:str = None

    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    def __init__(self, model_name):
        self.name = model_name

    def build(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = Net().to(self.device)
        summary(self.model, input_size=(1, 28, 28))

    def run(self):
        train_loader, test_loader = get_train_test_loaders()

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
        # New Line
        criterion = F.nll_loss
        num_epochs = 20

        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}')
            self.train(self.model, self.device, train_loader, optimizer, criterion)
            self.test(self.model, self.device, test_loader, criterion)
            scheduler.step()


    def train(self, model, device, train_loader, optimizer, criterion):
        model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Predict
            pred = model(data)

            # Calculate loss
            loss = criterion(pred, target)
            train_loss+=loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += utils.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        self.train_acc.append(100*correct/processed)
        self.train_losses.append(train_loss/len(train_loader))

    def test(self, model, device, test_loader, criterion):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

                correct += utils.GetCorrectPredCount(output, target)


        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def plot_results(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")