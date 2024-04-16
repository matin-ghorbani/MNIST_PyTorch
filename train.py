import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
# train = datasets.MNIST(root='data', train=True,
#                        transform=ToTensor(), download=False)
dataset = DataLoader(train, batch_size=32)


class ImageClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=64 * (28 - 6) * (28 - 6), out_features=10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = ImageClassifier().to(device)
    opt = Adam(params=clf.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    EPOCHS = 7

    for epoch in range(1, EPOCHS + 1):
        for batch in dataset:
            x, y = map(lambda a: a.to(device), batch)
            
            y_pred = clf(x)
            loss = loss_func(y_pred, y)

            # Apply Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    torch.save(clf.state_dict(), 'mnist_classifier_torch.pth')
