import torch
import torchvision
from torchvision import transforms
from torch.utils import data

def load_data(batch_size, resize=None):  
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=trans,
        download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root="data",
        train=False,
        transform=trans,
        download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))