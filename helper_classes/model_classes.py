import os
import shutil
import torch
import torch.nn.functional as F
import random
from torch import nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    """Create dataset inherited from torch.utils.data.Dataset

    Attributes:
        data_dir: train dir or test dir.
        alphabet_map: The map from char to index.
        img_names: File names of all image under the data_dir.
        labels: Labels of all image under the data_dir.
        trans: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
        to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

    """

    def __init__(self, data_dir, alphabet_map):
        self.data_dir = data_dir
        self.alphabet_map = alphabet_map
        self.img_names = os.listdir(self.data_dir)
        self.labels = [i.split('_')[1].split('.')[0] for i in self.img_names]
        self.trans = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        """Get single image by idx

        Args:
            idx: index

        Returns:
            img: torch.FloatTensor
            label: Actual lable of the image, like "ZOW-PRF-LFB".
        """
        img_path = os.path.join(self.data_dir, self.img_names[idx])
        img = Image.open(img_path)
        img = self.trans(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.labels)


class BiLSTM(nn.Module):
    """ Bidirectional LSTM and embedding layer.

    Attributes:
        rnn: Bidirectional LSTM
        linear: Embedding layer
    """

    def __init__(self, num_input, num_hiddens, num_output):
        super().__init__()
        self.rnn = nn.LSTM(num_input, num_hiddens, bidirectional=True)
        # the size of input of embedding layer should mutiply by 2, because of the bidirectional.
        self.linear = nn.Linear(num_hiddens * 2, num_output)

    def forward(self, X):
        rnn_out, _ = self.rnn(X)
        T, b, h = rnn_out.size()  # T: time step, b: batch size, h: hidden size
        t_rec = rnn_out.view(T * b, h)
        output = self.linear(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    """CRNN net, refer to the paper from https://arxiv.org/pdf/1507.05717v1.pdf.

    Attributes:
        cnn: nn.Sequential include conv2d/relu/maxpool2d/batchnorm layers,
        input size (1 x 32 X 200), output size ()
        rnn:
    """

    def __init__(self, num_class):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.rnn = nn.Sequential(
            BiLSTM(512, 256, 256),
            BiLSTM(256, 256, num_class)
        )

    def forward(self, X):
        cnn_out = self.cnn(X)  # cnn_out shape: (batch_size x channel x height x width)
        assert cnn_out.shape[2] == 1, "the height of conv must be 1"
        cnn_out = cnn_out.squeeze(2)  # squeeze the dim 2 (height) of cnn_out
        cnn_out = cnn_out.permute(2, 0, 1)  # move the width to the first dim, as the time step of rnn input
        output = self.rnn(cnn_out)  # output shape: (time step x batch_size x num_class)
        output = F.log_softmax(output, dim=2)  # do softmax at the dim of num_class
        return output

