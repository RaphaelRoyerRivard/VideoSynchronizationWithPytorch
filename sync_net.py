import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
from itertools import combinations
from data_loader import get_dataset
from trainer import fit
from torch.utils.data import DataLoader
cuda = torch.cuda.is_available()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    Taken from https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    Taken from https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class TripletNet(nn.Module):
    """
    https://github.com/adambielski/siamese-triplet/blob/master/networks.py
    """
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        # print("TripletNet input", x.shape)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        # print(x.type(), "vs", x1.type())
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


def reset_first_and_last_layers(model):
    # reset the weights of the first convolution layer because our data will have 3 channels, but not RGB
    conv1 = list(model.children())[0]
    nn.init.xavier_uniform_(conv1.weight)
    # reset the weights of the fully connected layer at the end because we want to learn an embedding that is useful to our sequence and not to classify images
    fc = list(model.children())[-1]
    nn.init.xavier_uniform_(fc.weight)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    embedding_net = models.resnet50(pretrained=True)
    reset_first_and_last_layers(embedding_net)
    model = TripletNet(embedding_net)
    model.cuda(0)
    model = nn.DataParallel(model).cuda()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = TripletLoss(margin=0.5)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100
    dataset = get_dataset()
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
    fit(train_loader, None, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    # for child in embedding_net.named_children():
    #     print(child)
    # summary(embedding_net, (3, 224, 224))

    # triplets = triplet_selector.get_triplets(None, torch.tensor(np.array([1, 0, 1, 0, 1])))
    # print(triplets)
