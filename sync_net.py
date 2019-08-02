import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
from itertools import combinations
from data_loader import get_datasets
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
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class CosineSimilarityTripletLoss(nn.Module):
    """
    Cosine Similarity Triplet loss
    """

    def __init__(self, margin):
        super(CosineSimilarityTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        batch_size, embedding_size = anchor.shape
        normalized_anchor = anchor / torch.norm(anchor)
        normalized_positive = positive / torch.norm(positive)
        normalized_negative = negative / torch.norm(negative)
        # print("normalized anchor", normalized_anchor)
        # print("normalized pos", normalized_positive)
        # print("normalized neg", normalized_negative)
        positive_similarity = torch.bmm(normalized_anchor.view(batch_size, 1, embedding_size), normalized_positive.view(batch_size, embedding_size, 1))  # It works like a batched dot product
        negative_similarity = torch.bmm(normalized_anchor.view(batch_size, 1, embedding_size), normalized_negative.view(batch_size, embedding_size, 1))  # It works like a batched dot product
        positive_distance = 1 - positive_similarity
        negative_distance = 1 - negative_similarity
        # print("pos dist", positive_distance)
        # print("neg dist", negative_distance)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean() if size_average else losses.sum()


class LosslessTripletLoss(nn.Module):
    """
    Class taken (and modified) from

    Lossless Triplet loss
    A more efficient loss function for Siamese NN
    by Marc-Olivier Arsenault
    Feb 15, 2018
    https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    """

    """
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)
    """
    def __init__(self, N=3, beta=None, epsilon=1e-8):
        super(LosslessTripletLoss, self).__init__()
        self.N = N
        self.beta = N if beta is None else beta
        self.epsilon = epsilon

    def forward(self, anchor, positive, negative, size_average=True):
        # distance between the anchor and the positive
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), 1)
        # distance between the anchor and the negative
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), 1)

        # -ln(-x/N+1)
        pos_dist = -torch.log(-(pos_dist / self.beta) + 1 + self.epsilon)
        neg_dist = -torch.log(-((self.N - neg_dist) / self.beta) + 1 + self.epsilon)

        # compute loss
        losses = neg_dist + pos_dist

        # TODO find why it sometimes return nan
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
        if len(x.shape) == 4:
            return self.embedding_net(x)
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
    reset_first_layer(model)
    # reset the weights of the fully connected layer at the end because we want to learn an embedding that is useful to our sequence and not to classify images
    model.fc = nn.Linear(2048, 16)
    nn.init.xavier_uniform_(model.fc.weight)


def reset_first_layer(model):
    # reset the weights of the first convolution layer because our data will have 3 channels, but not RGB
    nn.init.xavier_uniform_(model.conv1.weight)


def replace_last_layer(model, out_features):
    # reset the weights of the fully connected layer at the end because we want to learn an embedding that is useful to our sequence and not to classify images
    model.fc = nn.Linear(2048, out_features)
    nn.init.xavier_uniform_(model.fc.weight)


def add_sigmoid_activation(model):
    return nn.Sequential(model, nn.modules.Sigmoid())


if __name__ == "__main__":
    # torch.cuda.set_device(0)
    # embedding_net = models.resnet50(pretrained=True)
    # reset_first_and_last_layers(embedding_net)
    # model = TripletNet(embedding_net)
    # model.cuda(0)
    # model = nn.DataParallel(model).cuda()
    # lr = 1e-3
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = TripletLoss(margin=0.5)
    # scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    # n_epochs = 20
    # log_interval = 100
    # dataset = get_datasets()
    # train_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
    # fit(train_loader, None, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    # for child in embedding_net.named_children():
    #     print(child)
    # summary(embedding_net, (3, 224, 224))

    # triplets = triplet_selector.get_triplets(None, torch.tensor(np.array([1, 0, 1, 0, 1])))
    # print(triplets)

    # loss = CosineSimilarityTripletLoss(margin=0.5)
    # a = torch.FloatTensor([[1, 1]])
    # b = torch.FloatTensor([[-1, 1]])
    # c = torch.FloatTensor([[-1, -1]])
    # d = torch.FloatTensor([[0, 1]])
    # print("loss", loss.forward(a, b, d))
