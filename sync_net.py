import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
from itertools import combinations
from data_loader import get_datasets
from trainer import fit
from torch.utils.data import DataLoader
from utils import pairwise_distances
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
        normalized_anchors = anchor / torch.norm(anchor, dim=-1).view(batch_size, 1)
        normalized_positives = positive / torch.norm(positive, dim=-1).view(batch_size, 1)
        normalized_negatives = negative / torch.norm(negative, dim=-1).view(batch_size, 1)
        # print("normalized anchors", normalized_anchors)
        # print("normalized positives", normalized_positives)
        # print("normalized negatives", normalized_negatives)
        positive_similarities = torch.bmm(normalized_anchors.view(batch_size, 1, embedding_size), normalized_positives.view(batch_size, embedding_size, 1))  # It works like a batched dot product
        negative_similarities = torch.bmm(normalized_anchors.view(batch_size, 1, embedding_size), normalized_negatives.view(batch_size, embedding_size, 1))  # It works like a batched dot product
        positive_distances = 1 - positive_similarities
        negative_distances = 1 - negative_similarities
        # print("pos dist", positive_distances)
        # print("neg dist", negative_distances)
        losses = F.relu(positive_distances - negative_distances + self.margin)
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


class MultiSiameseCosineSimilarityLoss(nn.Module):
    """
    Multi Siamese Similarity loss
    Takes a batch of embeddings with masks for positive pairs and negative pairs.
    Useful to get a loss over several pairs with few images.
    """

    """
    Parameters
    embeddings: matrix of size (batch_size, embedding_size)
    positive_matrix: matrix of positive pairs of size (batch_size, batch_size)
    negative_matrix: matrix of negative pairs of size (batch_size, batch_size)
    
    Returns
    loss: scalar between 0 and 4 where 0 represents perfect similarity between positive pairs and perfect dissimilarity 
          between negative pairs while 4 is the opposite.
    average_positive_similarity: normalized dot product of positive pairs of embeddings
    average_negative_similarity: normalized dot product of negative pairs of embeddings
    """
    def forward(self, embeddings, positive_matrix, negative_matrix):
        batch_size, embedding_size = embeddings.shape

        # normalize embeddings
        normalized_embeddings = embeddings / torch.norm(embeddings, dim=-1).view(batch_size, 1)

        # calculate cosine similarity for every combination
        cosine_similarities = torch.bmm(normalized_embeddings.view(1, batch_size, embedding_size),
                                        normalized_embeddings.t().view(1, embedding_size, batch_size)).cpu()

        # apply the masks (positive and negative matrices) over the cosine similarity matrix
        positive_similarities = cosine_similarities * positive_matrix
        negative_similarities = cosine_similarities * negative_matrix

        # calculate the average positive and negative similarity
        positive_count = positive_matrix.sum()
        negative_count = negative_matrix.sum()
        average_positive_similarity = positive_similarities.sum() / (positive_count if positive_count > 0 else 1)
        average_negative_similarity = negative_similarities.sum() / (negative_count if negative_count > 0 else 1)

        positive_value = 1 - average_positive_similarity
        negative_value = 1 + average_negative_similarity
        loss = positive_value + negative_value
        return loss, average_positive_similarity, average_negative_similarity


class SoftMultiSiameseCosineSimilarityLoss(nn.Module):
    """
    Soft Multi Siamese Similarity loss
    Takes a batch of embeddings to computes the cosine similarity for each pair and compare it with the pair similarity
    matrix (ground truth).
    Useful to get a loss over several pairs with few images.
    """

    """
    Parameters
    embeddings: 1 or 2 matrices of size (batch_size, embedding_size)
    similarity_matrix: matrix of pair similarity of size (batch_size, batch_size)
    masks: matrix of masks of size (batch_size, batch_size) to consider only some pairs
    
    Returns
    loss: scalar between 0 and 1 where 0 represents perfect pair similarity while 1 is the opposite.
    """
    def forward(self, embeddings, similarity_matrix, masks):
        if len(embeddings) != len(similarity_matrix):
            embedding_size = embeddings.shape[1]
            batch_size_a = similarity_matrix.shape[1]
            batch_size_b = similarity_matrix.shape[2]

            # normalize embeddings
            normalized_embeddings_a = embeddings[:batch_size_a] / torch.norm(embeddings[:batch_size_a], dim=-1).view(batch_size_a, 1)
            normalized_embeddings_b = embeddings[batch_size_a:] / torch.norm(embeddings[batch_size_a:], dim=-1).view(batch_size_b, 1)

            # calculate cosine similarity for every combination
            cosine_similarities = torch.bmm(normalized_embeddings_a.view(1, batch_size_a, embedding_size),
                                            normalized_embeddings_b.t().view(1, embedding_size, batch_size_b)).cpu()

        else:
            batch_size, embedding_size = embeddings.shape

            # normalize embeddings
            normalized_embeddings = embeddings / torch.norm(embeddings, dim=-1).view(batch_size, 1)

            # calculate cosine similarity for every combination
            cosine_similarities = torch.bmm(normalized_embeddings.view(1, batch_size, embedding_size),
                                            normalized_embeddings.t().view(1, embedding_size, batch_size)).cpu()

        # we want the similarity to be between 0 (dissimilar) to 1 (similar)
        cosine_similarities = (cosine_similarities + 1) / 2

        # apply the masks to ignore some pairs
        cosine_similarities *= masks
        similarity_matrix *= masks

        diff = torch.abs(cosine_similarities - similarity_matrix)
        nonzero = torch.nonzero(diff)
        count = nonzero.shape[0]
        similarity_loss = diff.sum() / count if count > 0 else 0

        return similarity_loss, similarity_loss, 0


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


class MultiSiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(MultiSiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        # (batch_size, channels, width, height)
        # print("MultiSiameseNet input", x.shape)
        embeddings = self.embedding_net(x)
        return embeddings


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

    loss_fn = MultiSiameseCosineSimilarityLoss()
    embeddings = torch.FloatTensor([[1, 1], [0.5, 1], [-1, -1], [-0.5, -1]])
    positive_matrix = torch.FloatTensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    negative_matrix = torch.FloatTensor([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
    print("loss", loss_fn.forward(embeddings, positive_matrix, negative_matrix))
