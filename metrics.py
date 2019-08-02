import torch


def digit_precision(value, count):
    return int(value * 10**count) / 10**count


class Metric:
    def __call__(self, outputs, loss_outputs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError


class EmbeddingL2DistanceMetric(Metric):
    def __init__(self):
        self.embeddings = []

    def __call__(self, outputs, loss_outputs):
        self.embeddings.append(outputs)

    def reset(self):
        self.embeddings = []

    def name(self):
        return self.__class__.__name__ + " (pos-, neg+, diff+)"

    def value(self):
        average_positive_dist = 0
        average_negative_dist = 0
        count = 0
        for triplets in self.embeddings:
            anchors = triplets[0]  # (batch_size, embedding_size)
            positives = triplets[1]  # (batch_size, embedding_size)
            negatives = triplets[2]  # (batch_size, embedding_size)
            count += anchors.shape[0]
            average_positive_dist += (positives - anchors).pow(2).sum()
            average_negative_dist += (negatives - anchors).pow(2).sum()
        if count == 0:
            return 0, 0, 0
        average_positive_dist /= count
        average_negative_dist /= count
        return digit_precision(average_positive_dist.item(), 2), \
               digit_precision(average_negative_dist.item(), 2), \
               digit_precision(average_negative_dist.item() - average_positive_dist.item(), 2)


class EmbeddingCosineSimilarityMetric(Metric):
    def __init__(self):
        self.embeddings = []

    def __call__(self, outputs, loss_outputs):
        self.embeddings.append(outputs)

    def reset(self):
        self.embeddings = []

    def name(self):
        return self.__class__.__name__ + " (pos+, neg-)"

    def value(self):
        average_positive_similarity = 0
        average_negative_similarity = 0
        count = 0
        for triplets in self.embeddings:
            anchors = triplets[0]  # (batch_size, embedding_size)
            positives = triplets[1]  # (batch_size, embedding_size)
            negatives = triplets[2]  # (batch_size, embedding_size)
            batch_size = anchors.shape[0]
            count += batch_size
            for i in range(batch_size):
                normalized_anchor = anchors[i] / torch.norm(anchors[i])
                normalized_positive = positives[i] / torch.norm(positives[i])
                normalized_negative = negatives[i] / torch.norm(negatives[i])
                average_positive_similarity += normalized_anchor.dot(normalized_positive)
                average_negative_similarity += normalized_anchor.dot(normalized_negative)
        if count == 0:
            return 0, 0
        average_positive_similarity /= count
        average_negative_similarity /= count
        return digit_precision(average_positive_similarity.item(), 3), digit_precision(average_negative_similarity.item(), 3)
