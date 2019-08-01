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
        for batch in self.embeddings:
            for triplet in batch:
                count += 1
                average_positive_dist += (triplet[1] - triplet[0]).pow(2).sum()
                average_negative_dist += (triplet[2] - triplet[0]).pow(2).sum()
        if count == 0:
            return 0, 0
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
        for batch in self.embeddings:
            for triplet in batch:
                count += 1
                normalized_anchor = triplet[0] / torch.norm(triplet[0])
                average_positive_similarity += normalized_anchor.dot(triplet[1] / torch.norm(triplet[1]))
                average_negative_similarity += normalized_anchor.dot(triplet[2] / torch.norm(triplet[2]))
        if count == 0:
            return 0, 0
        average_positive_similarity /= count
        average_negative_similarity /= count
        return digit_precision(average_positive_similarity.item(), 3), digit_precision(average_negative_similarity.item(), 3)
