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
        self.batches = []

    def __call__(self, outputs, loss_outputs):
        self.batches.append(outputs)

    def reset(self):
        self.batches = []

    def name(self):
        return self.__class__.__name__ + " (pos-, neg+, diff+)"

    def value(self):
        average_positive_dist = 0
        average_negative_dist = 0
        count = 0
        for batch in self.batches:
            anchors = batch[0]  # (batch_size, embedding_size)
            positives = batch[1]  # (batch_size, embedding_size)
            negatives = batch[2]  # (batch_size, embedding_size)
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
        self.batches = []
        self.multisiamese_mode = False

    def __call__(self, outputs, loss_outputs):
        if len(loss_outputs) == 3:
            self.multisiamese_mode = True
            self.batches.append((loss_outputs[1], loss_outputs[2]))  # positive similarity, negative similarity
        else:
            self.multisiamese_mode = False
            self.batches.append(outputs)

    def reset(self):
        self.batches = []
        self.multisiamese_mode = False

    def name(self):
        return self.__class__.__name__ + " (pos+, neg-)"

    def value(self):
        average_positive_similarity = 0
        average_negative_similarity = 0
        count = 0
        for batch in self.batches:
            if self.multisiamese_mode:
                average_positive_similarity += batch[0]
                average_negative_similarity += batch[1]
                count += 1
            else:
                anchors = batch[0]  # (batch_size, embedding_size)
                positives = batch[1]  # (batch_size, embedding_size)
                negatives = batch[2]  # (batch_size, embedding_size)
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
        if not self.multisiamese_mode:
            average_positive_similarity = average_positive_similarity.item()
            average_negative_similarity = average_negative_similarity.item()
        return digit_precision(average_positive_similarity, 3), digit_precision(average_negative_similarity, 3)


class EmbeddingCosineSimilarityAndDistanceLossMetric(Metric):
    def __init__(self):
        self.batches = []

    def __call__(self, outputs, loss_outputs):
        self.batches.append((loss_outputs[1], loss_outputs[2]))  # positive similarity, negative similarity

    def reset(self):
        self.batches = []

    def name(self):
        return self.__class__.__name__ + " (sim-, dist-)"

    def value(self):
        if len(self.batches) == 0:
            return 0, 0
        sim = 0
        dist = 0
        for batch in self.batches:
            sim += batch[0]
            dist += batch[1]
        avg_sim = sim / len(self.batches)
        avg_dist = dist / len(self.batches)
        return digit_precision(avg_sim, 3), digit_precision(avg_dist, 3)
