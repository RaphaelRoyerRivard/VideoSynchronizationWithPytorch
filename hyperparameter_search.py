from importlib import reload
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import os
import sync_net
import trainer
import metrics
import data_loader
import utils
reload(sync_net)
reload(trainer)
reload(metrics)
reload(data_loader)
reload(utils)
from sync_net import reset_first_layer, replace_last_layer, add_sigmoid_activation, stop_running_var, freeze_model, TripletNet, MultiSiameseNet, TripletLoss, CosineSimilarityTripletLoss, LosslessTripletLoss, MultiSiameseCosineSimilarityLoss, SoftMultiSiameseCosineSimilarityLoss
from trainer import fit
from metrics import EmbeddingL2DistanceMetric, EmbeddingCosineSimilarityMetric, EmbeddingCosineSimilarityAndDistanceLossMetric
from data_loader import get_datasets, get_test_set, get_multisiamese_datasets, get_soft_multisiamese_datasets
from utils import pathfinding
cuda = torch.cuda.is_available()

random_parameters = {
    "lr": (1e-3, 1e-5),
    "fc": [4, 8, 16, 32, 64, 128, 256, 512],
    "batch_size": [16, 32, 64],
    "dropout": [False, True],
    "dropout_rate": (0.01, 0.6),
    "scheduler_type": ["step", "cosine_annealing"],
    "scheduler_step_size": (3, 7),
    "scheduler_gamma": (0.01, 0.2),
    "scheduler_eta_min": (0, 1e-5),
    "use_max_cycles": [False, True],
    "max_cycles_for_pairs": (1.0, 5.0),
    "inter_video_pairs": [False, True],
    "data_augmentation": [False, True]
}


def generate_config():
    config = {}
    for key, value in random_parameters.items():
        if type(value) is tuple:
            if type(value[0]) is int and type(value[1]) is int:
                config[key] = np.random.randint(value[0], value[1] + 1)
            else:
                config[key] = np.random.rand() * (value[1] - value[0]) + value[0]
        else:
            config[key] = value[np.random.randint(len(value))]
    print("Generated config:", config)
    return config


def setup():
    torch.cuda.set_device(0)
    embedding_net = models.mobilenet_v2(pretrained=True)
    if config["dropout"]:
        replace_last_layer(embedding_net, config["fc"], dropout=config["dropout_rate"])
    else:
        replace_last_layer(embedding_net, config["fc"])
    embedding_net.apply(stop_running_var)
    model = MultiSiameseNet(embedding_net)
    model.cuda(0)
    model = nn.DataParallel(model).cuda()
    lr = config["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = TripletLoss(margin=0.5)
    # loss_fn = CosineSimilarityTripletLoss(margin=0.5)
    # loss_fn = LosslessTripletLoss()
    # loss_fn = MultiSiameseCosineSimilarityLoss()
    loss_fn = SoftMultiSiameseCosineSimilarityLoss()
    n_epochs = 20
    if config["scheduler_type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer, config["scheduler_step_size"], gamma=config["scheduler_gamma"], last_epoch=-1)
    else:
        scheduler = lr_scheduler.CosineAnnealingLr(optimizer, T_max=n_epochs, eta_min=config["scheduler_eta_min"], last_epoch=-1)
    # scheduler = lr_scheduler.CosineAnnealingLr(optimizer, T_0=4, T_mult=2, eta_min=5e-6, last_epoch=-1)
    log_interval = 100
    start_epoch = 0
    save_path = r"E:\Users\root\Projects\VideoSynchronizationWithPytorch\trainings\hyperparameter_search"
    highest_id = 0
    if os.path.isdir(save_path):
        for folder in os.listdir(save_path):
            id = int(folder)
            if id > highest_id:
                highest_id = id
    save_path += fr"\{highest_id + 1}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    config_file = json.dumps(config)
    f = open(save_path + r"\config.json", "w")
    f.write(config_file)
    f.close()

    return model, loss_fn, optimizer, scheduler, n_epochs, log_interval, start_epoch, save_path


def load_training_set():
    training_path = r'C:\Users\root\Data\Angiographie'
    validation_paths = [
        r'C:\Users\root\Data\Angiographie\ABL-5',
        r'C:\Users\root\Data\Angiographie\G1',
        r'C:\Users\root\Data\Angiographie\G18'
    ]
    max_cycles_for_pairs = config["max_cycles_for_pairs"] if config["use_max_cycles"] else 0
    sequence = 3
    batch_size = config["batch_size"]
    inter_video_pairs = config["inter_video_pairs"]
    use_data_augmentation = config["data_augmentation"]
    training_set, validation_set = get_soft_multisiamese_datasets(training_path, validation_paths, max_cycles_for_pairs, sequence, 1000, batch_size, inter_video_pairs, use_data_augmentation)
    return training_set, validation_set


def train():
    train_loader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
    metrics = []  # [EmbeddingCosineSimilarityAndDistanceLossMetric()]
    fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=start_epoch, save_progress_path=save_path, metrics=metrics, measure_weights=True, show_plots=False)


def load_best_model():
    test_folder_path = save_path
    # Find latest model weights in folder
    latest_epoch = -1
    for file in os.listdir(test_folder_path):
        if ".pth" in file:
            epoch = int(file.split("_")[-1].split(".")[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
    load_state_path = test_folder_path + fr"\training_state_{latest_epoch}.pth"

    print(load_state_path)
    state = torch.load(load_state_path)
    model.load_state_dict(state['model'])
    model.eval()


def load_test_set():
    test_paths = [
        r'C:\Users\root\Data\Angiographie\ABL-5',
        r'C:\Users\root\Data\Angiographie\G1',
        r'C:\Users\root\Data\Angiographie\G18'
    ]
    test_set = get_test_set(test_paths)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    return test_set, test_loader


def compute_matrices():
    def calc_distance_and_similarity_matrices(embeddings1, embeddings2):
        distances = []
        similarities = []
        for i in range(len(embeddings1)):
            distances_i = []
            similarities_i = []
            for j in range(len(embeddings2)):
                # Distance
                dist_val = torch.sum(torch.abs(embeddings1[i] - embeddings2[j]))
                distances_i.append(dist_val.cpu().numpy())
                # Similarity
                normalized_embedding_i = embeddings1[i] / torch.norm(embeddings1[i])
                normalized_embedding_j = embeddings2[j] / torch.norm(embeddings2[j])
                sim_val = 1 - normalized_embedding_i.dot(normalized_embedding_j)
                similarities_i.append(sim_val.cpu().numpy())
            distances.append(distances_i)
            similarities.append(similarities_i)
        distances = np.array(distances)
        similarities = np.array(similarities)
        return distances, similarities

    with torch.no_grad():
        all_embeddings = []
        names = []
        for batch_index, sequences in enumerate(test_loader):
            name = sequences[1][0]
            sequences = sequences[0]

            # sequences: (batch, video_frame, channel, width, height)
            print(f"Batch {batch_index + 1}/{len(test_loader)} ({name}) with {len(sequences[0])} sequences")
            embeddings = model(sequences[0])
            all_embeddings.append(embeddings)
            names.append(name)

        distance_matrices = {}
        similarity_matrices = {}
        current_name = None
        for i in range(len(all_embeddings)):
            name_i = names[i].split(' ')[0]
            if not name_i == current_name:
                print(f"Combinations of {name_i}")
            distance_matrices[names[i]] = {}
            similarity_matrices[names[i]] = {}
            current_name = name_i
            for j in range(i, len(all_embeddings)):
                if name_i == names[j].split(' ')[0]:
                    print(f"Comparison of {names[i]} and {names[j]}")
                    distance_matrix, similarity_matrix = calc_distance_and_similarity_matrices(all_embeddings[i], all_embeddings[j])
                    distance_matrices[names[i]][names[j]] = distance_matrix
                    similarity_matrices[names[i]][names[j]] = similarity_matrix

        return distance_matrices, similarity_matrices


def run_pathfinding():
    all_distance_scores = np.array([])
    all_similarity_scores = np.array([])
    ordered_distance_scores = []
    for batch_index_a, sequences_a in enumerate(test_loader):
        name_a = sequences_a[1][0]
        sequences_a = sequences_a[0][0]
        for batch_index_b, sequences_b in enumerate(test_loader):
            name_b = sequences_b[1][0]
            sequences_b = sequences_b[0][0]
            if name_b in distance_matrices[name_a]:
                symmetrical = name_a == name_b
                distance_matrix = np.copy(distance_matrices[name_a][name_b])
                similarity_matrix = np.copy(similarity_matrices[name_a][name_b])
                ground_truth = test_set.get_similarity_matrix(name_a, name_b)
                for i in range(2):
                    matrix = distance_matrix if i == 0 else similarity_matrix
                    line_value = matrix.max() * 1.5
                    if symmetrical:
                        # Erase center line
                        for i in range(len(matrix)):
                            matrix[i, i] = matrix.max()
                            for j in range(4):
                                if i > j:
                                    offset = j + 1
                                    matrix[i-offset, i] = matrix.max()
                                    matrix[i, i-offset] = matrix.max()
                    nodes = pathfinding(matrix, symmetrical)
                    scores = []
                    for pathfinding_index, node in enumerate(nodes):
                        pairs = []
                        current_scores = []
                        while node is not None:
                            pairs.append(node.point)
                            node = node.parent
                        if len(pairs) < 10:
                            continue
                        for point in pairs:
                            current_scores.append(ground_truth[point])
                            matrix[point] += line_value
                        scores += current_scores

                    scores = np.array(scores)
                    print(name_a, name_b, "Mean score:", scores.mean(), f"for {len(scores)} scores")
                    if i == 0:
                        all_distance_scores = np.append(all_distance_scores, scores)
                        ordered_distance_scores.append((scores.mean(), name_a + " - " + name_b))
                    else:
                        all_similarity_scores = np.append(all_similarity_scores, scores)

    ordered_distance_scores.sort(key=lambda x: x[0])
    return all_distance_scores, all_similarity_scores, ordered_distance_scores


def compute_global_pathfinding_results():
    print("Total distance score mean:", round(all_distance_scores.mean() * 10000) / 10000)
    print("Total distance score variance:", round(all_distance_scores.var() * 10000) / 10000)
    print("Total similarity score mean:", round(all_similarity_scores.mean() * 10000) / 10000)
    print("Total similarity score variance:", round(all_similarity_scores.var() * 10000) / 10000)
    print("Ordered distance scores:", ordered_distance_scores)

    f = open(save_path + r"\result.txt", "w")
    f.write(f"{round(all_distance_scores.mean() * 10000) / 10000}")
    f.close()


if __name__ == '__main__':
    config = generate_config()
    model, loss_fn, optimizer, scheduler, n_epochs, log_interval, start_epoch, save_path = setup()
    training_set, validation_set = load_training_set()
    train()
    load_best_model()
    test_set, test_loader = load_test_set()
    distance_matrices, similarity_matrices = compute_matrices()
    all_distance_scores, all_similarity_scores, ordered_distance_scores = run_pathfinding()
    compute_global_pathfinding_results()
