from importlib import reload
import json
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
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
from data_loader import get_image_size_from_model_type, get_datasets, get_test_set, get_multisiamese_datasets, get_soft_multisiamese_datasets
from utils import pathfinding
cuda = torch.cuda.is_available()

random_parameters = {
    "model_type": ["resnet50", "mobilenet", "efficientnet-b0", "efficientnet-b1"],  # "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7"],
    "lr": (1e-3, 1e-5),
    "fc": [4, 8, 16, 32, 64, 128, 256],
    "batch_size": [8, 16, 32],
    "dropout": [False, True],
    "dropout_rate": (0.01, 0.6),
    "scheduler_type": ["step"],  # "cosine_annealing"],
    "scheduler_step_size": (3, 7),
    "scheduler_gamma": (0.01, 0.2),
    "scheduler_t_mult": (0.5, 2),
    "scheduler_eta_min": (0, 1e-5),
    "use_max_cycles": [False, True],
    "max_cycles_for_pairs": (1.0, 5.0),
    "inter_video_pairs": [False, True],
    "data_augmentation": [False, True],
    "dataset_normalization": [False, True]
}


def get_max_batch_size_for_model_type(model_type):
    max_batch_size = {
        "resnet50": 32,
        "mobilenet": 32,
        "efficientnet-b0": 24,
        "efficientnet-b1": 12,
        "efficientnet-b2": 24,
        "efficientnet-b3": 15,
        "efficientnet-b4": 6
    }
    return max_batch_size[model_type]


def generate_xp_folder():
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
    return save_path


def generate_config(save_path):
    config = {}
    for key, value in random_parameters.items():
        if type(value) is tuple:
            if type(value[0]) is int and type(value[1]) is int:
                config[key] = np.random.randint(value[0], value[1] + 1)
            else:
                config[key] = np.random.rand() * (value[1] - value[0]) + value[0]
        else:
            config[key] = value[np.random.randint(len(value))]
            if key == "batch_size":
                max_batch_size = get_max_batch_size_for_model_type(config["model_type"])
                if config[key] > max_batch_size:
                    config[key] = max_batch_size
    print("Generated config:", config)
    config_file = json.dumps(config)
    f = open(save_path + r"\config.json", "w")
    f.write(config_file)
    f.close()
    return config


def setup(config):
    torch.cuda.set_device(0)
    if config["model_type"] == "mobilenet":
        embedding_net = models.mobilenet_v2(pretrained=True)
    elif config["model_type"] == "resnet50":
        embedding_net = models.resnet50(pretrained=True)
    else:
        embedding_net = EfficientNet.from_pretrained(config["model_type"])
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
        # scheduler = lr_scheduler.CosineAnnealingLr(optimizer, T_max=n_epochs, eta_min=config["scheduler_eta_min"], last_epoch=-1)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=5e-6, last_epoch=-1)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["scheduler_step_size"], T_mult=config["scheduler_t_mult"], eta_min=config["scheduler_eta_min"], last_epoch=-1)
    log_interval = 100
    start_epoch = 0

    return model, loss_fn, optimizer, scheduler, n_epochs, log_interval, start_epoch


def load_training_set(config):
    training_path = r'C:\Users\root\Data\Angiographie'
    validation_paths = [
        fr'{training_path}\AC-1',  # 6 sequences
        fr'{training_path}\P31',  # 6 sequences
        fr'{training_path}\KC-3',  # 6 sequences
        fr'{training_path}\P28',  # 12 sequences
        fr'{training_path}\P26',  # 10 sequences
        fr'{training_path}\pt006',  # 9 sequences
        fr'{training_path}\pt026',  # 8 sequences
        fr'{training_path}\pt036',  # 6 sequences
        fr'{training_path}\pt041',  # 9 sequences
        fr'{training_path}\pt047'  # 13 sequences
    ]
    test_paths = [
        fr'{training_path}\G3',  # 7 sequences
        fr'{training_path}\G14',  # 13 sequences
        fr'{training_path}\P21',  # 10 sequences
        fr'{training_path}\P27',  # 8 sequences
        fr'{training_path}\MJY-9',  # 2 sequences
        fr'{training_path}\KR-11',  # 7 sequences
        fr'{training_path}\pt004',  # 6 sequences
        fr'{training_path}\pt010',  # 11 sequences
        fr'{training_path}\pt011',  # 2 sequences
        fr'{training_path}\pt028',  # 11 sequences
        fr'{training_path}\pt037',  # 1 sequence
    ]
    max_cycles_for_pairs = config["max_cycles_for_pairs"] if config["use_max_cycles"] else 0
    sequence = 3
    batch_size = config["batch_size"]
    inter_video_pairs = config["inter_video_pairs"]
    use_data_augmentation = config["data_augmentation"]
    use_data_normalization = config["dataset_normalization"]
    img_size = get_image_size_from_model_type(config["model_type"])
    training_set, validation_set = get_soft_multisiamese_datasets(training_path, validation_paths, test_paths, max_cycles_for_pairs, sequence, 1000, batch_size, inter_video_pairs, use_data_augmentation, use_data_normalization, img_size)
    normalization_parameters = None if not use_data_normalization else {
        "mean": training_set.data_mean,
        "std": training_set.data_std,
        # "normalize": training_set.normalize
    }
    return training_set, validation_set, normalization_parameters


def train(training_set, validation_set, model, loss_fn, optimizer, scheduler, n_epochs, log_interval, start_epoch, save_path):
    train_loader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
    metrics = []  # [EmbeddingCosineSimilarityAndDistanceLossMetric()]
    fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=start_epoch, save_progress_path=save_path, metrics=metrics, measure_weights=True, show_plots=False)


def load_best_model(save_path, model):
    test_folder_path = save_path
    # Find latest model weights in folder
    latest_epoch = -1
    weights_files = []
    for file in os.listdir(test_folder_path):
        if ".pth" in file:
            epoch = int(file.split("_")[-1].split(".")[0])
            weights_files.append((file, epoch))
            if epoch > latest_epoch:
                latest_epoch = epoch

    # Delete old model weights
    for weights_file in weights_files:
        if weights_file[1] != latest_epoch:
            os.remove(test_folder_path + "\\" + weights_file[0])

    load_state_path = test_folder_path + fr"\training_state_{latest_epoch}.pth"

    print(load_state_path)
    state = torch.load(load_state_path)
    model.load_state_dict(state['model'])
    model.eval()


def load_test_set(config):
    test_paths = [
        r'C:\Users\root\Data\Angiographie\G3',  # 7 sequences
        r'C:\Users\root\Data\Angiographie\P21',  # 10 sequences
        r'C:\Users\root\Data\Angiographie\G14',  # 4 sequences
        r'C:\Users\root\Data\Angiographie\MJY-9',  # 2 sequences
        r'C:\Users\root\Data\Angiographie\KR-11'  # 7 sequences
    ]
    img_size = get_image_size_from_model_type(config["model_type"])
    test_set = get_test_set(test_paths, img_size)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    return test_set, test_loader


def compute_matrices(test_loader, model, normalization_parameters):
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
            if normalization_parameters is not None:
                sequences -= normalization_parameters["mean"]
                sequences /= normalization_parameters["std"]
                # sequences = normalization_parameters["normalize"](torch.from_numpy(sequences))

            # sequences: (batch, video_frame, channel, width, height)
            print(f"Batch {batch_index + 1}/{len(test_loader)} ({name}) with {len(sequences[0])} sequences")
            embeddings = model(sequences[0])
            all_embeddings.append(embeddings)
            names.append(name)

        distance_matrices = {}
        similarity_matrices = {}
        combination_matrices = {}
        current_name = None
        for i in range(len(all_embeddings)):
            name_i = names[i].split(' ')[0]
            if not name_i == current_name:
                print(f"Combinations of {name_i}")
            distance_matrices[names[i]] = {}
            similarity_matrices[names[i]] = {}
            combination_matrices[names[i]] = {}
            current_name = name_i
            for j in range(i, len(all_embeddings)):
                if name_i == names[j].split(' ')[0]:
                    print(f"Comparison of {names[i]} and {names[j]}")
                    distance_matrix, similarity_matrix = calc_distance_and_similarity_matrices(all_embeddings[i], all_embeddings[j])
                    distance_matrices[names[i]][names[j]] = distance_matrix
                    similarity_matrices[names[i]][names[j]] = similarity_matrix
                    combination_matrices[names[i]][names[j]] = np.copy(distance_matrix * (similarity_matrix+1))

        return distance_matrices, similarity_matrices, combination_matrices


def run_pathfinding(test_set, test_loader, distance_matrices, similarity_matrices, combination_matrices):
    all_distance_scores = np.array([])
    all_similarity_scores = np.array([])
    all_combination_scores = np.array([])
    ordered_distance_scores = []
    all_paths = {}
    for batch_index_a, sequences_a in enumerate(test_loader):
        name_a = sequences_a[1][0]
        for batch_index_b, sequences_b in enumerate(test_loader):
            name_b = sequences_b[1][0]
            if name_b in distance_matrices[name_a]:
                symmetrical = name_a == name_b
                distance_matrix = np.copy(distance_matrices[name_a][name_b])
                similarity_matrix = np.copy(similarity_matrices[name_a][name_b])
                combination_matrix = np.copy(combination_matrices[name_a][name_b])
                ground_truth = test_set.get_similarity_matrix(name_a, name_b)
                for matrix_type in range(3):
                    matrix = distance_matrix if matrix_type == 0 else similarity_matrix if matrix_type == 1 else combination_matrix
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
                    if name_a not in all_paths:
                        all_paths[name_a] = {}
                    if name_b not in all_paths[name_a]:
                        all_paths[name_a][name_b] = ([], [], [])
                    for pathfinding_index, node in enumerate(nodes):
                        pairs = []
                        current_scores = []
                        path_scores = []
                        while node is not None:
                            pairs.append(node.point)
                            node = node.parent
                        # if len(pairs) < 10:
                        #     continue
                        for point in pairs:
                            current_scores.append(ground_truth[point])
                            path_scores.append((point, ground_truth[point]))
                            matrix[point] += line_value
                        scores += current_scores
                        all_paths[name_a][name_b][matrix_type].append(path_scores)

                    scores = np.array(scores)
                    print(name_a, name_b, "Mean score:", scores.mean(), f"for {len(scores)} scores")
                    if matrix_type == 0:
                        all_distance_scores = np.append(all_distance_scores, scores)
                        ordered_distance_scores.append((scores.mean(), name_a + " - " + name_b))
                    elif matrix_type == 1:
                        all_similarity_scores = np.append(all_similarity_scores, scores)
                    else:
                        all_combination_scores = np.append(all_combination_scores, scores)

    ordered_distance_scores.sort(key=lambda x: x[0])
    return all_distance_scores, all_similarity_scores, all_combination_scores, ordered_distance_scores, all_paths


def find_best_path(all_paths, test_set, save_path):
    average_scores = [0, 0, 0]
    count = 0
    for name_a, video_pairs in all_paths.items():
        for name_b, matrix_types in video_pairs.items():
            for matrix_type, paths in enumerate(matrix_types):
                # Find longest path
                max_length = 0
                for path in paths:
                    dist = np.sqrt((path[0][0][0] - path[-1][0][0]) ** 2 + (path[0][0][1] - path[-1][0][1]) ** 2)
                    if dist > max_length:
                        max_length = dist

                # Find the straightest long path
                max_straightness = 0
                straightest_path = -1
                for path_index, path in enumerate(paths):
                    dist = np.sqrt((path[0][0][0] - path[-1][0][0]) ** 2 + (path[0][0][1] - path[-1][0][1]) ** 2)
                    if dist >= max_length * 0.9:
                        straightness = dist / (len(path) - 1) / np.sqrt(2)
                        if straightness >= max_straightness:
                            max_straightness = straightness
                            straightest_path = path_index

                if straightest_path > -1:
                    # Find the max score path on the ground truth matrix
                    ground_truth = test_set.get_similarity_matrix(name_a, name_b)
                    symmetrical = name_a == name_b
                    starting_point = paths[straightest_path][0][0]
                    ground_truth_node = pathfinding(1-ground_truth, symmetrical, starting_points=[starting_point])[0]
                    ground_truth_path = []
                    while ground_truth_node is not None:
                        ground_truth_path.append((ground_truth_node.point, ground_truth[ground_truth_node.point]))
                        ground_truth_node = ground_truth_node.parent

                    score = np.array([x[1] for x in paths[straightest_path]]).mean()
                    max_score = np.array([x[1] for x in ground_truth_path]).mean()
                    adjusted_score = score / max_score
                    average_scores[matrix_type] += adjusted_score
                    print(f"{name_a}, {name_b}: Path {straightest_path} of length {len(paths[straightest_path])} with straightness of {round(max_straightness * 1000) / 10}% and score of {round(score * 1000) / 1000}/{round(max_score * 1000) / 1000}={round(adjusted_score * 1000) / 1000}")
                else:
                    print(f"{name_a}, {name_b}, {matrix_type}: No valid path found. Score of 0")
            count += 1

    f = open(save_path + r"\result.txt", "w")
    for i, matrix_type in enumerate(["Distance", "Similarity", "Combination"]):
        print(f"Average score ({matrix_type})", average_scores[i] / count)
        average_scores[i] /= count
        f.write(f"{round(average_scores[i] * 10000) / 10000}\n")
    f.close()


# def compute_global_pathfinding_results():
#     print("Total distance score mean:", round(all_distance_scores.mean() * 10000) / 10000)
#     print("Total distance score variance:", round(all_distance_scores.var() * 10000) / 10000)
#     print("Total similarity score mean:", round(all_similarity_scores.mean() * 10000) / 10000)
#     print("Total similarity score variance:", round(all_similarity_scores.var() * 10000) / 10000)
#     print("Ordered distance scores:", ordered_distance_scores)
#
#     f = open(save_path + r"\result.txt", "w")
#     f.write(f"{round(all_distance_scores.mean() * 10000) / 10000}")
#     f.close()


def evaluate_xps_without_result():
    base_path = r"E:\Users\root\Projects\VideoSynchronizationWithPytorch\trainings\hyperparameter_search"
    for path, subfolders, files in os.walk(base_path):
        if "config.json" in files and "progress.txt" in files and "result.txt" not in files:
            f = open(path + r"\config.json", 'r')
            config = f.readline()
            config = json.loads(config)
            f.close()
            save_path = path
            model, _, _, _, _, _, _ = setup(config)
            load_best_model(save_path, model)
            test_set, test_loader = load_test_set(config)
            distance_matrices, similarity_matrices, combination_matrices = compute_matrices(test_loader, model)
            all_distance_scores, all_similarity_scores, all_combination_scores, ordered_distance_scores, all_paths = run_pathfinding(test_set, test_loader, distance_matrices, similarity_matrices, combination_matrices)
            find_best_path(all_paths, test_set, save_path)


if __name__ == '__main__':
    # Generate config, train and evaluate
    save_path = generate_xp_folder()
    config = generate_config(save_path)
    model, loss_fn, optimizer, scheduler, n_epochs, log_interval, start_epoch = setup(config)
    training_set, validation_set, normalization_parameters = load_training_set(config)
    train(training_set, validation_set, model, loss_fn, optimizer, scheduler, n_epochs, log_interval, start_epoch, save_path)
    load_best_model(save_path, model)
    test_set, test_loader = load_test_set(config)
    distance_matrices, similarity_matrices, combination_matrices = compute_matrices(test_loader, model, normalization_parameters)
    all_distance_scores, all_similarity_scores, all_combination_scores, ordered_distance_scores, all_paths = run_pathfinding(test_set, test_loader, distance_matrices, similarity_matrices, combination_matrices)
    find_best_path(all_paths, test_set, save_path)

    # Load configs and evaluate
    # evaluate_xps_without_result()
