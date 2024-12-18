import cv2
import numpy as np
import os
import re
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import optical_flow, optical_flow2


class VideoFrameProvider(object):
    def __init__(self, images, names):
        self.current_video_id = 0
        self.type = "images"
        self.frames = [x[0] for x in images]
        # self.hb_frequencies = [x[1] for x in images]
        # self.contracted_frames = [x[2] for x in images]
        self.r_peaks = [x[1] for x in images]
        self.names = names

    def _get_videos(self):
        return self.frames

    def video_count(self):
        return len(self._get_videos())

    def select_video(self, video_id):
        assert 0 <= video_id < self.video_count(), f"'video_id' {video_id} is out of bounds (max {self.video_count()-1})"
        self.current_video_id = video_id

    def get_current_video_name(self):
        return self.names[self.current_video_id]

    def get_current_video_frame_count(self):
        return len(self.frames[self.current_video_id])

    def get_current_video_frame(self, frame_id):
        assert 0 <= frame_id < self.get_current_video_frame_count(), f"'frame_id' {frame_id} is out of bounds (max is {self.get_current_video_frame_count()})"
        return self.frames[self.current_video_id][frame_id]

    def get_current_video_heartbeat_frequency(self):
        # return self.hb_frequencies[self.current_video_id]
        r_peaks = self.get_current_video_r_peaks()
        diffs = []
        for i in range(1, len(r_peaks)):
            diffs.append(r_peaks[i] - r_peaks[i-1])
        return np.array(diffs).mean()

    # def get_current_video_contracted_frame(self):
    #     return self.contracted_frames[self.current_video_id]

    def get_current_video_r_peaks(self):
        return self.r_peaks[self.current_video_id]


def get_image_size_from_model_type(model_type):
    image_size = {
        "resnet50": 224,
        "mobilenet": 224,
        "efficientnet-b0": 224,
        "efficientnet-b1": 240,
        "efficientnet-b2": 260,
        "efficientnet-b3": 300,
        "efficientnet-b4": 380,
        "efficientnet-b5": 456,
        "efficientnet-b6": 528,
        "efficientnet-b7": 600
    }
    return image_size[model_type]


def get_all_valid_frames_in_paths(base_paths, paths_to_ignore, img_size=224):
    frames_only = []
    all_valid_frames = []
    video_names = []
    relevant_frames_file_name = "relevant_frames.txt"
    r_peaks_file_name = "r-peaks.npy"
    for base_path in base_paths:
        for path, subfolders, files in os.walk(base_path):
            should_ignore = False
            for path_to_ignore in paths_to_ignore:
                if path_to_ignore is not None and path_to_ignore in path:
                    should_ignore = True
                    break
            if should_ignore:
                continue
            if path.split("\\")[-1] == "seg":
                continue
            if relevant_frames_file_name not in files:
                continue
            if r_peaks_file_name not in files:
                continue

            split_path = path.split("\\export\\")
            angle = split_path[1]
            patient = split_path[0].split("\\Angiographie\\")[1]
            video_names.append(patient + ' ' + angle)

            valid_frames = []
            rfile = open(path + "/" + relevant_frames_file_name, "r")
            relevant_frames_line = rfile.readline()
            rfile.close()
            info = relevant_frames_line.split(';')
            first_frame = int(info[0])
            last_frame = int(info[1])
            if last_frame - first_frame + 1 < 15:
                print(f"Ignoring video {patient} {angle} since it only has {last_frame - first_frame + 1} valid frames")
                continue
            # freq = float(info[2])
            # contracted = float(info[3]) - first_frame
            # assert 0 <= contracted <= last_frame, f"Contracted frame of id {contracted} for {patient} - {angle} should be a valid frame between {first_frame} and {last_frame}"
            frame_count = 0
            for filename in files:
                if not bool(re.search('.*Frame[0-9]+\.jpg', filename)):
                    continue
                frame_id = int(filename.split("Frame")[1].split('.')[0])
                if frame_id > frame_count:
                    frame_count = frame_id
                if first_frame <= frame_id <= last_frame:
                    img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_size, img_size))
                    img = img.astype(np.float32)
                    img /= 255
                    valid_frames.append((frame_id, img))
                    frames_only.append(img)
            valid_frames.sort()
            valid_frames = [x[1] for x in valid_frames]
            all_r_peaks = np.load(path + "\\" + r_peaks_file_name)
            r_peaks = all_r_peaks * frame_count
            r_peaks = r_peaks - first_frame
            valid_indices = np.where(r_peaks > 0)[0]
            if len(valid_indices) > 0 and valid_indices[0] > 0:
                valid_indices = np.append(valid_indices[0] - 1, valid_indices)
            r_peaks = r_peaks[valid_indices]
            valid_indices = np.where(r_peaks < last_frame - first_frame)[0]
            if len(valid_indices) > 0 and valid_indices[-1] + 1 < len(r_peaks):
                valid_indices = np.append(valid_indices, valid_indices[-1] + 1)
            r_peaks = r_peaks[valid_indices]
            # all_valid_frames.append((valid_frames, freq, contracted))
            # print(len(valid_frames), f"valid frames [{first_frame}, {last_frame}] @{freq} and contracted at index {contracted}, in", path)
            all_valid_frames.append((valid_frames, r_peaks))
            print(f"{len(valid_frames)}/{frame_count} valid frames [{first_frame}, {last_frame}] with {len(r_peaks)} R-peaks ({len(all_r_peaks)} total), in", path)
    frames_only = np.array(frames_only)
    print(frames_only.shape)
    print("Original", frames_only.mean(), frames_only.std(), frames_only.min(), frames_only.max())
    # frames_only -= frames_only.mean()
    # print("Centered", frames_only.mean(), frames_only.std(), frames_only.min(), frames_only.max())
    # frames_only /= frames_only.std()
    # print("Standardized", frames_only.mean(), frames_only.std(), frames_only.min(), frames_only.max())
    # frames_only *= 0.229
    # print("Adjusted std", frames_only.mean(), frames_only.std(), frames_only.min(), frames_only.max())
    # frames_only += 0.485
    # print("Adjusted mean", frames_only.mean(), frames_only.std(), frames_only.min(), frames_only.max())
    # frames_only = (frames_only - frames_only.min()) / (frames_only.max() - frames_only.min())
    # print("Normalized", frames_only.mean(), frames_only.std(), frames_only.min(), frames_only.max())
    return all_valid_frames, video_names, frames_only.mean(), frames_only.std()


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    https://github.com/adambielski/siamese-triplet/blob/master/utils.py
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AngioSequenceTripletDataset(Dataset):
    """
    Yield frame triplets for phase 0
    """
    def __init__(self, path, path_to_ignore, sequence):
        """Sample most trivial training data for phase 1 (intra-video sampling of consecutive frames)

        Args:
            path (str): path in which we can find sequences of images
        """
        self.files, self.names, self.data_mean, self.data_std = get_all_valid_frames_in_paths(path, path_to_ignore)
        self.video_frame_provider = VideoFrameProvider(images=self.files, names=self.names)
        self.sequence = sequence
        self._calc_all_triplets()

    def _calc_all_triplets(self):
        self.triplets = []
        self.triplet_video_indices = []
        self.triplet_video_offset = []
        video_count = self.video_frame_provider.video_count()
        for video_id in range(video_count):
            video_triplets = []
            self.video_frame_provider.select_video(video_id)
            video_frame_count = self.video_frame_provider.get_current_video_frame_count()
            hb_freq = self.video_frame_provider.get_current_video_heartbeat_frequency()
            min_pos_dist = 1
            max_pos_dist = round(hb_freq / 8)
            min_neg_dist = round(hb_freq * 3 / 8)
            max_neg_dist = round(hb_freq * 5 / 8)
            frames = list(range(video_frame_count))[self.sequence-1:-max_neg_dist]
            print(f"Video {video_id} has {video_frame_count} frames and {len(frames)} anchors from {frames[0]} to {frames[-1]} @{hb_freq} f/hb")
            for a in frames:
                for p in range(min_pos_dist, max_pos_dist+1):
                    for n in range(min_neg_dist, max_neg_dist+1):
                        video_triplets.append([a, a+p, a+n])
                        self.triplet_video_indices.append(video_id)
            previous_count = 0 if video_id == 0 else self.triplet_video_offset[-1] + len(self.triplets[-1])
            self.triplets.append(video_triplets)
            self.triplet_video_offset.append(previous_count)

    def __len__(self):
        return len(self.triplet_video_indices)

    def __getitem__(self, item):
        video_id = self.triplet_video_indices[item]
        offset = self.triplet_video_offset[video_id]
        triplet_id = item - offset
        triplet = self.triplets[video_id][triplet_id]
        # print(f"triplet {item} {triplet} with id {triplet_id} is in video {video_id} which has an offset of {offset} ")
        self.video_frame_provider.select_video(video_id)
        anchor = []
        positive = []
        negative = []
        for sequence_index in reversed(range(self.sequence)):
            anchor.append(self.video_frame_provider.get_current_video_frame(triplet[0] - sequence_index))
            positive.append(self.video_frame_provider.get_current_video_frame(triplet[1] - sequence_index))
            negative.append(self.video_frame_provider.get_current_video_frame(triplet[2] - sequence_index))
        return np.array([anchor, positive, negative])


class AngioSequenceMultiSiameseDataset(Dataset):
    """
    Yield matrices of positive and negative pairs
    """
    def __init__(self, path, path_to_ignore, sequence, epoch_size, batch_size):
        """
        Sample most trivial training data for phase 0 (intra-video sampling of consecutive frames)

        Args:
            path (str): path in which we can find sequences of images
        """
        self.files, self.names, self.data_mean, self.data_std = get_all_valid_frames_in_paths(path, path_to_ignore)
        self.video_frame_provider = VideoFrameProvider(images=self.files, names=self.names)
        self.sequence = sequence
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self._calc_all_positive_and_negative_pairs()

    def _calc_all_positive_and_negative_pairs(self):
        self.frame_pairs = []
        video_count = self.video_frame_provider.video_count()
        for video_id in range(video_count):
            # print("video", video_id)
            self.video_frame_provider.select_video(video_id)
            video_frame_count = self.video_frame_provider.get_current_video_frame_count()
            video_frame_pairs = np.zeros((video_frame_count, video_frame_count))
            hb_freq = self.video_frame_provider.get_current_video_heartbeat_frequency()
            # print("hb_freq", hb_freq)
            min_pos_dist = round(hb_freq * 7 / 8)
            max_pos_dist = round(hb_freq / 8)
            min_neg_dist = round(hb_freq * 3 / 8)
            max_neg_dist = round(hb_freq * 5 / 8)
            # print("min_pos_dist", min_pos_dist)
            # print("max_pos_dist", max_pos_dist)
            # print("min_neg_dist", min_neg_dist)
            # print("max_neg_dist", max_neg_dist)
            for i in range(video_frame_count):
                for j in range(i+1, video_frame_count):
                    a = i % hb_freq
                    b = j % hb_freq
                    frame_diff = abs(a - b)
                    frame_diff = min(frame_diff, hb_freq - frame_diff)
                    # if frame_diff >= min_pos_dist or frame_diff <= max_pos_dist:
                    if frame_diff == 1:
                        video_frame_pairs[i][j] = video_frame_pairs[j][i] = 1
                    elif min_neg_dist <= frame_diff <= max_neg_dist:
                        video_frame_pairs[i][j] = video_frame_pairs[j][i] = -1
            # print(video_frame_pairs)
            self.frame_pairs.append(video_frame_pairs)
            # plt.imshow(video_frame_pairs)
            # plt.show()

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, item):
        return next(self.__iter__())

    def __iter__(self):
        count = 0
        while count < self.epoch_size:
            count += 1
            # Select a random video
            video_id = np.random.randint(0, self.video_frame_provider.video_count())
            self.video_frame_provider.select_video(video_id)
            # print("video_id", video_id)
            # print("fpb", self.video_frame_provider.get_current_video_heartbeat_frequency())
            # print("frame_pairs", self.frame_pairs[video_id])
            frame_count = self.video_frame_provider.get_current_video_frame_count()
            possible_frames = np.arange(self.sequence - 1, frame_count)
            possible_frames_count = len(possible_frames)

            # if there are more frames in the video than the size of our batch, we can sample from it
            if possible_frames_count > self.batch_size:
                # Randomly select frames in our video based on the batch size
                frame_indices = np.random.choice(possible_frames, self.batch_size, replace=False)
                # print("frame_indices", frame_indices)
                # print(f"frame_pairs[{frame_indices[0]}]", self.frame_pairs[video_id][frame_indices[0]])
                positive_matrix = np.zeros((self.batch_size, self.batch_size), dtype=np.float32)
                negative_matrix = np.zeros((self.batch_size, self.batch_size), dtype=np.float32)
            else:
                frame_indices = possible_frames
                positive_matrix = np.zeros((possible_frames_count, possible_frames_count), dtype=np.float32)
                negative_matrix = np.zeros((possible_frames_count, possible_frames_count), dtype=np.float32)

            frame_sequences = []
            # Compare every frame to create the positive and negative matrices
            for i, frame_a_index in enumerate(frame_indices):
                for j in range(i+1, len(frame_indices)):
                    pair = self.frame_pairs[video_id][frame_a_index, frame_indices[j]]
                    if pair == 1:  # positive pair
                        positive_matrix[i, j] = positive_matrix[j, i] = 1
                    elif pair == -1:  # negative pair
                        negative_matrix[i, j] = negative_matrix[j, i] = 1

                # Create frame sequence
                frame_sequence = []
                for sequence_index in reversed(range(self.sequence)):
                    frame_sequence.append(self.video_frame_provider.get_current_video_frame(frame_a_index - sequence_index))
                frame_sequences.append(frame_sequence)

            # yield torch.FloatTensor(frame_sequences), torch.from_numpy(positive_matrix), torch.from_numpy(negative_matrix)
            yield np.array(frame_sequences), positive_matrix, negative_matrix, self.video_frame_provider.get_current_video_name()


class AngioSequenceSoftMultiSiameseDataset(Dataset):
    """
    Yield matrices of similarity between pairs
    """
    def __init__(self, paths, path_to_ignores, sequence, max_cycles_for_pairs, epoch_size, batch_size, inter_video_pairs, use_data_augmentation, use_data_normalization, img_size=224):
        """
        Sample most trivial training data for phase 0 (intra-video sampling of consecutive frames)

        Args:
            paths (list of str): paths in which we can find sequences of images
            path_to_ignores (list of str): paths that will be ignored when searching for sequences of images
            sequence (int): number of frames to use as input for the NN (use 0 to duplicate image in RGB channels, otherwise use 3)
            max_cycles_for_pairs (float): Number of cycles to limit the pairs, the rest is masked
            epoch_size (int): Number of matrices to sample per epoch
            batch_size (int): Size of the sampled matrices
            inter_video_pairs (bool): True to pair the frames of different videos together, False to limit to only intra-video pairs
        """
        self.files, self.names, self.data_mean, self.data_std = get_all_valid_frames_in_paths(paths, path_to_ignores, img_size)
        self.video_frame_provider = VideoFrameProvider(images=self.files, names=self.names)
        self.sequence = sequence
        self.max_cycles_for_pairs = max_cycles_for_pairs
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.inter_video_pairs = inter_video_pairs
        self.use_data_augmentation = use_data_augmentation
        self.use_data_normalization = use_data_normalization
        if use_data_augmentation:
            self.data_augmentation = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.)),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), fillcolor=0)
            ])
        self.frame_pair_values, self.frame_pair_masks = calc_similarity_between_all_pairs(self.video_frame_provider, self.max_cycles_for_pairs)
        # if use_data_normalization:
        #     self.normalize = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, item):
        return next(self.__iter__())

    def __iter__(self):
        epoch_count = 0
        while epoch_count < self.epoch_size:
            epoch_count += 1
            # Select a random video
            video_id = np.random.randint(0, self.video_frame_provider.video_count())
            self.video_frame_provider.select_video(video_id)
            video_name = self.video_frame_provider.get_current_video_name()
            frame_count = self.video_frame_provider.get_current_video_frame_count()
            possible_frames = np.arange(self.sequence - 1, frame_count)
            possible_frames_count = len(possible_frames)

            if self.inter_video_pairs:
                video_id_b = np.random.randint(0, self.video_frame_provider.video_count())
                self.video_frame_provider.select_video(video_id_b)
                video_name_b = self.video_frame_provider.get_current_video_name()
                frame_count_b = self.video_frame_provider.get_current_video_frame_count()
                possible_frames_b = np.arange(self.sequence - 1, frame_count_b)
                possible_frames_count_b = len(possible_frames_b)
                # print(f"Sampled videos {video_id} and {video_id_b} with {possible_frames_count} and {possible_frames_count_b} frames")

                # if there are more frames in the video than the size of our batch, we can sample from it
                if possible_frames_count > self.batch_size:
                    # Randomly select frames in our video based on the batch size
                    frame_indices_a = np.random.choice(possible_frames, self.batch_size, replace=False)
                else:
                    frame_indices_a = possible_frames

                # if there are more frames in the video than the size of our batch, we can sample from it
                if possible_frames_count_b > self.batch_size:
                    # Randomly select frames in our video based on the batch size
                    frame_indices_b = np.random.choice(possible_frames_b, self.batch_size, replace=False)
                else:
                    frame_indices_b = possible_frames_b

                similarity_matrix = np.zeros((len(frame_indices_a), len(frame_indices_b)), dtype=np.float32)
                masks = np.zeros((len(frame_indices_a), len(frame_indices_b)), dtype=np.float32)
                # print("Creating similarity matrix of shape", similarity_matrix.shape)

                frame_sequences_a = []
                frame_sequences_b = []
                self.video_frame_provider.select_video(video_id)
                # Compare every frame to create the similarity matrix
                for i, frame_a_index in enumerate(frame_indices_a):
                    for j in range(len(frame_indices_b)):
                        frame_b_index = frame_indices_b[j]
                        if video_id <= video_id_b:
                            # print(frame_a_index, frame_b_index, self.frame_pair_values[video_id][video_id_b-video_id].shape)
                            pair_value = self.frame_pair_values[video_id][video_id_b-video_id][frame_a_index, frame_b_index]
                            pair_mask = self.frame_pair_masks[video_id][video_id_b-video_id][frame_a_index, frame_b_index]
                        else:
                            # print(frame_b_index, frame_a_index, self.frame_pair_values[video_id_b][video_id-video_id_b].shape)
                            pair_value = self.frame_pair_values[video_id_b][video_id-video_id_b][frame_b_index, frame_a_index]
                            pair_mask = self.frame_pair_masks[video_id_b][video_id-video_id_b][frame_b_index, frame_a_index]
                        similarity_matrix[i, j] = pair_value
                        masks[i, j] = pair_mask

                    # Create frame sequence
                    frame_sequence = []
                    for sequence_index in reversed(range(self.sequence)):
                        frame_sequence.append(self.video_frame_provider.get_current_video_frame(frame_a_index - sequence_index))

                    # Duplicate frame to have a gray RBG image
                    if self.sequence == 1:
                        frame_sequence.append(frame_sequence[0])
                        frame_sequence.append(frame_sequence[0])

                    # Apply data augmentation
                    if self.use_data_augmentation:
                        frame_sequence = np.array([x * 255 for x in frame_sequence], dtype=np.uint8)
                        frame_sequence = np.moveaxis(frame_sequence, 0, 2)
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(frame_sequence)
                        # plt.title("Before")
                        frame_sequence = self.data_augmentation(frame_sequence)
                        frame_sequence = np.asarray(frame_sequence, dtype=np.float32) / 255
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(frame_sequence)
                        # plt.title("After")
                        # plt.show()
                        frame_sequence = np.moveaxis(frame_sequence, 2, 0)

                    # Apply data normalization
                    if self.use_data_normalization:
                        frame_sequence -= self.data_mean
                        frame_sequence /= self.data_std
                        # frame_sequence = self.normalize(frame_sequence)

                    frame_sequences_a.append(frame_sequence)

                self.video_frame_provider.select_video(video_id_b)
                for j in range(len(frame_indices_b)):
                    frame_b_index = frame_indices_b[j]
                    # Create frame sequence
                    frame_sequence = []
                    for sequence_index in reversed(range(self.sequence)):
                        frame_sequence.append(self.video_frame_provider.get_current_video_frame(frame_b_index - sequence_index))

                    # Duplicate frame to have a gray RBG image
                    if self.sequence == 1:
                        frame_sequence.append(frame_sequence[0])
                        frame_sequence.append(frame_sequence[0])

                    # Apply data augmentation
                    if self.use_data_augmentation:
                        frame_sequence = np.array(frame_sequence)
                        frame_sequence = (frame_sequence * 255).astype(np.uint8)
                        frame_sequence = np.moveaxis(frame_sequence, 0, 2)
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(frame_sequence)
                        # plt.title("Before")
                        frame_sequence = self.data_augmentation(frame_sequence)
                        frame_sequence = np.asarray(frame_sequence)
                        # print(frame_sequence.shape)
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(frame_sequence)
                        # plt.title("After")
                        # plt.show()
                        frame_sequence = np.moveaxis(frame_sequence, 2, 0)
                        frame_sequence = frame_sequence.astype(np.float32) / 255

                    # Apply data normalization
                    if self.use_data_normalization:
                        frame_sequence -= self.data_mean
                        frame_sequence /= self.data_std
                        # frame_sequence = self.normalize(frame_sequence)

                    frame_sequences_b.append(frame_sequence)

                appended_frame_sequences = np.append(np.array(frame_sequences_a), np.array(frame_sequences_b), axis=0)
                yield appended_frame_sequences, (similarity_matrix, masks), {"frame_indices_a": frame_indices_a, "frame_indices_b": frame_indices_b, "video_name_a": video_name, "video_name_b": video_name_b}

            else:  # No inter-video pairs
                # if there are more frames in the video than the size of our batch, we can sample from it
                if possible_frames_count > self.batch_size:
                    # Randomly select frames in our video based on the batch size
                    frame_indices = np.random.choice(possible_frames, self.batch_size, replace=False)
                else:
                    frame_indices = possible_frames
                similarity_matrix = np.zeros((len(frame_indices), len(frame_indices)), dtype=np.float32)
                masks = np.zeros((len(frame_indices), len(frame_indices)), dtype=np.float32)

                frame_sequences = []
                # Compare every frame to create the similarity matrix
                for i, frame_a_index in enumerate(frame_indices):
                    for j in range(i+1, len(frame_indices)):
                        frame_b_index = frame_indices[j]
                        pair_value = self.frame_pair_values[video_id][0][frame_a_index, frame_b_index]
                        pair_mask = self.frame_pair_masks[video_id][0][frame_a_index, frame_b_index]
                        similarity_matrix[i, j] = similarity_matrix[j, i] = pair_value
                        masks[i, j] = masks[j, i] = pair_mask

                    # Create frame sequence
                    frame_sequence = []
                    for sequence_index in reversed(range(self.sequence)):
                        frame_sequence.append(self.video_frame_provider.get_current_video_frame(frame_a_index - sequence_index))

                    # Duplicate frame to have a gray RBG image
                    if self.sequence == 1:
                        frame_sequence.append(frame_sequence[0])
                        frame_sequence.append(frame_sequence[0])

                    # Apply data augmentation
                    if self.use_data_augmentation:
                        frame_sequence = np.array(frame_sequence)
                        frame_sequence = (frame_sequence * 255).astype(np.uint8)
                        frame_sequence = np.moveaxis(frame_sequence, 0, 2)
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(frame_sequence)
                        # plt.title("Before")
                        frame_sequence = self.data_augmentation(frame_sequence)
                        frame_sequence = np.asarray(frame_sequence)
                        # print(frame_sequence.shape)
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(frame_sequence)
                        # plt.title("After")
                        # plt.show()
                        frame_sequence = np.moveaxis(frame_sequence, 2, 0)
                        frame_sequence = frame_sequence.astype(np.float32) / 255

                    # Apply data normalization
                    if self.use_data_normalization:
                        frame_sequence = np.array(frame_sequence)
                        frame_sequence -= self.data_mean
                        frame_sequence /= self.data_std
                        # frame_sequence = (frame_sequence * 255).astype(np.uint8)
                        # frame_sequence = np.moveaxis(frame_sequence, 0, 2)
                        # frame_sequence = self.normalize(frame_sequence)
                        # frame_sequence = np.asarray(frame_sequence)

                    frame_sequences.append(frame_sequence)
                frame_sequences = np.array(frame_sequences)
                yield frame_sequences, (similarity_matrix, masks), {"frame_indices": frame_indices, "video_name": video_name}


class AngioSequenceTestDataset(Dataset):

    def __init__(self, paths, img_size=224, calc_ground_truth_matrices=True):
        self.files, self.names, self.data_mean, self.data_std = get_all_valid_frames_in_paths(paths, [], img_size)
        self.video_frame_provider = VideoFrameProvider(images=self.files, names=self.names)
        self.sequence_length = 3
        if calc_ground_truth_matrices:
            self.frame_pair_values, _ = calc_similarity_between_all_pairs(self.video_frame_provider)
        else:
            self.frame_pair_values = None

    def __len__(self):
        return self.video_frame_provider.video_count()

    def __getitem__(self, item):
        self.video_frame_provider.select_video(item)
        sequences = []
        frame_count = self.video_frame_provider.get_current_video_frame_count()
        for i in range(self.sequence_length - 1, frame_count):
            sequence = []
            for sequence_index in reversed(range(self.sequence_length)):
                sequence.append(self.video_frame_provider.get_current_video_frame(i - sequence_index))
            sequences.append(sequence)
        return np.array(sequences), self.video_frame_provider.get_current_video_name()

    def get_similarity_matrix(self, video_name_a, video_name_b):
        index_a = self.names.index(video_name_a)
        index_b = self.names.index(video_name_b)
        if index_a > index_b:
            temp = index_a
            index_a = index_b
            index_b = temp
        return self.frame_pair_values[index_a][index_b-index_a][2:, 2:]  # We want to skip the first two frames as they are used in the first sequence

    def get_hb_freq(self, video_name):
        index = self.names.index(video_name)
        self.video_frame_provider.select_video(index)
        return self.video_frame_provider.get_current_video_heartbeat_frequency()

    def get_r_peaks(self, video_name):
        index = self.names.index(video_name)
        self.video_frame_provider.select_video(index)
        return self.video_frame_provider.get_current_video_r_peaks()


def calc_similarity_between_all_pairs(video_frame_provider, max_cycles_for_pairs=0.):
    all_frame_pair_values = []
    all_frame_pair_masks = []
    video_count = video_frame_provider.video_count()
    for video_a_id in range(video_count):
        print(f"Computing pair similarities ({video_a_id+1}/{video_count})")
        frame_pair_values = []
        frame_pair_masks = []
        video_frame_provider.select_video(video_a_id)
        video_a_frame_count = video_frame_provider.get_current_video_frame_count()
        hb_freq_a = video_frame_provider.get_current_video_heartbeat_frequency()
        r_peaks_a = video_frame_provider.get_current_video_r_peaks()
        # Loop through all videos (even the same one)
        for video_b_id in range(video_a_id, video_count):
            video_frame_provider.select_video(video_b_id)
            video_b_frame_count = video_frame_provider.get_current_video_frame_count()
            hb_freq_b = video_frame_provider.get_current_video_heartbeat_frequency()
            r_peaks_b = video_frame_provider.get_current_video_r_peaks()
            video_frame_pair_values = np.zeros((video_a_frame_count, video_b_frame_count))
            video_frame_pair_masks = np.ones((video_a_frame_count, video_b_frame_count))

            # Loop on each frame pair
            for i in range(video_a_frame_count):
                # Compute cycle progression for frame i of video a
                cycle_progression_a = get_cycle_progression(i, r_peaks_a, video_a_frame_count)

                for j in range(video_b_frame_count):
                    # Compute cycle progression for frame j of video b
                    cycle_progression_b = get_cycle_progression(j, r_peaks_b, video_b_frame_count)

                    # Compute similarity of the pair
                    similarity = abs(cycle_progression_a - cycle_progression_b)
                    similarity = min(similarity, 1 - similarity) * 2
                    video_frame_pair_values[i, j] = 1 - similarity
                    # print(i, j, previous_r_peak, next_r_peak, video_frame_pair_values[i, j])
                    if max_cycles_for_pairs > 0 and video_a_id == video_b_id:
                        video_frame_pair_masks[i, j] = 1 if abs(i - j) <= hb_freq_a * max_cycles_for_pairs else 0

            # plt.imshow(1 - video_frame_pair_values)
            # plt.title(f"Similarity matrix for videos {video_a_id} and {video_b_id}")
            # plt.colorbar()
            # plt.show()
            frame_pair_values.append(video_frame_pair_values)
            frame_pair_masks.append(video_frame_pair_masks)

        all_frame_pair_values.append(frame_pair_values)
        all_frame_pair_masks.append(frame_pair_masks)

    return all_frame_pair_values, all_frame_pair_masks


def get_cycle_progression(frame, r_peaks, video_frame_count):
    if frame <= r_peaks[0]:
        first_diff = r_peaks[1] - r_peaks[0]
        previous_r_peak = 0 if r_peaks[0] >= first_diff else r_peaks[0] - first_diff
    else:
        previous_r_peak = r_peaks[(frame - r_peaks[np.where(frame - r_peaks > 0)[0]]).argmin()]
    if frame > r_peaks[-1]:
        last_diff = r_peaks[-1] - r_peaks[-2]
        next_r_peak = video_frame_count if video_frame_count - r_peaks[-1] >= last_diff else r_peaks[-1] + last_diff
    else:
        for r_peak in r_peaks:
            if r_peak >= frame:
                next_r_peak = r_peak
                break
    cycle_progression = (frame - previous_r_peak) / (next_r_peak - previous_r_peak)
    return cycle_progression


def get_triplets_parameters(path, path_to_ignore):
    return {
        'path': path,
        'path_to_ignore': path_to_ignore,
        'sequence': 3
    }


def get_multisiamese_datasets(training_path, validation_path, epoch_size, batch_size):
    training_set = AngioSequenceMultiSiameseDataset(training_path, validation_path, 3, epoch_size, batch_size)
    validation_set = None if validation_path is None else AngioSequenceMultiSiameseDataset(validation_path, [], 3, round(epoch_size / 10), batch_size)
    return training_set, validation_set


def get_soft_multisiamese_datasets(training_paths, validation_paths, test_paths, max_cycles_for_pairs, sequence, epoch_size, batch_size, inter_video_pairs, use_data_augmentation, use_data_normalization, img_size=224):
    training_paths = [training_paths] if not type(training_paths) == list else training_paths
    validation_paths = [validation_paths] if not type(validation_paths) == list else validation_paths
    test_paths = [test_paths] if not type(test_paths) == list else test_paths
    training_set = AngioSequenceSoftMultiSiameseDataset(training_paths, validation_paths + test_paths, sequence, max_cycles_for_pairs, epoch_size, batch_size, inter_video_pairs, use_data_augmentation, use_data_normalization, img_size=img_size)
    validation_set = None if len(validation_paths) == 0 or validation_paths[0] is None else AngioSequenceSoftMultiSiameseDataset(validation_paths, [], sequence, max_cycles_for_pairs, round(epoch_size / 5), batch_size, inter_video_pairs, use_data_augmentation=False, use_data_normalization=use_data_normalization, img_size=img_size)
    return training_set, validation_set


def get_datasets(training_path, validation_path):
    training_params = get_triplets_parameters(training_path, validation_path)
    validation_params = get_triplets_parameters(validation_path, None)
    training_set = AngioSequenceTripletDataset(**training_params)
    validation_set = AngioSequenceTripletDataset(**validation_params)
    return training_set, validation_set


def get_test_set(test_paths, img_size=224, calc_ground_truth_matrices=True):
    test_paths = [test_paths] if not type(test_paths) == list else test_paths
    return AngioSequenceTestDataset(test_paths, img_size, calc_ground_truth_matrices)


def get_frame_indices_of_most_distant_similar_pair_with_randomness(similarity_matrix, masks, real_frame_indices):
    size = similarity_matrix.shape[0]
    i = np.random.randint(0, size)
    print("i", i)
    print(similarity_matrix[i])
    possible_j = np.array([j for j in range(size) if similarity_matrix[i, j] == 1 and masks[i, j] == 1])
    if len(possible_j) == 0:
        return i, -1
    print("possible j", possible_j)
    diff = np.array([np.abs(real_frame_indices[j] - real_frame_indices[i]) for j in possible_j])
    print("diff", diff)
    j = possible_j[np.argmax(diff)]
    print(f"Most distant similar pair: ({i}, {j})")
    return i, j


def show_superimposed_frames(sequences, i, j, real_frame_indices, video_name):
    if j < 0:
        return
    frame_i = sequences[i][0]
    frame_j = sequences[j][0]
    superposed_frames = torch.stack([frame_i, frame_j, torch.zeros(frame_i.shape)], dim=-1)
    plt.imshow(superposed_frames)
    plt.title(f"Comparison of farthest similar valid frames {real_frame_indices[i]} and {real_frame_indices[j]} ({video_name})")
    plt.show()


if __name__ == '__main__':
    # training_path = r'C:\Users\root\Data\Angiographie'
    # validation_path = r'C:\Users\root\Data\Angiographie\KR-11'
    training_path = r'C:\Users\root\Data\Angiographie\P28'
    validation_path = None
    test_path = None

    # # Multisiamese
    # training_set, validation_set = get_multisiamese_datasets(training_path, validation_path, 1, 10)
    # training_dataloader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    # for i_batch, data in enumerate(training_dataloader):
    #     print(type(data))
    #     sequences = data[0][0]
    #     positive_matrix = data[1]
    #     negative_matrix = data[2]
    #     print("sequences", sequences.shape)
    #     print("positive_matrix", positive_matrix)
    #     print("negative_matrix", negative_matrix)

    # Soft Multisiamese
    test_paths = [
        r'C:\Users\root\Data\Angiographie\MJY-9',  # 2 sequences
    ]
    test_set = get_test_set(test_paths, 240)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    for batch_index, sequences in enumerate(test_loader):
        print(batch_index)
    # training_set, validation_set = get_soft_multisiamese_datasets(training_path, validation_path, test_path, max_cycles_for_pairs=0, sequence=3, epoch_size=1000, batch_size=64, inter_video_pairs=True, use_data_augmentation=True)
    # training_dataloader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    # for i_batch, data in enumerate(training_dataloader):
    #     print(type(data))
    #     sequences = data[0][0]
    #     similarity_matrix = data[1][0][0]
    #     masks = data[1][1][0]
    #     frame_indices = data[2]["frame_indices"][0]
    #     video_name = data[2]["video_name"][0]
    #     print("sequences", sequences.shape)
    #     print("similarity_matrix", similarity_matrix.shape)
    #     print("masks", masks.shape)
    #     print(frame_indices)
    #     i, j = get_frame_indices_of_most_distant_similar_pair_with_randomness(similarity_matrix, masks, frame_indices)
    #     show_superimposed_frames(sequences, i, j, frame_indices, video_name)

    # # Optical Flow tests using Soft Multisiamese
    # max_cycle_for_pairs = 0
    # training_set, validation_set = get_soft_multisiamese_datasets(training_path, validation_path, max_cycle_for_pairs, 1, 10)
    # training_dataloader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    # for i_batch, data in enumerate(training_dataloader):
    #     sequences = data[0][0]
    #
    #     # # Optical Flow 1
    #     # plt.subplot(1, 4, 1)
    #     # plt.imshow(sequences[0].permute(1, 2, 0))
    #     # plt.subplot(1, 4, 2)
    #     # u, v = optical_flow(sequences[0][0], sequences[0][2], 4)
    #     # plt.imshow(np.stack([u, v, np.zeros(u.shape)], axis=2))
    #     # plt.subplot(1, 4, 3)
    #     # u, v = optical_flow(sequences[0][0], sequences[0][2], 8)
    #     # plt.imshow(np.stack([u, v, np.zeros(u.shape)], axis=2))
    #     # plt.subplot(1, 4, 4)
    #     # u, v = optical_flow(sequences[0][0], sequences[0][2], 12)
    #     # plt.imshow(np.stack([u, v, np.zeros(u.shape)], axis=2))
    #     # plt.show()
    #
    #     # Optical Flow 2
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(sequences[0].permute(1, 2, 0))
    #     of = optical_flow2(sequences[0][0], sequences[0][2])
    #     print(of.shape)
    #     plt.imshow(of)

    # training_set, validation_set = get_datasets(training_path, validation_path)
    # training_dataloader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=4)
    # validation_dataloader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=4)
    #
    # for i_batch, sample_batched in enumerate(validation_dataloader):
    #     print(i_batch, sample_batched.size(), sample_batched.type())

    # test_set = get_test_set(training_path)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    #
    # for i, data in enumerate(test_loader):
    #     print(i, data.shape)

    # while True:
    #     index = np.random.randint(0, len(dataset))
    #     triplet = dataset.__getitem__(index)
    # # for i, triplet in enumerate(data):
    #     # vstack to get 3x3 gray images, stack to get 3 color images
    #     a = triplet[0]  # np.stack(triplet[::3])
    #     p = triplet[1]  # np.stack(triplet[1::3])
    #     n = triplet[2]  # np.stack(triplet[2::3])
    #     # move axis for cv2 to show the images correctly
    #     a = np.moveaxis(a, 0, -1)
    #     p = np.moveaxis(p, 0, -1)
    #     n = np.moveaxis(n, 0, -1)
    #     # put the images next to each other
    #     img = np.hstack([a, p, n])
    #     # img = np.moveaxis(img, 0, -1)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
