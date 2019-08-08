import cv2
import numpy as np
import os
import re
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import optical_flow, optical_flow2


class VideoFrameProvider(object):
    def __init__(self, images, names):
        self.current_video_id = 0
        self.type = "images"
        self.frames = [x[0] for x in images]
        self.hb_frequencies = [x[1] for x in images]
        self.names = names

    def _get_videos(self):
        return self.frames

    def video_count(self):
        return len(self._get_videos())

    def select_video(self, video_id):
        assert 0 <= video_id < self.video_count(), "'video_id' is out of bounds"
        self.current_video_id = video_id

    def get_current_video_name(self):
        return self.names[self.current_video_id]

    def get_current_video_frame_count(self):
        return len(self.frames[self.current_video_id])

    def get_current_video_frame(self, frame_id):
        assert 0 <= frame_id < self.get_current_video_frame_count(), f"'frame_id' {frame_id} is out of bounds (max is {self.get_current_video_frame_count()})"
        return self.frames[self.current_video_id][frame_id]

    def get_current_video_heartbeat_frequency(self):
        return self.hb_frequencies[self.current_video_id]


def get_all_valid_frames_in_path(base_path, path_to_ignore):
    all_valid_frames = []
    video_names = []
    relevant_frames_file_name = "relevant_frames.txt"
    for path, subfolders, files in os.walk(base_path):
        if path_to_ignore is not None and path_to_ignore in path:
            continue
        if path.split("\\")[-1] == "seg":
            continue
        if relevant_frames_file_name not in files:
            continue

        split_path = path.split("\\export\\")
        angle = split_path[1]
        patient = split_path[0].split("\\Angiographie\\")[1]
        video_names.append(patient + ' ' + angle)

        valid_frames = []
        rfile = open(path + "/" + relevant_frames_file_name, "r")
        line = rfile.readline()
        rfile.close()
        info = line.split(';')
        first_frame = int(info[0])
        last_frame = int(info[1])
        freq = int(info[2])
        for filename in files:
            if not bool(re.search('.*Frame[0-9]+\.jpg', filename)):
                continue
            frame_id = int(filename.split("Frame")[1].split('.')[0])
            if first_frame <= frame_id <= last_frame:
                img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32)
                img /= 255
                valid_frames.append(img)
        all_valid_frames.append((valid_frames, freq))
        print(len(valid_frames), "valid frames in", path)
    return all_valid_frames, video_names


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
        self.files, self.names = get_all_valid_frames_in_path(path, path_to_ignore)
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
        self.files, self.names = get_all_valid_frames_in_path(path, path_to_ignore)
        self.video_frame_provider = VideoFrameProvider(images=self.files, names=self.names)
        self.sequence = sequence
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self._calc_all_positive_and_negative_pairs()

    def _calc_all_positive_and_negative_pairs(self):
        from matplotlib import pyplot as plt
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
                    if frame_diff >= min_pos_dist or frame_diff <= max_pos_dist:
                        video_frame_pairs[i][j] = video_frame_pairs[j][i] = 1
                    elif min_neg_dist <= frame_diff <= max_neg_dist:
                        video_frame_pairs[i][j] = video_frame_pairs[j][i] = -1
            # print(video_frame_pairs)
            self.frame_pairs.append(video_frame_pairs)
            plt.imshow(video_frame_pairs)
            plt.show()

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
            yield np.array(frame_sequences), positive_matrix, negative_matrix


class AngioSequenceSoftMultiSiameseDataset(Dataset):
    """
    Yield matrices of similarity between pairs
    """
    def __init__(self, path, path_to_ignore, sequence, epoch_size, batch_size):
        """
        Sample most trivial training data for phase 0 (intra-video sampling of consecutive frames)

        Args:
            path (str): path in which we can find sequences of images
        """
        self.files, self.names = get_all_valid_frames_in_path(path, path_to_ignore)
        self.video_frame_provider = VideoFrameProvider(images=self.files, names=self.names)
        self.sequence = sequence
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self._calc_similarity_between_all_pairs()

    def _calc_similarity_between_all_pairs(self):
        self.frame_pairs = []
        video_count = self.video_frame_provider.video_count()
        for video_id in range(video_count):
            # print("video", video_id)
            self.video_frame_provider.select_video(video_id)
            video_frame_count = self.video_frame_provider.get_current_video_frame_count()
            video_frame_pairs = np.zeros((video_frame_count, video_frame_count))
            hb_freq = self.video_frame_provider.get_current_video_heartbeat_frequency()
            for i in range(video_frame_count):
                for j in range(i+1, video_frame_count):
                    a = i % hb_freq
                    b = j % hb_freq
                    frame_diff = abs(a - b)
                    frame_diff = min(frame_diff, hb_freq - frame_diff)
                    video_frame_pairs[i][j] = video_frame_pairs[j][i] = 1 - frame_diff / (hb_freq / 2)
            self.frame_pairs.append(video_frame_pairs)

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
            frame_count = self.video_frame_provider.get_current_video_frame_count()
            possible_frames = np.arange(self.sequence - 1, frame_count)
            possible_frames_count = len(possible_frames)

            # if there are more frames in the video than the size of our batch, we can sample from it
            if possible_frames_count > self.batch_size:
                # Randomly select frames in our video based on the batch size
                frame_indices = np.random.choice(possible_frames, self.batch_size, replace=False)
                similarity_matrix = np.zeros((self.batch_size, self.batch_size), dtype=np.float32)
            else:
                frame_indices = possible_frames
                similarity_matrix = np.zeros((possible_frames_count, possible_frames_count), dtype=np.float32)

            frame_sequences = []
            # Compare every frame to create the positive and negative matrices
            for i, frame_a_index in enumerate(frame_indices):
                for j in range(i+1, len(frame_indices)):
                    pair_value = self.frame_pairs[video_id][frame_a_index, frame_indices[j]]
                    similarity_matrix[i, j] = similarity_matrix[j, i] = pair_value

                # Create frame sequence
                frame_sequence = []
                for sequence_index in reversed(range(self.sequence)):
                    frame_sequence.append(self.video_frame_provider.get_current_video_frame(frame_a_index - sequence_index))
                frame_sequences.append(frame_sequence)

            yield np.array(frame_sequences), similarity_matrix


class AngioSequenceTestDataset(Dataset):

    def __init__(self, path):
        self.files, self.names = get_all_valid_frames_in_path(path, None)
        self.video_frame_provider = VideoFrameProvider(images=self.files, names=self.names)
        self.sequence_length = 3

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


# def get_dump_data():
#     indices = list(range(sequence*3))
#
#     augmentors = [
#         imgaug.Brightness(30, clip=True),
#         imgaug.GaussianNoise(sigma=10),
#         imgaug.Contrast((0.8, 1.2), clip=False),
#         imgaug.Clip(),
#         imgaug.Flip(horiz=True),
#         imgaug.Flip(vert=True),
#     ]
#
#     df = AugmentImageComponents(df, augmentors, copy=False, index=indices)
#     df = AugmentImageComponents(df, [imgaug.Rotation(7)], copy=False, index=indices)
#
#     augmentors = [
#         imgaug.ToUint8(),
#         imgaug.Resize((224, 224))
#     ]
#
#     df = AugmentImageComponents(df, augmentors, copy=False, index=indices)
#
#     return dt


def get_triplets_parameters(path, path_to_ignore):
    return {
        'path': path,
        'path_to_ignore': path_to_ignore,
        'sequence': 3
    }


# def get_initialized_triplet_selector():
#     params = get_triplets_parameters()
#     return AllTripletSelector(**params)


def get_multisiamese_datasets(training_path, validation_path, epoch_size, batch_size):
    training_set = AngioSequenceMultiSiameseDataset(training_path, validation_path, 3, epoch_size, batch_size)
    validation_set = None if validation_path is None else AngioSequenceMultiSiameseDataset(validation_path, None, 3, round(epoch_size / 10), batch_size)
    return training_set, validation_set


def get_soft_multisiamese_datasets(training_path, validation_path, epoch_size, batch_size):
    training_set = AngioSequenceSoftMultiSiameseDataset(training_path, validation_path, 3, epoch_size, batch_size)
    validation_set = None if validation_path is None else AngioSequenceSoftMultiSiameseDataset(validation_path, None, 3, round(epoch_size / 10), batch_size)
    return training_set, validation_set


def get_datasets(training_path, validation_path):
    training_params = get_triplets_parameters(training_path, validation_path)
    validation_params = get_triplets_parameters(validation_path, None)
    training_set = AngioSequenceTripletDataset(**training_params)
    validation_set = AngioSequenceTripletDataset(**validation_params)
    return training_set, validation_set


def get_test_set(test_path):
    return AngioSequenceTestDataset(test_path)


if __name__ == '__main__':
    # training_path = r'C:\Users\root\Data\Angiographie'
    # validation_path = r'C:\Users\root\Data\Angiographie\KR-11'
    training_path = r'C:\Users\root\Data\Angiographie\AA-4'
    validation_path = None

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

    # # Soft Multisiamese
    # training_set, validation_set = get_soft_multisiamese_datasets(training_path, validation_path, 1, 10)
    # training_dataloader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    # for i_batch, data in enumerate(training_dataloader):
    #     print(type(data))
    #     sequences = data[0][0]
    #     similarity_matrix = data[1]
    #     print("sequences", sequences.shape)
    #     print("similarity_matrix", similarity_matrix)

    # Optical Flow tests using Soft Multisiamese
    training_set, validation_set = get_soft_multisiamese_datasets(training_path, validation_path, 1, 10)
    training_dataloader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    for i_batch, data in enumerate(training_dataloader):
        sequences = data[0][0]

        # # Optical Flow 1
        # plt.subplot(1, 4, 1)
        # plt.imshow(sequences[0].permute(1, 2, 0))
        # plt.subplot(1, 4, 2)
        # u, v = optical_flow(sequences[0][0], sequences[0][2], 4)
        # plt.imshow(np.stack([u, v, np.zeros(u.shape)], axis=2))
        # plt.subplot(1, 4, 3)
        # u, v = optical_flow(sequences[0][0], sequences[0][2], 8)
        # plt.imshow(np.stack([u, v, np.zeros(u.shape)], axis=2))
        # plt.subplot(1, 4, 4)
        # u, v = optical_flow(sequences[0][0], sequences[0][2], 12)
        # plt.imshow(np.stack([u, v, np.zeros(u.shape)], axis=2))
        # plt.show()

        # Optical Flow 2
        plt.subplot(1, 2, 1)
        plt.imshow(sequences[0].permute(1, 2, 0))
        of = optical_flow2(sequences[0][0], sequences[0][2])
        print(of.shape)
        plt.imshow(of)

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
