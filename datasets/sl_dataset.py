# pytorch dataset for SL learning
# matlab code (line 26-33):
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference:
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py

import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from datasets.get_train_dbs import get_train_dbs, get_train_dbs_mot
from utils.augmentations import SubtractMeans
from utils.get_video_infos import get_video_infos, get_video_infos_mot, get_video_infos_adnet_mot

class SLDatasetMot(data.Dataset):
    def __init__(self, train_db, num_obj_to_track=2, transform=None):
        self.num_obj_to_track = num_obj_to_track
        self.img_path_np_dict = None
        self.transform = transform
        self.train_db = train_db

    def __getitem__(self, index):
        # im = cv2.imread(self.train_db['img_path'][index])
        indx1,indx2 = np.unravel_index(index, (self.train_db['bboxes_list'].shape[0], self.train_db['bboxes_list'].shape[2]))
        im = self.img_path_np_dict[self.train_db['img_path'][index]]
        bbox_list = self.train_db['bboxes_list'][indx1,:,indx2,:]
        action_labels = self.train_db['labels_list'][indx1,:,indx2,:].astype(np.float32)
        score_label = self.train_db['score_labels'][index]
        # vid_idx = self.train_db['vid_idx'][index]

        patch_list = []
        for i in range(self.num_obj_to_track):
            patch, bbox_list[i], action_labels[i], score_label = self.transform(im, bbox_list[i], action_labels[i], score_label)
            patch_list.append(patch)
        # im = np.zeros((3, 112, 112), dtype=np.float32)
        return torch.stack(patch_list), bbox_list, action_labels, score_label, index

    def __len__(self):
        return len(self.train_db['img_path'])

class SLDataset(data.Dataset):
    # train_videos = get_train_videos(opts)
    # train_videos = {  # the format of train_videos
    #         'video_names' : video_names,
    #         'video_paths' : video_paths,
    #         'bench_names' : bench_names
    #     }
    def __init__(self, train_db, transform=None):
        self.img_path_np_dict = None
        self.transform = transform
        self.train_db = train_db

    def __getitem__(self, index):
        # im = cv2.imread(self.train_db['img_path'][index])
        im = self.img_path_np_dict[self.train_db['img_path'][index]]
        bbox = self.train_db['bboxes'][index]
        action_label = np.array(self.train_db['labels'][index], dtype=np.float32)
        score_label = self.train_db['score_labels'][index]
        # vid_idx = self.train_db['vid_idx'][index]

        if self.transform is not None:
            im, bbox, action_label, score_label = self.transform(im, bbox, action_label, score_label)
        # im = np.zeros((3, 112, 112), dtype=np.float32)
        return im, bbox, action_label, score_label, index

    def __len__(self):
        return len(self.train_db['img_path'])

    #########################################################
    # ADDITIONAL FUNCTIONS

    def pull_image(self, index):
        im = cv2.imread(self.train_db['img_path'][index])
        return im

    def pull_anno(self, index):
        action_label = self.train_db['labels'][index]
        score_label = self.train_db['score_labels'][index]
        return action_label, score_label


def initialize_pos_neg_dataset(train_videos, opts, transform=None, multidomain=True):
    """
    Return list of pos and list of neg dataset for each domain.
    Args:
        train_videos:
        opts:
        transform:
        multidomain:
    Returns:
        datasets_pos: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
        datasets_neg: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
    """
    num_videos = len(train_videos['video_names'])

    datasets_pos = []
    datasets_neg = []
    pos_count = 0
    neg_count = 0
    mean_subtractor = SubtractMeans()

    for vid_idx in tqdm(range(num_videos)):
        file_name_set = set()
        train_db_pos = {
            'img_path': [],  # list of string
            'bboxes': [],  # list of ndarray left top coordinate [left top width height]
            'labels': [],  # list of ndarray #action elements. One hot vector
            'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
            'vid_idx': []  # list of int. Each video (or domain) index
        }
        train_db_neg = {
            'img_path': [],  # list of string
            'bboxes': [],  # list of ndarray left top coordinate [left top width height]
            'labels': [],  # list of ndarray #action elements. One hot vector
            'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
            'vid_idx': []  # list of int. Each video (or domain) index
        }
        print("Generating dataset from video {}/{} from bench {} (current total (pos-neg): {}-{})...".format(
            vid_idx + 1, num_videos, train_videos['bench_names'][vid_idx],
            len(train_db_pos['labels']), len(train_db_neg['labels'])))

        # print("generating dataset from video " + str(vid_idx + 1) + "/" + str(num_videos) +
        #       "(current total data (pos-neg): " + str(len(train_db_pos['labels'])) +
        #       "-" + str(len(train_db_neg['labels'])) + ")")

        bench_name = train_videos['bench_names'][vid_idx]
        video_name = train_videos['video_names'][vid_idx]
        video_path = train_videos['video_paths'][vid_idx]

        vid_info = get_video_infos(bench_name, video_path, video_name)

        train_db_pos_, train_db_neg_ = get_train_dbs(vid_info, opts)
        # separate for each bboxes sample
        for sample_idx in range(len(train_db_pos_)):
            # for img_path_idx in range(len(train_db_pos_[sample_idx]['score_labels'])):
            train_db_pos['img_path'].extend(train_db_pos_[sample_idx]['img_path'])
            train_db_pos['bboxes'].extend(train_db_pos_[sample_idx]['bboxes'])
            train_db_pos['labels'].extend(train_db_pos_[sample_idx]['labels'])
            train_db_pos['score_labels'].extend(train_db_pos_[sample_idx]['score_labels'])
            train_db_pos['vid_idx'].extend(np.repeat(vid_idx, len(train_db_pos_[sample_idx]['img_path'])))

        pos_count += len(train_db_pos['labels'])
        print("Finished generating positive dataset (current total data: {})".format(pos_count))

        for sample_idx in range(len(train_db_neg_)):
            # for img_path_idx in range(len(train_db_neg_[sample_idx]['score_labels'])):
            train_db_neg['img_path'].extend(train_db_neg_[sample_idx]['img_path'])
            train_db_neg['bboxes'].extend(train_db_neg_[sample_idx]['bboxes'])
            train_db_neg['labels'].extend(train_db_neg_[sample_idx]['labels'])
            train_db_neg['score_labels'].extend(train_db_neg_[sample_idx]['score_labels'])
            train_db_neg['vid_idx'].extend(np.repeat(vid_idx, len(train_db_neg_[sample_idx]['img_path'])))

        neg_count += len(train_db_neg['labels'])
        file_name_set.update(train_db_neg['img_path'])
        file_name_set.update(train_db_pos['img_path'])

        img_path_np_dict = {}
        print('Loading images into memory...')
        for image_name in tqdm(file_name_set):
            im = cv2.imread(image_name)
            im, _, _, _ = mean_subtractor.__call__(im)
            img_path_np_dict[image_name] = im
        print("Finished generating negative dataset (current total data: {})".format(neg_count))

        dataset_pos = SLDataset(train_db_pos, transform=transform)
        dataset_pos.img_path_np_dict = img_path_np_dict
        dataset_neg = SLDataset(train_db_neg, transform=transform)
        dataset_neg.img_path_np_dict = img_path_np_dict

        if multidomain:
            datasets_pos.append(dataset_pos)
            datasets_neg.append(dataset_neg)
        else:
            datasets_pos.extend(dataset_pos)
            datasets_neg.extend(dataset_neg)

    return datasets_pos, datasets_neg


def initialize_pos_neg_dataset_mot(train_videos, opts, transform=None, multidomain=True):
    """
    Return list of pos and list of neg dataset for each domain.
    Args:
        train_videos:
        opts:
        transform:
        multidomain:
    Returns:
        datasets_pos: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
        datasets_neg: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
    """
    num_videos = len(train_videos['video_names'])

    pos_count = 0
    neg_count = 0

    datasets_pos_all = []
    datasets_neg_all = []
    mean_subtractor = SubtractMeans()
    for vid_idx in tqdm(range(num_videos)):
        train_db_pos = {
            'img_path': [],  # list of string
            'bboxes': [],  # list of ndarray left top coordinate [left top width height]
            'labels': [],  # list of ndarray #action elements. One hot vector
            'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
            'vid_idx': []  # list of int. Each video (or domain) index
        }
        train_db_neg = {
            'img_path': [],  # list of string
            'bboxes': [],  # list of ndarray left top coordinate [left top width height]
            'labels': [],  # list of ndarray #action elements. One hot vector
            'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
            'vid_idx': []  # list of int. Each video (or domain) index
        }

        # print("generating dataset from video " + str(vid_idx + 1) + "/" + str(num_videos) +
        #       "(current total data (pos-neg): " + str(len(train_db_pos['labels'])) +
        #       "-" + str(len(train_db_neg['labels'])) + ")")

        bench_name = train_videos['bench_names'][vid_idx]
        video_name = train_videos['video_names'][vid_idx]
        video_path = train_videos['video_paths'][vid_idx]

        obj_infos = get_video_infos_mot(bench_name, video_path, video_name, opts['num_obj'],
                                        opts['min_frames_visible'], opts['area_thresh'])
        datasets_pos = []
        datasets_neg = []
        file_name_set = set()
        for obj_indx, obj_info in enumerate(obj_infos):
            train_db_pos_, train_db_neg_ = get_train_dbs(obj_info, opts)
            # separate for each bboxes sample
            for sample_idx in range(len(train_db_pos_)):
                # for img_path_idx in range(len(train_db_pos_[sample_idx]['score_labels'])):
                train_db_pos['img_path'].extend(train_db_pos_[sample_idx]['img_path'])
                train_db_pos['bboxes'].extend(train_db_pos_[sample_idx]['bboxes'])
                train_db_pos['labels'].extend(train_db_pos_[sample_idx]['labels'])
                train_db_pos['score_labels'].extend(train_db_pos_[sample_idx]['score_labels'])
                train_db_pos['vid_idx'].extend(np.repeat(vid_idx, len(train_db_pos_[sample_idx]['img_path'])))

            pos_count += len(train_db_pos['labels'])
            print("Generated pos dataset from obj {}/{} (total pos data: {})".format(obj_indx, len(obj_infos),
                                                                                     pos_count))

            for sample_idx in range(len(train_db_neg_)):
                # for img_path_idx in range(len(train_db_neg_[sample_idx]['score_labels'])):
                train_db_neg['img_path'].extend(train_db_neg_[sample_idx]['img_path'])
                train_db_neg['bboxes'].extend(train_db_neg_[sample_idx]['bboxes'])
                train_db_neg['labels'].extend(train_db_neg_[sample_idx]['labels'])
                train_db_neg['score_labels'].extend(train_db_neg_[sample_idx]['score_labels'])
                train_db_neg['vid_idx'].extend(np.repeat(vid_idx, len(train_db_neg_[sample_idx]['img_path'])))

            neg_count += len(train_db_neg['labels'])
            print("Generated neg dataset from obj {}/{} (total neg data: {})".format(obj_indx, len(obj_infos),
                                                                                     neg_count))
            file_name_set.update(train_db_pos['img_path'])
            file_name_set.update(train_db_neg['img_path'])
            dataset_pos = SLDataset(train_db_pos, transform=transform)
            dataset_neg = SLDataset(train_db_neg, transform=transform)

            datasets_pos.append(dataset_pos)
            datasets_neg.append(dataset_neg)

        if multidomain:
            print('Loading images into memory...')
            img_path_np_dict = {}
            for image_name in tqdm(file_name_set):
                im = cv2.imread(image_name)
                im, _, _, _ = mean_subtractor.__call__(im)
                img_path_np_dict[image_name] = im
            for dataset_pos, dataset_neg in zip(datasets_pos, datasets_neg):
                dataset_pos.img_path_np_dict = img_path_np_dict
                dataset_neg.img_path_np_dict = img_path_np_dict

            datasets_pos_all.append(torch.utils.data.ConcatDataset(datasets_pos))
            datasets_neg_all.append(torch.utils.data.ConcatDataset(datasets_neg))
        else:  # TODO DOESNT WORK
            datasets_pos_all.extend(torch.utils.data.ConcatDataset(datasets_pos))
            datasets_neg_all.extend(torch.utils.data.ConcatDataset(datasets_neg))
    return datasets_pos_all, datasets_neg_all



def initialize_pos_neg_dataset_adnet_mot(train_videos, opts, transform=None, multidomain=True, num_obj_to_track=2):
    """
    Return list of pos and list of neg dataset for each domain.
    Args:
        train_videos:
        opts:
        transform:
        multidomain:
    Returns:
        datasets_pos: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
        datasets_neg: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
    """
    num_videos = len(train_videos['video_names'])

    datasets_pos = []
    datasets_neg = []
    pos_count = 0
    neg_count = 0
    mean_subtractor = SubtractMeans()

    for vid_idx in tqdm(range(num_videos)):
        file_name_set = set()
        train_db_pos = {
            'img_path': [],  # list of string
            'bboxes_list': [],  # list of ndarray left top coordinate [left top width height]
            'labels_list': [],  # list of ndarray #action elements. One hot vector
            'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
            'vid_idx': []  # list of int. Each video (or domain) index
        }
        train_db_neg = {
            'img_path': [],  # list of string
            'bboxes_list': [],  # list of ndarray left top coordinate [left top width height]
            'labels_list': [],  # list of ndarray #action elements. One hot vector
            'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
            'vid_idx': []  # list of int. Each video (or domain) index
        }
        print("Generating dataset from video {}/{} from bench {} (current total (pos-neg): {}-{})...".format(
            vid_idx + 1, num_videos, train_videos['bench_names'][vid_idx],
            len(train_db_pos['score_labels']), len(train_db_neg['score_labels'])))

        # print("generating dataset from video " + str(vid_idx + 1) + "/" + str(num_videos) +
        #       "(current total data (pos-neg): " + str(len(train_db_pos['labels'])) +
        #       "-" + str(len(train_db_neg['labels'])) + ")")

        bench_name = train_videos['bench_names'][vid_idx]
        video_name = train_videos['video_names'][vid_idx]
        video_path = train_videos['video_paths'][vid_idx]

        vid_info = get_video_infos_adnet_mot(bench_name, video_path, video_name, num_obj_to_track)

        train_db_pos_, train_db_neg_ = get_train_dbs_mot(vid_info, opts, num_obj_to_track)
        # separate for each bboxes sample
        for sample_idx in range(len(train_db_pos_)):
            # for img_path_idx in range(len(train_db_pos_[sample_idx]['score_labels'])):
            train_db_pos['img_path'].extend(train_db_pos_[sample_idx]['img_path'])
            train_db_pos['bboxes_list'].append(train_db_pos_[sample_idx]['bboxes_list'])
            train_db_pos['labels_list'].append(train_db_pos_[sample_idx]['labels_list'])
            train_db_pos['score_labels'].extend(train_db_pos_[sample_idx]['score_labels'])
            train_db_pos['vid_idx'].extend(np.repeat(vid_idx, len(train_db_pos_[sample_idx]['img_path'])))

        train_db_pos['bboxes_list'] = np.array(train_db_pos['bboxes_list'])
        train_db_pos['labels_list'] = np.array(train_db_pos['labels_list'])
        pos_count += len(train_db_pos['score_labels']) * num_obj_to_track
        print("Finished generating positive dataset (current total data: {})".format(pos_count))

        for sample_idx in range(len(train_db_neg_)):
            # for img_path_idx in range(len(train_db_neg_[sample_idx]['score_labels'])):
            train_db_neg['img_path'].extend(train_db_neg_[sample_idx]['img_path'])
            train_db_neg['bboxes_list'].append(train_db_neg_[sample_idx]['bboxes_list'])
            train_db_neg['labels_list'].append(train_db_neg_[sample_idx]['labels_list'])
            train_db_neg['score_labels'].extend(train_db_neg_[sample_idx]['score_labels'])
            train_db_neg['vid_idx'].extend(np.repeat(vid_idx, len(train_db_neg_[sample_idx]['img_path'])))

        train_db_neg['bboxes_list'] = np.array(train_db_neg['bboxes_list'])
        train_db_neg['labels_list'] = np.array(train_db_neg['labels_list'])

        neg_count += len(train_db_neg['score_labels']) * num_obj_to_track
        file_name_set.update(train_db_neg['img_path'])
        file_name_set.update(train_db_pos['img_path'])

        img_path_np_dict = {}
        print('Loading images into memory...')
        for image_name in tqdm(file_name_set):
            im = cv2.imread(image_name)
            im, _, _, _ = mean_subtractor.__call__(im)
            img_path_np_dict[image_name] = im
        print("Finished generating negative dataset (current total data: {})".format(neg_count))

        dataset_pos = SLDatasetMot(train_db_pos, transform=transform)
        dataset_pos.img_path_np_dict = img_path_np_dict
        dataset_neg = SLDatasetMot(train_db_neg, transform=transform)
        dataset_neg.img_path_np_dict = img_path_np_dict

        if multidomain:
            datasets_pos.append(dataset_pos)
            datasets_neg.append(dataset_neg)
        else:
            datasets_pos.extend(dataset_pos)
            datasets_neg.extend(dataset_neg)

    return datasets_pos, datasets_neg

