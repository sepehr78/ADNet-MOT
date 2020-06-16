from torch.autograd import Variable
import numpy as np
from utils.get_video_infos import get_video_infos, get_video_infos_adnet_mot
import cv2
from utils.do_action import do_action
from utils.overlap_ratio import overlap_ratio
import torch.nn as nn
import math
import torch
from utils.augmentations import CropRegion


def RL_loss(saved_log_probs, rewards, use_gpu=True):
    policy_loss = []
    # gamma = 0.99
    # eps = np.finfo(np.float32).eps.item()
    # R = 0
    # returns = []
    # for r in rewards:
    #     R = r + gamma * R
    #     returns.append(R)

    # returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    for prob, reward in zip(saved_log_probs, rewards):
        policy_loss.append(-prob * reward)
    policy_loss = torch.stack(policy_loss).sum()
    # policy_loss = policy_loss.unsqueeze(0).requires_grad_(True)
    if use_gpu:
        return policy_loss.cuda().requires_grad_(True)
    else:
        return policy_loss


class TrackingPolicyLoss(nn.Module):
    def __init__(self):
        super(TrackingPolicyLoss, self).__init__()

    # TODO: make sure about this calculation...
    # https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py#L68
    def forward(self, saved_log_probs, rewards, use_gpu=True):
        policy_loss = []
        # for log_prob, reward in zip(saved_log_probs, rewards):
        for idx in range(len(saved_log_probs)):
            if len(saved_log_probs) == 1:
                policy_loss.append(-saved_log_probs * rewards.type(saved_log_probs.type()))
            else:
                policy_loss.append(-saved_log_probs[idx] * rewards[idx].float())
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss = policy_loss.unsqueeze(0).requires_grad_(True)
        if use_gpu:
            return policy_loss.cuda().requires_grad_(True)
        else:
            return policy_loss


# TrackingEnvironment for all of the videos in one epoch
# Number of steps can be set in opts['train']['RL_steps'] before initializing this environment
class TrackingEnvironment(object):
    def __init__(self, train_videos, opts, transform, args):
        self.videos = []  # list of clips dict

        self.opts = opts
        self.transform = transform
        self.args = args

        self.RL_steps = self.opts['train']['RL_steps']  # clip length

        video_names = train_videos['video_names']
        video_paths = train_videos['video_paths']
        bench_names = train_videos['bench_names']

        vid_idxs = np.random.permutation(len(video_names))

        for vid_idx in vid_idxs:
            # dict consist of set of clips in ONE video
            clips = {
                'img_path': [],
                'frame_start': [],
                'frame_end': [],
                'init_bbox': [],
                'end_bbox': [],
                'vid_idx': [],
            }
            # Load current training video info
            video_name = video_names[vid_idx]
            video_path = video_paths[vid_idx]
            bench_name = bench_names[vid_idx]

            # TODO MAYBE ADD MOT
            vid_info = get_video_infos(bench_name, video_path, video_name)

            if self.RL_steps is None:
                self.RL_steps = len(vid_info['gt']) - 1
                vid_clip_starts = [0]
                vid_clip_ends = [len(vid_info['gt']) - 1]
            else:
                vid_clip_starts = np.array(range(len(vid_info['gt']) - self.RL_steps))
                vid_clip_starts = np.random.permutation(vid_clip_starts)
                vid_clip_ends = vid_clip_starts + self.RL_steps

            # number of clips in one video
            num_train_clips = min(opts['train']['rl_num_batches'], len(vid_clip_starts))

            # print("num_train_clips of vid " + str(vid_idx) + ": ", str(num_train_clips))

            for clipIdx in range(num_train_clips):
                frameStart = vid_clip_starts[clipIdx]
                frameEnd = vid_clip_ends[clipIdx]

                clips['img_path'].append(vid_info['img_files'][frameStart:frameEnd])
                clips['frame_start'].append(frameStart)
                clips['frame_end'].append(frameEnd)
                clips['init_bbox'].append(vid_info['gt'][frameStart])
                clips['end_bbox'].append(vid_info['gt'][frameEnd])
                clips['vid_idx'].append(vid_idx)

            if num_train_clips > 0:  # small hack
                self.videos.append(clips)

        self.clip_idx = -1  # hack for reset function
        self.vid_idx = 0

        self.state = None  # current bbox
        self.gt = None  # end bbox
        self.current_img = None  # current image frame
        self.current_patch = None  # current patch (transformed)
        self.current_img_idx = 0

        self.reset()

    # return state, reward, done, info. Also update the curr_patch based on the new bounding box
    # state: next bounding box
    # reward: the reward
    # done: True if finishing one clip.
    # info: a dictionary
    def step(self, action):
        info = {
            'finish_epoch': False
        }

        # do action
        self.state = do_action(self.state, self.opts, action, self.current_img.shape)
        del self.current_patch
        self.current_patch, _, _, _ = self.transform(self.current_img, self.state)

        if action == self.opts['stop_action']:
            reward, done, finish_epoch = self.go_to_next_frame()

            info['finish_epoch'] = finish_epoch

        else:  # just go to the next patch (still same frame/current_img)
            reward = 0
            done = False
            # self.current_patch, _, _, _ = self.transform(self.current_img, self.state)

        return self.state, reward, done, info

    # reset environment to new clip.
    # Return finish_epoch status: False if finish the epoch. True if still have clips remain
    def reset(self):
        while True:
            self.clip_idx += 1

            # if the clips in a video are finished... go to the next video
            if self.clip_idx >= len(self.videos[self.vid_idx]['frame_start']):
                self.vid_idx += 1
                self.clip_idx = 0
                if self.vid_idx >= len(self.videos):
                    self.vid_idx = 0
                    # one epoch finish... need to reinitialize the class to use this again randomly
                    return True

            # initialize state, gt, current_img_idx, current_img, and current_patch with new clip
            self.state = self.videos[self.vid_idx]['init_bbox'][self.clip_idx]
            self.gt = self.videos[self.vid_idx]['end_bbox'][self.clip_idx]

            frameStart = self.videos[self.vid_idx]['frame_start'][self.clip_idx]
            self.current_img_idx = 1  # self.current_img_idx = frameStart + 1
            self.current_img = cv2.imread(self.videos[self.vid_idx]['img_path'][self.clip_idx][self.current_img_idx])
            self.current_patch, _, _, _ = self.transform(self.current_img, np.array(self.state))

            if self.gt != '':  # small hack
                break

        return False

    def get_current_patch(self):
        return self.current_patch

    def get_current_train_vid_idx(self):
        return self.videos[self.vid_idx]['vid_idx'][0]

    def get_current_patch_unprocessed(self):
        crop = CropRegion()
        current_patch_unprocessed, _, _, _ = crop(self.current_img, np.array(self.state))
        return current_patch_unprocessed

    def get_state(self):
        return self.state

    def get_current_img(self):
        return self.current_img

    def go_to_next_frame(self):
        self.current_img_idx += 1
        finish_epoch = False

        # if already in the end of a clip...
        if self.current_img_idx >= len(self.videos[self.vid_idx]['img_path'][self.clip_idx]):
            # calculate reward before reset
            reward = reward_original(np.array(self.gt), np.array(self.state))

            # print("reward=" + str(reward))

            # reset (reset state, gt, current_img_idx, current_img and current_img_patch)
            finish_epoch = self.reset()  # go to the next clip (or video)

            done = True  # done means one clip is finished

        # just go to the next frame (means new patch and new image)
        else:
            reward = 0
            done = False
            # note: reset already read the current_img and current_img_patch
            self.current_img = cv2.imread(self.videos[self.vid_idx]['img_path'][self.clip_idx][self.current_img_idx])
            self.current_patch, _, _, _ = self.transform(self.current_img, self.state)

        return reward, done, finish_epoch


# Number of steps can be set in opts['train']['RL_steps'] before initializing this environment

class TrackingEnvironmentMot(object):
    def __init__(self, train_videos, opts, transform, args, num_obj_to_track=2):
        self.videos = []  # list of clips dict
        self.num_obj_to_track = num_obj_to_track

        self.opts = opts
        self.transform = transform
        self.args = args

        self.RL_steps = self.opts['train']['RL_steps']  # clip length

        video_names = train_videos['video_names']
        video_paths = train_videos['video_paths']
        bench_names = train_videos['bench_names']

        vid_idxs = np.random.permutation(len(video_names))

        for vid_idx in vid_idxs:
            # dict consist of set of clips in ONE video
            clips = {
                'img_path': [],
                'frame_start': [],
                'frame_end': [],
                'init_bbox_list': [],
                'end_bbox_list': [],
                'vid_idx': [],
            }
            # Load current training video info
            video_name = video_names[vid_idx]
            video_path = video_paths[vid_idx]
            bench_name = bench_names[vid_idx]

            # TODO MAYBE ADD MOT
            vid_info = get_video_infos_adnet_mot(bench_name, video_path, video_name, num_obj_to_track)

            if self.RL_steps is None:
                self.RL_steps = len(vid_info['gt']) - 1
                vid_clip_starts = [0]
                vid_clip_ends = [len(vid_info['gt']) - 1]
            else:
                vid_clip_starts = np.array(range(vid_info['gt_list'].shape[1] - self.RL_steps))
                vid_clip_starts = np.random.permutation(vid_clip_starts)
                vid_clip_ends = vid_clip_starts + self.RL_steps

            # number of clips in one video
            num_train_clips = min(opts['train']['rl_num_batches'], len(vid_clip_starts))

            # print("num_train_clips of vid " + str(vid_idx) + ": ", str(num_train_clips))

            for clipIdx in range(num_train_clips):
                frameStart = vid_clip_starts[clipIdx]
                frameEnd = vid_clip_ends[clipIdx]

                clips['img_path'].append(vid_info['img_files'][frameStart:frameEnd])
                clips['frame_start'].append(frameStart)
                clips['frame_end'].append(frameEnd)
                clips['init_bbox_list'].append(vid_info['gt_list'][:, frameStart])
                clips['end_bbox_list'].append(vid_info['gt_list'][:, frameEnd])
                clips['vid_idx'].append(vid_idx)

            if num_train_clips > 0:  # small hack
                self.videos.append(clips)

        self.clip_idx = -1  # hack for reset function
        self.vid_idx = 0

        self.state_list = None  # current bbox
        self.gt_list = None  # end bbox
        self.current_img = None  # current image frame
        self.current_patch_list = None  # current patch (transformed)
        self.current_img_idx = 0

        self.reset()

    # return state, reward, done, info. Also update the curr_patch based on the new bounding box
    # state: next bounding box
    # reward: the reward
    # done: True if finishing one clip.
    # info: a dictionary
    def step(self, action_list):
        info = {
            'finish_epoch': False
        }

        # do action
        for i in range(self.num_obj_to_track):
            self.state_list[i] = do_action(self.state_list[i], self.opts, action_list[i], self.current_img.shape)
        del self.current_patch_list
        self.current_patch_list = []
        for i in range(self.num_obj_to_track):
            temp, _, _, _ = self.transform(self.current_img, np.array(self.state_list[i]))
            self.current_patch_list.append(temp)

        if False and self.opts['stop_action'] in action_list:
            reward_list, done, finish_epoch = self.go_to_next_frame()

            info['finish_epoch'] = finish_epoch

        else:  # just go to the next patch (still same frame/current_img)
            reward_list = [0] * self.num_obj_to_track
            done = False
            # self.current_patch, _, _, _ = self.transform(self.current_img, self.state)

        return self.state_list, reward_list, done, info

    # reset environment to new clip.
    # Return finish_epoch status: False if finish the epoch. True if still have clips remain
    def reset(self):
        while True:
            self.clip_idx += 1

            # if the clips in a video are finished... go to the next video
            if self.clip_idx >= len(self.videos[self.vid_idx]['frame_start']):
                self.vid_idx += 1
                self.clip_idx = 0
                if self.vid_idx >= len(self.videos):
                    self.vid_idx = 0
                    # one epoch finish... need to reinitialize the class to use this again randomly
                    return True

            # initialize state, gt, current_img_idx, current_img, and current_patch with new clip
            self.state_list = self.videos[self.vid_idx]['init_bbox_list'][self.clip_idx]
            self.gt_list = self.videos[self.vid_idx]['end_bbox_list'][self.clip_idx]

            frameStart = self.videos[self.vid_idx]['frame_start'][self.clip_idx]
            self.current_img_idx = 1  # self.current_img_idx = frameStart + 1
            self.current_img = cv2.imread(self.videos[self.vid_idx]['img_path'][self.clip_idx][self.current_img_idx])
            self.current_patch_list = []
            for i in range(self.num_obj_to_track):
                temp, _, _, _ = self.transform(self.current_img, np.array(self.state_list[i]))
                self.current_patch_list.append(temp)

            # if self.gt != '':  # small hack
            #     break

        return False

    def get_current_patch(self):
        return self.current_patch_list

    def get_current_train_vid_idx(self):
        return self.videos[self.vid_idx]['vid_idx'][0]

    def get_current_patch_unprocessed(self, i):
        crop = CropRegion()
        current_patch_unprocessed, _, _, _ = crop(self.current_img, np.array(self.state_list[i]))
        return current_patch_unprocessed

    def get_state(self):
        return self.state_list

    def get_current_img(self):
        return self.current_img

    def go_to_next_frame(self):
        self.current_img_idx += 1
        finish_epoch = False

        # if already in the end of a clip...
        if self.current_img_idx >= len(self.videos[self.vid_idx]['img_path'][self.clip_idx]):
            # calculate reward before reset
            reward_arr = []
            for i in range(self.num_obj_to_track):
                reward_arr.append(reward_original(np.array(self.gt_list[i]), np.array(self.state_list[i])))

            # print("reward=" + str(reward))

            # reset (reset state, gt, current_img_idx, current_img and current_img_patch)
            finish_epoch = self.reset()  # go to the next clip (or video)

            done = True  # done means one clip is finished

        # just go to the next frame (means new patch and new image)
        else:
            reward_arr = [0] * self.num_obj_to_track
            done = False
            # note: reset already read the current_img and current_img_patch
            self.current_img = cv2.imread(self.videos[self.vid_idx]['img_path'][self.clip_idx][self.current_img_idx])
            self.current_patch_list = []
            for i in range(self.num_obj_to_track):
                temp, _, _, _ = self.transform(self.current_img, np.array(self.state_list[i]))
                self.current_patch_list.append(temp)
        return reward_arr, done, finish_epoch


def reward_original(gt, box):
    iou = overlap_ratio(gt, box)
    if iou > 0.7:
        reward = 1
    else:
        reward = -1

    return reward
