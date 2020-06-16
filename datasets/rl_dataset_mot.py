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
from datasets.get_train_dbs import get_train_dbs
from utils.get_video_infos import get_video_infos

import time
from trainers.RL_tools import TrackingEnvironment, RL_loss, TrackingEnvironmentMot
from utils.augmentations import ADNet_Augmentation
from utils.display import display_result, draw_box
from torch.distributions import Categorical


class RLDatasetMot(data.Dataset):

    def __init__(self, net, domain_specific_nets, train_videos, opts, args, num_obj_to_track=2):
        self.num_obj_to_track = num_obj_to_track
        self.env = None

        # these lists won't include the ground truth
        self.action_list = []  # a_t,l  # argmax of self.action_prob_list
        self.action_prob_list = []  # output of network (fc6_out)
        self.log_probs_list = []  # log probs from each self.action_prob_list member
        self.reward_list = []  # tracking score
        self.patch_list = []  # input of network
        self.action_dynamic_list = []  # action_dynamic used for inference (means before updating the action_dynamic)
        self.result_box_list = []
        self.vid_idx_list = []

        self.reset(net, domain_specific_nets, train_videos, opts, args)

    def __getitem__(self, index):

        # return self.action_list[index], self.action_prob_list[index], self.log_probs_list[index], \
        #        self.reward_list[index], self.patch_list[index], self.action_dynamic_list[index], \
        #        self.result_box_list[index]

        # TODO: currently only log_probs_list, reward_list, and vid_idx_list contains data
        return self.log_probs_list[index], self.reward_list[index], self.vid_idx_list[index]

    def __len__(self):
        return len(self.log_probs_list)

    def reset(self, net, domain_specific_nets, train_videos, opts, args):
        self.action_list = []  # a_t,l  # argmax of self.action_prob_list
        self.action_prob_list = []  # output of network (fc6_out)
        self.log_probs_list = []  # log probs from each self.action_prob_list member
        self.reward_list = []  # tracking score
        self.patch_list = []  # input of network
        self.action_dynamic_list = []  # action_dynamic used for inference (means before updating the action_dynamic)
        self.result_box_list = []
        self.vid_idx_list = []

        # print('Generating reinforcement learning dataset...')
        transform = ADNet_Augmentation(opts, True)

        self.env = TrackingEnvironmentMot(train_videos, opts, transform, args)
        clip_idx = 0
        while True:  # for every clip (l)
            tic = time.time()
            num_step_history = []  # T_l

            num_frame = 1  # the first frame won't be tracked..
            t = 0
            box_history_clip = []  # for checking oscillation in a clip

            if args.cuda:
                net.module.reset_action_dynamic()
            else:
                net.reset_action_dynamic()  # action dynamic should be in a clip (what makes sense...)

            while True:  # for every frame in a clip (t)

                if args.display_images:
                    im_with_bb = display_result(self.env.get_current_img(), self.env.get_state())
                    for i in range(self.num_obj_to_track):
                        cv2.imshow('patch_{}'.format(i), self.env.get_current_patch_unprocessed(i))
                    cv2.waitKey(1)
                elif args.save_result_images: # TODO FIX IT
                    im_with_bb = draw_box(self.env.get_current_img(), self.env.get_state())
                    cv2.imwrite('images/' + str(clip_idx) + '-' + str(t) + '.jpg', im_with_bb)

                curr_patch_list = self.env.get_current_patch()
                if args.cuda:
                    curr_patch_list = torch.stack(curr_patch_list).to('cuda', non_blocking=True)

                # self.patch_list.append(curr_patch.cpu().data.numpy())  # TODO: saving patch takes cuda memory

                # TODO: saving action_dynamic takes cuda memory
                # if args.cuda:
                #     self.action_dynamic_list.append(net.module.get_action_dynamic())
                # else:
                #     self.action_dynamic_list.append(net.get_action_dynamic())

                curr_patch_list = curr_patch_list.unsqueeze(0)  # 1 batch input [1, curr_patch.shape]

                # load ADNetDomainSpecific with video index
                if args.multidomain:
                    vid_idx = self.env.get_current_train_vid_idx()
                else:
                    vid_idx = 0
                if args.cuda:
                    net.module.load_domain_specific(domain_specific_nets[vid_idx])
                else:
                    net.load_domain_specific(domain_specific_nets[vid_idx])

                action_probs, fc7_out = net.forward(curr_patch_list, update_action_dynamic=True)

                action_probs = torch.nn.Sigmoid()(action_probs)
                action1 = action_probs[0,0:11].argmax().item()
                action2 = action_probs[0, 11:22].argmax().item()

                new_state_list, reward_list, done, info = self.env.step([action1, action2])

                # action = action_probs.argmax().item()

                # m = Categorical(probs=fc6_out)
                # action_ = m.sample()  # action and action_ are same value. Only differ in the type (int and tensor)
                # self.log_probs_list.append(log_prob)
                self.vid_idx_list.append(vid_idx)
                # net.module.add_action_to_hist(action)
                # self.action_list.append(action)
                # TODO: saving action_prob_list takes cuda memory
                # self.action_prob_list.append(action_prob)


                # loss = RL_loss(m.log_prob(sampled_action), torch.Tensor([reward]))
                # loss.backward(retain_graph=True)

                # check oscilating
                # if any((np.array(new_state).round() == x).all() for x in np.array(box_history_clip).round()):
                #     action = opts['stop_action']
                #     reward, done, finish_epoch = self.env.go_to_next_frame()
                #     info['finish_epoch'] = finish_epoch

                # check if number of action is already too much
                if t > opts['num_action_step_max']:
                    action = opts['stop_action']
                    reward_list, done, finish_epoch = self.env.go_to_next_frame()
                    info['finish_epoch'] = finish_epoch

                # TODO: saving result_box takes cuda memory
                # self.result_box_list.append(list(new_state))
                # box_history_clip.append(new_state)

                t += 1

                # if action == opts['stop_action']:
                #     num_frame += 1
                #     num_step_history.append(t)
                #     t = 0



                if done:  # if finish the clip
                    break

            tracking_scores_size = np.array(num_step_history).sum()
            tracking_scores = np.full(tracking_scores_size, reward_list)  # seems no discount factor whatsoever

            self.reward_list.extend(tracking_scores)
            # self.reward_list.append(tracking_scores)
            toc = time.time() - tic
            # print('forward time (clip ' + str(clip_idx) + " - t " + str(t) + ") = "
            #       + str(toc) + " s")
            clip_idx += 1

            if info['finish_epoch']:
                break

        # print('generating reinforcement learning dataset finish')
