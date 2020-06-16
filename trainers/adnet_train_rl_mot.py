# matlab code: https://github.com/hellbell/ADNet/blob/master/train/adnet_train_RL.m
# policy gradient in pytorch: https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
import sys

from tqdm import tqdm

from datasets.rl_dataset_mot import RLDatasetMot

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import numpy as np
from utils.get_video_infos import get_video_infos
import cv2
from utils.augmentations import ADNet_Augmentation
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
from trainers.RL_tools import TrackingEnvironment, RL_loss
import torch.optim as optim
from trainers.RL_tools import TrackingPolicyLoss
from torch.distributions import Categorical
from utils.display import display_result
import copy
from datasets.rl_dataset import RLDataset
import torch.utils.data as data
from tensorboardX import SummaryWriter
from models.ADNet import adnet


def adnet_train_rl_mot(net, domain_specific_nets, train_videos, opts, args, num_obj_to_track=2):
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print(
                "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.visualize:
        writer = SummaryWriter(log_dir=os.path.join('tensorboardx_log', args.save_file_RL))

    if args.cuda:
        # net.module.set_phase('train')
        # net.train()
        net.module.set_phase('test')
        net.eval()

        # I just make the learning rate same with SL, except that we don't train the FC7
        # optimizer = optim.Adam(net.parameters())
        optimizer = optim.Adam([
            {'params': net.module.base_network.parameters(), 'lr': 1e-5},
            {'params': net.module.fc4_5.parameters()},
            {'params': net.module.fc6.parameters()},
            {'params': net.module.fc7.parameters(), 'lr': 0}],
            lr=5e-5, weight_decay=opts['train']['weightDecay'])
    else:
        net.set_phase('train')

        # I just make the learning rate same with SL, except that we don't train the FC7
        optimizer = optim.SGD([
            {'params': net.base_network.parameters(), 'lr': 1e-4},
            {'params': net.fc4_5.parameters()},
            {'params': net.fc6.parameters()},
            {'params': net.fc7.parameters(), 'lr': 0}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    clip_idx_epoch = 0
    # prev_net = copy.deepcopy(net)
    dataset = RLDatasetMot(net, domain_specific_nets, train_videos, opts, args)
    reward_arr = []

    # TODO DELETE TESTING
    for i in tqdm(range(10)):
        reward_arr.append(np.mean(dataset.reward_list))
        dataset.reset(net, domain_specific_nets, train_videos, opts, args)
    print("Mean reward: {}".format(np.mean(reward_arr)))
    sys.exit(0)
    total_it = 0
    running_reward = 10
    # data_loader = data.DataLoader(dataset, len(dataset), num_workers=0, shuffle=False, pin_memory=False)
    for epoch in tqdm(range(args.start_epoch, opts['numEpoch'])):
        if epoch != args.start_epoch:
            # prev_net = copy.deepcopy(net)  # save the not updated net for generating data
            dataset.reset(net, domain_specific_nets, train_videos, opts, args)
            # create batch iterator
            # data_loader = data.DataLoader(dataset, len(dataset), num_workers=0, shuffle=False, pin_memory=False)
        # batch_iterator = iter(data_loader)

        # epoch_size = len(dataset) // opts['minibatch_size']   # 1 epoch, how many iterations

        total_loss_epoch = 0
        total_reward_epoch = 0
        for iteration, (log_probs, reward, vid_idx) in [(0, (dataset.log_probs_list, dataset.reward_list, dataset.vid_idx_list))]:
            # load train data
            # action, action_prob, log_probs, reward, patch, action_dynamic, result_box = next(batch_iterator)
            # log_probs, reward, vid_idx = next(batch_iterator)
            log_probs = torch.cat(log_probs).to('cuda', non_blocking=True)
            reward = torch.LongTensor(reward).to('cuda', non_blocking=True)
            vid_idx = torch.LongTensor(vid_idx).long().cpu()

            # train
            tic = time.time()

            # find out the unique value in vid_idx
            vid_idx_unique = vid_idx.unique()

            reward_sum = 0
            # separate the batch with each video idx
            for vid_id_unique in vid_idx_unique:
                # index where vid_idx is equal to the value
                idx_vid_idx = (vid_idx == vid_id_unique).nonzero().squeeze()

                # if args.cuda:
                #     net.module.load_domain_specific(domain_specific_nets[vid_id_unique])
                # else:
                #     net.load_domain_specific(domain_specific_nets[vid_id_unique])

                optimizer.zero_grad()
                # loss = criterion(tracking_scores, num_frame, num_step_history, action_prob_history)
                loss = RL_loss(log_probs[idx_vid_idx], reward[idx_vid_idx])
                loss.backward()
                optimizer.step()  # update
                reward_sum_ = reward.sum().item()
                reward_sum += reward_sum_
                del log_probs
                del reward
                del vid_idx

                # optimizer.zero_grad()

                # save the ADNetDomainSpecific back to their module
                if args.cuda:
                    domain_specific_nets[vid_id_unique].load_weights_from_adnet(net.module)
                else:
                    domain_specific_nets[vid_id_unique].load_weights_from_adnet(net)

            reward_sum = reward_sum / len(vid_idx_unique)

            toc = time.time() - tic
            # print('epoch ' + str(epoch) + ' - iteration ' + str(iteration) + ' - train time: ' + str(toc) + " s")

            if args.visualize:
                writer.add_scalar('data/iter_reward_sum', reward_sum, total_it)
                writer.add_scalar('data/iter_loss', loss, total_it)

            # if iteration % 1000 == 0:
            #     torch.save({
            #         'epoch': epoch,
            #         'adnet_state_dict': net.state_dict(),
            #         'adnet_domain_specific_state_dict': domain_specific_nets,
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }, os.path.join(args.save_folder, args.save_file_RL) + '_epoch' + repr(epoch) + '_iter' + repr(iteration) +'.pth')

            total_loss_epoch += loss
            total_reward_epoch += reward_sum * 1.0
            clip_idx_epoch += 1
            total_it += 1

        running_reward = 0.05 * total_reward_epoch / len(dataset) + (1 - 0.05) * running_reward
        if args.visualize:
            writer.add_scalar('data/epoch_reward_ave', 1.0 * total_reward_epoch / len(dataset), epoch)
            writer.add_scalar('data/epoch_loss', total_loss_epoch / len(dataset), epoch)
        writer.add_scalar('data/running_reward', running_reward, epoch)

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'adnet_state_dict': net.state_dict(),
                'adnet_domain_specific_state_dict': domain_specific_nets,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_folder, args.save_file_RL) + 'epoch' + repr(epoch) + '.pth')

    torch.save({
        'epoch': epoch,
        'adnet_state_dict': net.state_dict(),
        'adnet_domain_specific_state_dict': domain_specific_nets,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(os.path.join(args.save_folder, args.save_file_RL) + '.pth'))

    return net

# # test the module
# from models.ADNet import adnet
# from utils.get_train_videos import get_train_videos
# from options.general import opts
# import argparse
#
# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")
#
# parser = argparse.ArgumentParser(
#     description='ADNet training')
# parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
# parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
# parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
# parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
# parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
# parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
# parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
# parser.add_argument('--save_folder', default='weights', help='Location to save checkpoint models')
#
# parser.add_argument('--save_file', default='ADNet_RL_', type=str, help='save file part of file name')
# parser.add_argument('--start_epoch', default=0, type=int, help='Begin counting epochs starting from this value')
#
# parser.add_argument('--run_supervised', default=True, type=str2bool, help='Whether to run supervised learning or not')
#
# args = parser.parse_args()
#
# opts['minibatch_size'] = 32  # TODO: still don't know what this parameter for....
#
# net = adnet(opts)
# train_videos = get_train_videos(opts)
# adnet_train_rl(net, train_videos, opts, args)
