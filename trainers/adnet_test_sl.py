import cv2
# matlab code:
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference: https://github.com/amdegroot/ssd.pytorch/blob/master/train.py

import sys

from tqdm import tqdm

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from models.ADNet import adnet
from utils.get_train_videos import get_train_videos
from datasets.sl_dataset import initialize_pos_neg_dataset_mot, initialize_pos_neg_dataset
from utils.augmentations import ADNet_Augmentation

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from random import shuffle

import os
import time
import numpy as np

from tensorboardX import SummaryWriter


def adnet_test_sl(args, opts, mot):
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
        writer = SummaryWriter(log_dir=os.path.join('tensorboardx_log', args.save_file))

    train_videos = get_train_videos(opts)
    opts['num_videos'] = len(train_videos['video_names'])

    net, domain_specific_nets = adnet(opts=opts, trained_file=args.resume, multidomain=args.multidomain)

    if args.cuda:
        net = nn.DataParallel(net)
        cudnn.benchmark = True

        net = net.cuda()

    net.eval()

    action_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.BCELoss()

    print('generating Supervised Learning dataset..')
    # dataset = SLDataset(train_videos, opts, transform=
    if mot:
        datasets_pos, datasets_neg = initialize_pos_neg_dataset_mot(train_videos, opts,
                                                                    transform=ADNet_Augmentation(opts))
    else:
        datasets_pos, datasets_neg = initialize_pos_neg_dataset(train_videos, opts, transform=ADNet_Augmentation(opts))
    number_domain = opts['num_videos']
    assert number_domain == len(datasets_pos), "Num videos given in opts is incorrect! It should be {}".format(
        len(datasets_neg))

    batch_iterators_pos_val = []
    batch_iterators_neg_val = []

    # calculating number of data
    len_dataset_pos = 0
    len_dataset_neg = 0
    for dataset_pos in datasets_pos:
        len_dataset_pos += len(dataset_pos)
    for dataset_neg in datasets_neg:
        len_dataset_neg += len(dataset_neg)

    epoch_size_pos = len_dataset_pos // opts['minibatch_size']
    epoch_size_neg = len_dataset_neg // opts['minibatch_size']
    epoch_size = epoch_size_pos + epoch_size_neg  # 1 epoch, how many iterations
    print("1 epoch = " + str(epoch_size) + " iterations")

    max_iter = opts['numEpoch'] * epoch_size
    print("maximum iteration = " + str(max_iter))

    data_loaders_pos_val = []
    data_loaders_neg_val = []

    for dataset_pos in datasets_pos:
        data_loaders_pos_val.append(
            data.DataLoader(dataset_pos, opts['minibatch_size'], num_workers=2, shuffle=True, pin_memory=True))
    for dataset_neg in datasets_neg:
        data_loaders_neg_val.append(
            data.DataLoader(dataset_neg, opts['minibatch_size'], num_workers=2, shuffle=True, pin_memory=True))

    net.eval()

    for curr_domain in range(number_domain):
        accuracy = []
        action_loss_val = []
        score_loss_val = []

        # load ADNetDomainSpecific with video index
        if args.cuda:
            net.module.load_domain_specific(domain_specific_nets[curr_domain])
        else:
            net.load_domain_specific(domain_specific_nets[curr_domain])
        for i, temp in enumerate([data_loaders_pos_val[curr_domain], data_loaders_neg_val[curr_domain]]):
            dont_show = False
            for images, bbox, action_label, score_label, indices in tqdm(temp):
                images = images.to('cuda', non_blocking=True)
                action_label = action_label.to('cuda', non_blocking=True)
                score_label = score_label.float().to('cuda', non_blocking=True)

                # forward
                action_out, score_out = net(images)

                if i == 0:  # if positive
                    action_l = action_criterion(action_out, torch.max(action_label, 1)[1])
                    action_loss_val.append(action_l.item())
                    accuracy.append(int(action_label.argmax(axis=1).eq(action_out.argmax(axis=1)).sum()) / len(action_label))

                score_l = score_criterion(score_out, score_label.reshape(-1, 1))
                score_loss_val.append(score_l.item())

                if args.display_images and not dont_show:
                    if i == 0:
                        dataset = datasets_pos[curr_domain]
                        color = (0, 255, 0)
                        conf = 1
                    else:
                        dataset = datasets_neg[curr_domain]
                        color = (0, 0, 255)
                        conf = 0
                    for j, index in enumerate(indices):
                        im = cv2.imread(dataset.train_db['img_path'][index])
                        bbox = dataset.train_db['bboxes'][index]
                        action_label = np.array(dataset.train_db['labels'][index])
                        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

                        print("\n\nTarget actions: {}".format(action_label.argmax()))
                        print("Predicted actions: {}".format(action_out.data[j].argmax()))

                        print("Target conf: {}".format(conf))
                        print("Predicted conf: {}".format(score_out.data[j]))
                        # print("Score loss: {}".format(score_l.item()))
                        # print("Action loss: {}".format(action_l.item()))
                        cv2.imshow("Test", im)
                        key = cv2.waitKey(0) & 0xFF

                        # if the `q` key was pressed, break from the loop
                        if key == ord("q"):
                            dont_show = True
                            break
                        elif key == ord("s"):
                            cv2.imwrite("vid {} t:{} p:{} c:{}.png".format(curr_domain, action_label.argmax(),
                                                                    action_out.data[i].argmax(), score_out.data[i].item()), im)

        print("Vid. {}".format(curr_domain))
        print("\tAccuracy: {}".format(np.mean(accuracy)))
        print("\tScore loss: {}".format(np.mean(score_loss_val)))
        print("\tAction loss: {}".format(np.mean(action_loss_val)))


    sys.exit(0)
    return net, domain_specific_nets, train_videos
