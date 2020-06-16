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


def adnet_train_sl_only_score(args, opts, mot):
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

    if args.cuda:
        optimizer = optim.Adam([
            {'params': net.module.base_network.parameters(), 'lr': 1e-4},
            {'params': net.module.fc4_5.parameters()},
            {'params': net.module.fc6.parameters()},
            {'params': net.module.fc7.parameters()}],  # as action dynamic is zero, it doesn't matter
            lr=1e-3, weight_decay=opts['train']['weightDecay'])
    else:
        optimizer = optim.SGD([
            {'params': net.base_network.parameters(), 'lr': 1e-4},
            {'params': net.fc4_5.parameters()},
            {'params': net.fc6.parameters()},
            {'params': net.fc7.parameters()}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    if args.resume:
        # net.load_weights(args.resume)
        checkpoint = torch.load(args.resume)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    net.train()

    if not args.resume:
        print('Initializing weights...')

        if args.cuda:
            norm_std = 0.01
            # fc 4
            nn.init.normal_(net.module.fc4_5[0].weight.data, std=norm_std)
            net.module.fc4_5[0].bias.data.fill_(0.1)
            # fc 5
            nn.init.normal_(net.module.fc4_5[3].weight.data, std=norm_std)
            net.module.fc4_5[3].bias.data.fill_(0.1)

            # fc 6
            nn.init.normal_(net.module.fc6.weight.data, std=norm_std)
            net.module.fc6.bias.data.fill_(0)
            # fc 7
            nn.init.normal_(net.module.fc7.weight.data, std=norm_std)
            net.module.fc7.bias.data.fill_(0)
        else:
            scal = torch.Tensor([0.01])
            # fc 4
            nn.init.normal_(net.fc4_5[0].weight.data)
            net.fc4_5[0].weight.data = net.fc4_5[0].weight.data * scal.expand_as(net.fc4_5[0].weight.data)
            net.fc4_5[0].bias.data.fill_(0.1)
            # fc 5
            nn.init.normal_(net.fc4_5[3].weight.data)
            net.fc4_5[3].weight.data = net.fc4_5[3].weight.data * scal.expand_as(net.fc4_5[3].weight.data)
            net.fc4_5[3].bias.data.fill_(0.1)
            # fc 6
            nn.init.normal_(net.fc6.weight.data)
            net.fc6.weight.data = net.fc6.weight.data * scal.expand_as(net.fc6.weight.data)
            net.fc6.bias.data.fill_(0)
            # fc 7
            nn.init.normal_(net.fc7.weight.data)
            net.fc7.weight.data = net.fc7.weight.data * scal.expand_as(net.fc7.weight.data)
            net.fc7.bias.data.fill_(0)

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

    batch_iterators_pos_train = []
    batch_iterators_neg_train = []

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

    data_loaders_pos_train = []
    data_loaders_pos_val = []

    data_loaders_neg_train = []
    data_loaders_neg_val = []

    for dataset_pos in datasets_pos:
        num_val = int(0.1 * len(dataset_pos))
        num_train = len(dataset_pos) - num_val
        train, valid = torch.utils.data.random_split(dataset_pos, [num_train, num_val])
        data_loaders_pos_train.append(
            data.DataLoader(train, opts['minibatch_size'], num_workers=1, shuffle=True, pin_memory=True))
        data_loaders_pos_val.append(
            data.DataLoader(valid, opts['minibatch_size'], num_workers=0, shuffle=True, pin_memory=False))
    for dataset_neg in datasets_neg:
        num_val = int(0.1 * len(dataset_neg))
        num_train = len(dataset_neg) - num_val
        train, valid = torch.utils.data.random_split(dataset_neg, [num_train, num_val])
        data_loaders_neg_train.append(
            data.DataLoader(train, opts['minibatch_size'], num_workers=1, shuffle=True, pin_memory=True))
        data_loaders_neg_val.append(
            data.DataLoader(valid, opts['minibatch_size'], num_workers=0, shuffle=True, pin_memory=False))

    epoch = args.start_epoch
    if epoch != 0 and args.start_iter == 0:
        start_iter = epoch * epoch_size
    else:
        start_iter = args.start_iter

    which_dataset = list(np.full(epoch_size_pos, fill_value=1))
    which_dataset.extend(np.zeros(epoch_size_neg, dtype=int))
    shuffle(which_dataset)
    which_dataset = torch.Tensor(which_dataset).cuda()

    which_domain = np.random.permutation(number_domain)

    action_loss_tr = 0
    score_loss_tr = 0

    action_loss_val = 0
    score_loss_val = 0

    # training loop
    time_arr = np.zeros(10)
    for iteration in tqdm(range(start_iter, max_iter)):
        t0 = time.time()
        if args.multidomain:
            curr_domain = which_domain[iteration % len(which_domain)]
        else:
            curr_domain = 0
        # if new epoch (not including the very first iteration)
        if (iteration != start_iter) and (iteration % epoch_size == 0):
            epoch += 1
            shuffle(which_dataset)
            np.random.shuffle(which_domain)

            print('Saving state, epoch: {}'.format(epoch))
            domain_specific_nets_state_dict = []
            for domain_specific_net in domain_specific_nets:
                domain_specific_nets_state_dict.append(domain_specific_net.state_dict())

            torch.save({
                'epoch': epoch,
                'adnet_state_dict': net.state_dict(),
                'adnet_domain_specific_state_dict': domain_specific_nets,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_folder, args.save_file) +
               '_epoch' + repr(epoch) + '.pth')

            if args.visualize:
                writer.add_scalars('data/epoch_loss', {'action_loss_tr': action_loss_tr / epoch_size,
                                                       'score_loss_tr': score_loss_tr / epoch_size,
                                                       'total_tr': (action_loss_tr + score_loss_tr) / epoch_size},
                                   global_step=epoch)
                writer.add_scalars('data/epoch_loss', {'action_loss_val': action_loss_val / epoch_size,
                                                       'score_loss_val': score_loss_val / epoch_size,
                                                       'total_val': (action_loss_val + score_loss_val) / epoch_size},
                                   global_step=epoch)

            # reset epoch loss counters
            action_loss_tr = 0
            score_loss_tr = 0

            action_loss_val = 0
            score_loss_val = 0

        # if new epoch (including the first iteration), initialize the batch iterator
        # or just resuming where batch_iterator_pos and neg haven't been initialized
        if len(batch_iterators_pos_train) == 0 or len(batch_iterators_neg_train) == 0:
            # create batch iterator
            for data_loader_pos in data_loaders_pos_train:
                batch_iterators_pos_train.append(iter(data_loader_pos))
            # for data_loader_pos in data_loaders_pos_val:
            #     batch_iterators_pos_val.append(iter(data_loader_pos))

            for data_loader_neg in data_loaders_neg_train:
                batch_iterators_neg_train.append(iter(data_loader_neg))
            # for data_loader_neg in data_loaders_neg_val:
            #     batch_iterators_neg_val.append(iter(data_loader_neg))

        # if not batch_iterators_pos_train[curr_domain]:
        #     # create batch iterator
        #     batch_iterators_pos_train[curr_domain] = iter(data_loaders_pos_train[curr_domain])
        #
        # if not batch_iterators_neg_train[curr_domain]:
        #     # create batch iterator
        #     batch_iterators_neg_train[curr_domain] = iter(data_loaders_neg_train[curr_domain])

        # load train data
        if which_dataset[iteration % len(which_dataset)]:  # if positive
            try:
                images, bbox, action_label, score_label, vid_idx = next(batch_iterators_pos_train[curr_domain])
            except StopIteration:
                batch_iterators_pos_train[curr_domain] = iter(data_loaders_pos_train[curr_domain])
                images, bbox, action_label, score_label, vid_idx = next(batch_iterators_pos_train[curr_domain])
        else:
            try:
                images, bbox, action_label, score_label, vid_idx = next(batch_iterators_neg_train[curr_domain])
            except StopIteration:
                batch_iterators_neg_train[curr_domain] = iter(data_loaders_neg_train[curr_domain])
                images, bbox, action_label, score_label, vid_idx = next(batch_iterators_neg_train[curr_domain])

        # TODO: check if this requires grad is really false like in Variable
        if args.cuda:
            images = images.to('cuda', non_blocking=True)
            # bbox = torch.Tensor(bbox.cuda())
            action_label = action_label.to('cuda', non_blocking=True)
            score_label = score_label.float().to('cuda', non_blocking=True)

        else:
            images = torch.Tensor(images)
            bbox = torch.Tensor(bbox)
            action_label = torch.Tensor(action_label)
            score_label = torch.Tensor(score_label)

        # load ADNetDomainSpecific with video index
        if args.cuda:
            net.module.load_domain_specific(domain_specific_nets[curr_domain])
        else:
            net.load_domain_specific(domain_specific_nets[curr_domain])

        # TRAIN
        net.train()
        action_out, score_out = net(images)

        # backprop
        optimizer.zero_grad()
        if which_dataset[iteration % len(which_dataset)]:  # if positive
            accuracy = int(action_label.argmax(axis=1).eq(action_out.argmax(axis=1)).sum()) / len(action_label)
        else:
            accuracy = -1
        score_l = score_criterion(score_out, score_label.reshape(-1, 1))
        loss = score_l
        loss.backward()
        optimizer.step()

        score_loss_tr += score_l.item()

        # save the ADNetDomainSpecific back to their module
        if args.cuda:
            domain_specific_nets[curr_domain].load_weights_from_adnet(net.module)
        else:
            domain_specific_nets[curr_domain].load_weights_from_adnet(net)

        if args.visualize:
            writer.add_scalars('data/iter_loss', {'score_loss_tr': score_l.item()},
                               global_step=iteration)
            if accuracy >= 0:
                writer.add_scalars('data/iter_acc', {'accuracy_tr': accuracy},
                                   global_step=iteration)

        # VAL
        if False and iteration % (epoch_size // 2) == 0:
            net.eval()
            score_loss_val = 0
            accuracy = 0
            for temp in [data_loaders_pos_val[curr_domain], data_loaders_neg_val[curr_domain]]:
                for images, bbox, action_label, score_label, _ in temp:
                    if args.cuda:
                        images = images.to('cuda', non_blocking=True)
                        # bbox = torch.Tensor(bbox.cuda())
                        action_label = action_label.to('cuda', non_blocking=True)
                        score_label = score_label.float().to('cuda', non_blocking=True)

                    else:
                        images = torch.Tensor(images)
                        bbox = torch.Tensor(bbox)
                        action_label = torch.Tensor(action_label)
                        score_label = torch.Tensor(score_label)

                    # forward
                    action_out, score_out = net(images)

                    if which_dataset[iteration % len(which_dataset)]:  # if positive
                        accuracy += int(action_label.argmax(axis=1).eq(action_out.argmax(axis=1)).sum()) / len(action_label)
                    else:
                        accuracy = -1
                    score_l = score_criterion(score_out, score_label.reshape(-1, 1))
                    score_loss_val += score_l.item()

                    if args.visualize:
                        writer.add_scalars('data/iter_loss', {'score_loss_val': score_loss_val },
                                           global_step=iteration)
                        if accuracy >= 0:
                            writer.add_scalars('data/iter_acc', {'accuracy_val': accuracy},
                                               global_step=iteration)

        t1 = time.time()
        time_arr[iteration % 10] = t1 - t0

        if iteration % 10 == 0:
            # print('Avg. 10 iter time: %.4f sec.' % time_arr.sum())
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')
            if args.visualize and args.send_images_to_visualization:
                random_batch_index = np.random.randint(images.size(0))
                writer.add_image('image', images.data[random_batch_index].cpu().numpy(), random_batch_index)

        if args.visualize:
            writer.add_scalars('data/time', {'time_10_it': time_arr.sum()}, global_step=iteration)

            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                writer.add_scalars('data/epoch_loss', {'action_loss': action_loss_tr,
                                                       'score_loss': score_loss_tr,
                                                       'total': (action_loss_tr + score_loss_tr)}, global_step=epoch)

        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)

            domain_specific_nets_state_dict = []
            for domain_specific_net in domain_specific_nets:
                domain_specific_nets_state_dict.append(domain_specific_net.state_dict())

            torch.save({
                'epoch': epoch,
                'adnet_state_dict': net.state_dict(),
                'adnet_domain_specific_state_dict': domain_specific_nets,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_folder, args.save_file) +
               repr(iteration) + '_epoch' + repr(epoch) + '.pth')

    # final save
    torch.save({
        'epoch': epoch,
        'adnet_state_dict': net.state_dict(),
        'adnet_domain_specific_state_dict': domain_specific_nets,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.save_folder, args.save_file) + '.pth')

    return net, domain_specific_nets, train_videos
