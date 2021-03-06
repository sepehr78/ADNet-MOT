from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.legacy import nn as nnl
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn

import numpy as np
import os

from utils.get_action_history_onehot import get_action_history_onehot

from models.vggm import vggm

__all__ = ['vggm']

pretrained_settings = {
    'adnet': {
        'input_space': 'BGR',
        'input_size': [3, 112, 112],
        'input_range': [0, 255],
        'mean': [123.68, 116.779, 103.939],
        'std': [1, 1, 1],
        'num_classes': 11
    }
}


class ADNetDomainSpecificMot(nn.Module):
    """
    This module purpose is only for saving the state_dict's domain-specific layers of each domain.
    Put this module to CPU
    """

    def __init__(self, num_classes, num_history, num_obj_to_track=2):
        super(ADNetDomainSpecificMot, self).__init__()
        action_dynamic_size = num_classes * num_history * num_obj_to_track
        self.fc6 = nn.Linear(512 + action_dynamic_size, num_classes * num_obj_to_track)
        self.fc7 = nn.Linear(512 + action_dynamic_size, 1 * num_obj_to_track)

    def load_weights(self, base_file, video_index):
        """
        Load weights from file
        Args:
            base_file: (string)
            video_index: (int)
        """
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading ADNetDomainSpecific ' + str(video_index) + ' weights')
            checkpoint = torch.load(base_file, map_location=lambda storage, loc: storage)

            pretrained_dict = checkpoint['adnet_domain_specific_state_dict'][video_index].state_dict()
            model_dict = self.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(pretrained_dict)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_weights_from_adnet(self, adnet_net):
        """
        Load weights from ADNet. Use it after updating adnet to update the weights in this module
        Args:
            adnet_net: (ADNet) the updated ADNet whose fc6 and fc7
        """
        adnet_state_dict = adnet_net.state_dict()
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in adnet_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

        pass


class ADNetMot(nn.Module):

    def __init__(self, base_network, opts, num_classes=11, num_obj_to_track=2, phase='train', num_history=10,
                 use_gpu=True):
        super(ADNetMot, self).__init__()

        self.num_classes = num_classes
        self.phase = phase
        self.opts = opts
        self.use_gpu = use_gpu

        self.base_network = base_network
        self.fc4_5 = nn.Sequential(
            nn.Linear(18432 * num_obj_to_track, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  # [3]
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # -1 to differentiate between action '0' and haven't been explored
        self.action_history = np.full(num_history, -1)

        self.action_dynamic_size = num_classes * num_history * num_obj_to_track
        self.action_dynamic = torch.Tensor(np.zeros(self.action_dynamic_size))

        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()

        # self.fc6 = nn.Linear(512 + self.action_dynamic_size, self.num_classes)
        # self.fc7 = nn.Linear(512 + self.action_dynamic_size, 2)

        # self.rnn = nn.RNN(self.num_classes*2, 12, 1, batch_first=True)
        self.fc6 = nn.Linear(512 + self.action_dynamic_size, self.num_classes * num_obj_to_track)
        self.fc7 = nn.Linear(512 + self.action_dynamic_size, 1 * num_obj_to_track)

        # self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    # update_action_dynamic: history of action. We don't update the action_dynamic in SL learning.
    def forward(self, images_list, action_dynamic1=None, action_dynamic2=None, update_action_dynamic=False):
        """
        Args:
            x: (Tensor) the input of network
            action_dynamic: (Tensor) the previous state action dynamic.
                If None, use the self.action_dynamic in this Module
            update_action_dynamic: (bool) Whether to update the action_dynamic with the result.
                We don't update the action_dynamic in SL learning.
        """
        obj1 = images_list[:, 0, :, :, :]
        obj2 = images_list[:, 1, :, :, :]
        x1 = self.base_network(obj1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.base_network(obj2)
        x2 = x1.view(x2.size(0), -1)

        y = torch.cat((x1, x2), dim=1)
        x = self.fc4_5(y)

        if action_dynamic1 is None or action_dynamic2 is None:
            x = torch.cat((x, self.action_dynamic.expand(x.shape[0], self.action_dynamic.shape[0])), 1)
        else:
            x = torch.cat((x, torch.cat(action_dynamic1, action_dynamic2)))

        fc6_out = self.fc6(x)
        fc7_out = self.fc7(x)

        # if self.phase == 'test':
        # fc6_out = self.sigmoid(fc6_out)
        # fc7_out = self.sigmoid(fc7_out)

        # if update_action_dynamic:
        #     selected_action = np.argmax(fc6_out.detach().cpu().numpy())  # TODO: really okay to detach?
        #     self.action_history[1:] = self.action_history[0:-1]
        #     self.action_history[0] = selected_action
        #     self.update_action_dynamic(self.action_history)

        return fc6_out, fc7_out

    def add_action_to_hist(self, selected_action):
        self.action_history[1:] = self.action_history[0:-1]
        self.action_history[0] = selected_action
        self.update_action_dynamic(self.action_history)

    def load_domain_specific(self, adnet_domain_specific):
        """
        Load existing domain_specific weight to this model (i.e. fc6 and fc7). Do it before updating this model to
        update the weight to the specific domain
        Args:
             adnet_domain_specific: (ADNetDomainSpecific) the domain's ADNetDomainSpecific module.
        """
        # if self.use_gpu:
        #     adnet_domain_specific_ = nn.DataParallel(adnet_domain_specific)
        #     adnet_domain_specific_ = adnet_domain_specific_.cuda()
        # else:
        #     adnet_domain_specific_ = adnet_domain_specific

        domain_specific_state_dict = adnet_domain_specific.state_dict()
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in domain_specific_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

        # # return the adnet_domain_specific to cpu (to save gpu memory)
        # if self.use_gpu:
        #     adnet_domain_specific.cpu()

    def load_weights(self, base_file, domain_spec_index=None):
        """
        Args:
            base_file: (string) checkpoint filename
            domain_spec_index: (None or int) None if not loading.
                Fill it with int of the video idx to load the specific domain weight
        """
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            # self.load_state_dict(torch.load(base_file,
            #                      map_location=lambda storage, loc: storage))

            checkpoint = torch.load(base_file, map_location=lambda storage, loc: storage)

            # load adnet
            pretrained_dict = checkpoint['adnet_state_dict']
            model_dict = self.state_dict()

            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if 'module' in k:
                    name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            pretrained_dict = new_state_dict

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            self.load_state_dict(model_dict)

            # load specific domain
            if domain_spec_index is not None:
                self.load_domain_specific(checkpoint['adnet_domain_specific_state_dict'][domain_spec_index])

            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def update_action_dynamic(self, action_history):
        onehot_action = get_action_history_onehot(action_history, self.opts)

        action_dynamic = onehot_action

        self.action_dynamic = torch.Tensor(action_dynamic)
        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()

    def reset_action_dynamic(self):
        self.action_dynamic = torch.zeros(self.action_dynamic_size)
        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()

    def get_action_dynamic(self):
        return self.action_dynamic

    def set_phase(self, phase):
        self.phase = phase


def adnet_mot(opts, base_network='vggm', num_obj_to_track=2, trained_file=None, random_initialize_domain_specific=False,
              multidomain=True,
              vid_index=None):
    """
    Args:
        base_network: (string)
        trained_file: (None or string) saved filename
        random_initialize_domain_specific: (bool) if there is trained file, whether to use the weight in the file (True)
            or just random initialize (False). Won't matter if the trained_file is None (always False)
        multidomain: (bool) whether to have separate weight for each video or not. Default True: separate
        vid_index: (int) index of video for which to load weights (used for testing).
    Returns:
        adnet_model: (ADNet)
        domain_nets: (list of ADNetDomainSpecific) length: #videos
    """
    assert base_network in ['vggm'], "Base network variant is unavailable"

    num_classes = opts['num_actions']
    num_history = opts['num_action_history']

    assert num_classes in [11], "num classes does not exist"

    settings = pretrained_settings['adnet']

    if base_network == 'vggm':
        base_network = vggm()  # by default, load vggm's weights too
        base_network = base_network.features[0:10]

    else:  # change this part if adding more base network variant
        base_network = vggm()
        base_network = base_network.features[0:10]

    if trained_file:
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        print('Resuming training, loading {}...'.format(trained_file))

        adnet_model = ADNetMot(base_network=base_network, opts=opts, num_classes=num_classes, num_history=num_history)
        # if use_gpu:
        #     adnet_model = nn.DataParallel(adnet_model)
        #     cudnn.benchmark = True
        #     adnet_model = adnet_model.cuda()

        adnet_model.load_weights(trained_file, vid_index)

        adnet_model.input_space = settings['input_space']
        adnet_model.input_size = settings['input_size']
        adnet_model.input_range = settings['input_range']
        adnet_model.mean = settings['mean']
        adnet_model.std = settings['std']
    else:
        adnet_model = ADNetMot(base_network=base_network, opts=opts, num_classes=num_classes)

    # initialize domain-specific network
    domain_nets = []
    if multidomain:
        num_videos = opts['num_videos']
    else:
        num_videos = 1

    if not (trained_file and not random_initialize_domain_specific):
        scal = torch.Tensor([0.01])
        # fc 6
        nn.init.normal_(adnet_model.fc6.weight.data)
        adnet_model.fc6.weight.data = adnet_model.fc6.weight.data * scal.expand_as(
            adnet_model.fc6.weight.data)
        adnet_model.fc6.bias.data.fill_(0)
        # fc 7
        nn.init.normal_(adnet_model.fc7.weight.data)
        adnet_model.fc7.weight.data = adnet_model.fc7.weight.data * scal.expand_as(
            adnet_model.fc7.weight.data)
        adnet_model.fc7.bias.data.fill_(0)

    for idx in range(num_videos):
        domain_nets.append(ADNetDomainSpecificMot(num_classes=num_classes, num_history=num_history))

        scal = torch.Tensor([0.01])

        if trained_file and not random_initialize_domain_specific:
            domain_nets[idx].load_weights(trained_file, idx)
        else:  # random initialization of weights for fc 6 and fc7
            # fc 6
            nn.init.normal_(domain_nets[idx].fc6.weight.data)
            domain_nets[idx].fc6.weight.data = domain_nets[idx].fc6.weight.data * scal.expand_as(
                domain_nets[idx].fc6.weight.data)
            domain_nets[idx].fc6.bias.data.fill_(0)
            # fc 7
            nn.init.normal_(domain_nets[idx].fc7.weight.data)
            domain_nets[idx].fc7.weight.data = domain_nets[idx].fc7.weight.data * scal.expand_as(
                domain_nets[idx].fc7.weight.data)
            domain_nets[idx].fc7.bias.data.fill_(0)

    return adnet_model, domain_nets
