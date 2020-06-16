# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/train/get_train_dbs.m
import random
import sys

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import numpy.matlib

from utils.gen_samples import gen_samples
from utils.overlap_ratio import overlap_ratio
from utils.gen_action_labels import gen_action_labels

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    drawpoly(img,pts,color,thickness,style)

def show_examples_test(pos_examples, neg_examples, img_path):
    i = 0
    while True:
        # random sample some boxes to show
        num_sample = 5

        pos_boxes = random.sample(pos_examples, num_sample)
        neg_boxes = random.sample(neg_examples, num_sample)
        img = cv2.imread(img_path)

        for pos_box, neg_box in zip(pos_boxes, neg_boxes):
            cv2.rectangle(img, (pos_box[0], pos_box[1]), (pos_box[0] + pos_box[2], pos_box[1] + pos_box[3]),
                          (0, 255, 0), 1)
            cv2.rectangle(img, (neg_box[0], neg_box[1]), (neg_box[0] + neg_box[2], neg_box[1] + neg_box[3]),
                          (0, 0, 255), 1)
        cv2.imshow('test', img)

        key = cv2.waitKey(0) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        elif key == ord("s"):
            cv2.imwrite("sl_training_sample_{}.png".format(i), img)
            i += 1


def get_train_dbs(vid_info, opts):
    img = cv2.imread(vid_info['img_files'][0])

    opts['scale_factor'] = 1.05
    opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    if vid_info['db_name'] == 'alov300':
        train_sequences = vid_info['gt_use'] == 1
    else:
        train_sequences = list(range(0, vid_info['nframes'], gt_skip))

    train_db_pos = []
    train_db_neg = []

    for train_i in range(len(train_sequences)):
        train_db_pos_ = {
            'img_path': [],
            'bboxes': [],
            'labels': [],
            'score_labels': []
        }
        train_db_neg_ = {
            'img_path': [],
            'bboxes': [],
            'labels': [],
            'score_labels': []
        }

        img_idx = train_sequences[train_i]
        gt_bbox = vid_info['gt'][img_idx]

        if len(gt_bbox) == 0:
            continue

        pos_examples = []
        while len(pos_examples) < opts['nPos_train']:
            pos = gen_samples('gaussian', gt_bbox, opts['nPos_train'] * 5, opts, 0.1, 5)
            r = overlap_ratio(pos, np.matlib.repmat(gt_bbox, len(pos), 1))
            pos = pos[np.array(r) > opts['posThre_train']]
            if len(pos) == 0:
                continue
            pos = pos[np.random.randint(low=0, high=len(pos),
                                        size=min(len(pos), opts['nPos_train'] - len(pos_examples))), :]
            pos_examples.extend(pos)

        neg_examples = []
        while len(neg_examples) < opts['nNeg_train']:
            # in original code, this 1 line below use opts['nPos_train'] instead of opts['nNeg_train']
            neg = gen_samples('gaussian', gt_bbox, opts['nNeg_train'] * 5, opts, 2, 10)
            r = overlap_ratio(neg, np.matlib.repmat(gt_bbox, len(neg), 1))
            neg = neg[np.array(r) < opts['negThre_train']]
            if len(neg) == 0:
                continue
            neg = neg[np.random.randint(low=0, high=len(neg),
                                        size=min(len(neg), opts['nNeg_train'] - len(neg_examples))), :]
            neg_examples.extend(neg)

        show_examples_test(pos_examples, neg_examples, vid_info['img_files'][img_idx])
        # examples = pos_examples + neg_examples
        action_labels_pos = gen_action_labels(opts['num_actions'], opts, np.array(pos_examples), gt_bbox)
        action_labels_neg = np.full((opts['num_actions'], len(neg_examples)), fill_value=-1)

        action_labels_pos = np.transpose(action_labels_pos).tolist()
        action_labels_neg = np.transpose(action_labels_neg).tolist()

        # action_labels = action_labels_pos + action_labels_neg

        train_db_pos_['img_path'] = np.full(len(pos_examples), vid_info['img_files'][img_idx])
        train_db_pos_['bboxes'] = pos_examples
        train_db_pos_['labels'] = action_labels_pos
        # score labels: 1 is positive. 0 is negative
        train_db_pos_['score_labels'] = list(np.ones(len(pos_examples), dtype=int))

        train_db_neg_['img_path'] = np.full(len(neg_examples), vid_info['img_files'][img_idx])
        train_db_neg_['bboxes'] = neg_examples
        train_db_neg_['labels'] = action_labels_neg
        # score labels: 1 is positive. 0 is negative
        train_db_neg_['score_labels'] = list(np.zeros(len(neg_examples), dtype=int))

        train_db_pos.append(train_db_pos_)
        train_db_neg.append(train_db_neg_)

    return train_db_pos, train_db_neg


def show_examples_test_mot(pos_examples_list, neg_examples_list, img_path):
    i = 0
    while True:
        # random sample some boxes to show
        num_sample = 5
        pos_boxes_list = []
        neg_boxes_list = []
        for pos_examples, neg_examples in zip(pos_examples_list, neg_examples_list):
            pos_boxes_list.append(random.sample(pos_examples, num_sample))
            neg_boxes_list.append(random.sample(neg_examples, num_sample))

        img = cv2.imread(img_path)

        for j, (pos_boxes, neg_boxes) in enumerate(zip(pos_boxes_list, neg_boxes_list)):
            for pos_box, neg_box in zip(pos_boxes, neg_boxes):
                if j == 0:
                    cv2.rectangle(img, (pos_box[0], pos_box[1]), (pos_box[0] + pos_box[2], pos_box[1] + pos_box[3]),
                             (0, 255, 0), 1)
                    cv2.rectangle(img, (neg_box[0], neg_box[1]), (neg_box[0] + neg_box[2], neg_box[1] + neg_box[3]),
                             (0, 0, 255), 1)
                else:
                    drawrect(img, (pos_box[0], pos_box[1]), (pos_box[0] + pos_box[2], pos_box[1] + pos_box[3]),
                                  (0, 255, 0), 1, 'line')
                    drawrect(img, (neg_box[0], neg_box[1]), (neg_box[0] + neg_box[2], neg_box[1] + neg_box[3]),
                                  (0, 0, 255), 1, 'line')
        cv2.imshow('test', img)

        key = cv2.waitKey(0) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        elif key == ord("s"):
            cv2.imwrite("sample_mot_{}.png".format(i), img)
            i += 1


def get_train_dbs_mot(vid_info, opts, num_obj_to_track):
    img = cv2.imread(vid_info['img_files'][0])

    opts['scale_factor'] = 1.05
    opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    if vid_info['db_name'] == 'alov300':
        train_sequences = vid_info['gt_use'] == 1
    else:
        train_sequences = list(range(0, vid_info['nframes'], gt_skip))

    train_db_pos = []
    train_db_neg = []

    for train_i in range(len(train_sequences)):
        train_db_pos_ = {
            'img_path': [],
            'bboxes_list': [],
            'labels_list': [],
            'score_labels': []
        }
        train_db_neg_ = {
            'img_path': [],
            'bboxes_list': [],
            'labels_list': [],
            'score_labels': []
        }

        img_idx = train_sequences[train_i]
        gt_list = vid_info['gt_list'][:, img_idx, :]

        pos_examples_list = []
        neg_examples_list = []
        action_labels_pos_list = []
        action_labels_neg_list = []
        for i in range(num_obj_to_track):
            pos_examples_list.append([])
            while len(pos_examples_list[i]) < opts['nPos_train']:
                pos = gen_samples('gaussian', gt_list[i], opts['nPos_train'] * 5, opts, 0.1, 5)
                r = overlap_ratio(pos, np.matlib.repmat(gt_list[i], len(pos), 1))
                pos = pos[np.array(r) > opts['posThre_train']]
                if len(pos) == 0:
                    continue
                pos = pos[np.random.randint(low=0, high=len(pos),
                                            size=min(len(pos), opts['nPos_train'] - len(pos_examples_list[i]))), :]
                pos_examples_list[i].extend(pos)

            neg_examples_list.append([])
            while len(neg_examples_list[i]) < opts['nNeg_train']:
                # in original code, this 1 line below use opts['nPos_train'] instead of opts['nNeg_train']
                neg = gen_samples('gaussian', gt_list[i], opts['nNeg_train'] * 5, opts, 2, 10)
                r = overlap_ratio(neg, np.matlib.repmat(gt_list[i], len(neg), 1))
                neg = neg[np.array(r) < opts['negThre_train']]
                if len(neg) == 0:
                    continue
                neg = neg[np.random.randint(low=0, high=len(neg),
                                            size=min(len(neg), opts['nNeg_train'] - len(neg_examples_list[i]))), :]
                neg_examples_list[i].extend(neg)

            # show_examples_test_mot(pos_examples_list, neg_examples_list, vid_info['img_files'][img_idx])
            # examples = pos_examples + neg_examples
            action_labels_pos_list.append(gen_action_labels(opts['num_actions'], opts, np.array(pos_examples_list[i]), gt_list[i]))
            action_labels_neg_list.append(np.full((opts['num_actions'], len(neg_examples_list[i])), fill_value=-1))

            action_labels_pos_list[i] = np.transpose(action_labels_pos_list[i])
            action_labels_neg_list[i] = np.transpose(action_labels_neg_list[i])

        # action_labels = action_labels_pos + action_labels_neg

        train_db_pos_['img_path'] = np.full(len(pos_examples_list[0]), vid_info['img_files'][img_idx])
        train_db_neg_['img_path'] = np.full(len(neg_examples_list[0]), vid_info['img_files'][img_idx])

        train_db_pos_['bboxes_list'] = pos_examples_list
        train_db_pos_['labels_list'] = action_labels_pos_list
        # score labels: 1 is positive. 0 is negative
        train_db_pos_['score_labels'] = list(np.ones(len(pos_examples_list[0]), dtype=int))

        train_db_neg_['bboxes_list'] = neg_examples_list
        train_db_neg_['labels_list'] = action_labels_neg_list
        # score labels: 1 is positive. 0 is negative
        train_db_neg_['score_labels'] = list(np.zeros(len(neg_examples_list[0]), dtype=int))

        train_db_pos.append(train_db_pos_)
        train_db_neg.append(train_db_neg_)

    return train_db_pos, train_db_neg

# test the module
# from utils.get_train_videos import get_train_videos
# from utils.init_params import opts
# from utils.get_video_infos import get_video_infos
# train_videos = get_train_videos(opts)
# bench_name = train_videos['bench_names'][0]
# video_name = train_videos['video_names'][0]
# video_path = train_videos['video_paths'][0]
# vid_info = get_video_infos(bench_name, video_path, video_name)
# get_train_dbs(vid_info, opts)
