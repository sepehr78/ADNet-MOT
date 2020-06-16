# Get video informations (image paths and ground truths)
# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_video_infos.m

import os
import sys
import glob
import numpy as np
import cv2


def get_video_infos(bench_name, video_path, video_name):
    assert bench_name in ['vot13', 'vot14', 'vot15']

    if bench_name in ['vot13', 'vot14', 'vot15']:
        video_info = {
            'gt': [],
            'img_files': [],
            'name': video_name,
            'db_name': bench_name,
            'nframes': 0
        }
        benchmarkSeqHome = video_path

        # img path
        imgDir = os.path.join('..', benchmarkSeqHome, video_name, 'color')
        if not os.path.exists(imgDir):
            print(imgDir + ' does not exist!')
            sys.exit(1)

        img_files = glob.glob(os.path.join(imgDir, '*.jpg'))
        img_files.sort(key=str.lower)

        for i in range(len(img_files)):
            img_path = os.path.join(img_files[i])
            video_info['img_files'].append(img_path)

        # gt path
        gtPath = os.path.join('..', benchmarkSeqHome, video_name, 'groundtruth.txt')

        if not os.path.exists(gtPath):
            print(gtPath + ' does not exist!')
            sys.exit(1)

        # parse gt
        gtFile = open(gtPath, 'r')
        gt = gtFile.read().split('\n')
        for i in range(len(gt)):
            if gt[i] == '' or gt[i] is None:
                continue
            gt[i] = gt[i].split(',')
            gt[i] = list(map(float, gt[i]))
        gtFile.close()

        # converts gt to x,y,widht,height format
        if len(gt[0]) >= 6:
            for gtidx in range(len(gt)):
                if gt[gtidx] == "":
                    continue
                x = gt[gtidx][0:len(gt[gtidx]):2]
                y = gt[gtidx][1:len(gt[gtidx]):2]
                gt[gtidx] = [min(x),
                             min(y),
                             max(x) - min(x),
                             max(y) - min(y)]

        video_info['gt'] = gt

        video_info['nframes'] = min(len(video_info['img_files']), len(video_info['gt']))
        video_info['img_files'] = video_info['img_files'][:video_info['nframes']]
        video_info['gt'] = video_info['gt'][:video_info['nframes']]

        return video_info


def get_video_infos_adnet_mot(bench_name, video_path, video_name, num_obj_to_track):
    video_info = {
        'gt_list': [],
        'img_files': [],
        'name': video_name,
        'db_name': bench_name,
        'nframes': 0
    }
    benchmarkSeqHome = video_path
    gt_list = []

    # img path
    imgDir = os.path.join('..', benchmarkSeqHome, video_name, 'color')
    if not os.path.exists(imgDir):
        print(imgDir + ' does not exist!')
        sys.exit(1)

    img_files = glob.glob(os.path.join(imgDir, '*.jpg'))
    img_files.sort(key=str.lower)

    for i in range(len(img_files)):
        img_path = os.path.join(img_files[i])
        video_info['img_files'].append(img_path)

    # gt path
    gt_path_list = []
    for i in range(num_obj_to_track):
        gt_path_list.append(os.path.join('..', benchmarkSeqHome, video_name, 'groundtruth{}.txt'.format(i)))

    # parse gt
    for gtPath in gt_path_list:
        gtFile = open(gtPath, 'r')
        gt = gtFile.read().split('\n')
        for i in range(len(gt)):
            if gt[i] == '' or gt[i] is None:
                continue
            gt[i] = gt[i].split(',')
            gt[i] = list(map(float, gt[i]))
        gtFile.close()

        # converts gt to x,y,widht,height format
        if len(gt[0]) >= 6:
            for gtidx in range(len(gt)):
                if gt[gtidx] == "":
                    continue
                x = gt[gtidx][0:len(gt[gtidx]):2]
                y = gt[gtidx][1:len(gt[gtidx]):2]
                gt[gtidx] = [min(x),
                             min(y),
                             max(x) - min(x),
                             max(y) - min(y)]
        gt.pop(-1)  # TODO GENERALIZE
        gt_list.append(gt)


    video_info['nframes'] = len(video_info['img_files'])
    video_info['gt_list'] = np.array(gt_list)
    return video_info


# TODO DELETE
def get_top_ids_to_keep(gt_np, num_obj):
    unique, counts = np.unique(gt_np[:, 1], return_counts=True)
    count_arr = np.vstack((unique, counts)).T  # count = num frames object was visible/tracked for
    count_arr = count_arr[(-count_arr[:, 1]).argsort()]  # sort in descending order of count
    ids_to_keep = count_arr[0:num_obj, 0]
    return ids_to_keep


def show_boxes_test(img_files, gt_np, ids_to_keep, starting_frame):
    # get rid of other ids
    gt_np = gt_np[np.any([gt_np[:, 1] == x for x in ids_to_keep], axis=0)]

    # check that there are no frame jumps
    # assert len(np.unique(gt_np[:, 0])) == gt_np[:, 0].max(), "There are frame skips!"
    frame_num = 0
    for file in img_files:
        if frame_num < starting_frame:
            frame_num += 1
            continue
        frame_num += 1
        file_str = os.path.splitext(file)[0]
        frame_str = file_str[file_str.rfind('/') + 1:]
        frame_id = int(frame_str)
        gt_list = list(gt_np[gt_np[:, 0] == frame_id][:, 2:])
        img = cv2.imread(file)
        for gt in gt_list:
            cv2.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 0, 255), 5)

        scale_percent = 80  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('test', resized)

        key = cv2.waitKey(0) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


def get_video_infos_mot(bench_name, video_path, video_name, num_obj, min_frames_visible, area_thresh):
    assert "MOT" in bench_name

    object_infos = []
    benchmarkSeqHome = video_path

    # img path
    imgDir = os.path.join('..', benchmarkSeqHome, video_name, 'img1')
    if not os.path.exists(imgDir):
        print(imgDir + ' does not exist!')
        sys.exit(1)

    img_files = glob.glob(os.path.join(imgDir, '*.jpg'))
    img_files.sort(key=str.lower)

    for i in range(len(img_files)):
        img_path = os.path.join(img_files[i])

    # gt path
    gtPath = os.path.join('..', benchmarkSeqHome, video_name, 'gt', 'gt.txt')

    if not os.path.exists(gtPath):
        print(gtPath + ' does not exist!')
        sys.exit(1)

    # parse gt
    gtFile = open(gtPath, 'r')
    gt = gtFile.read().split('\n')
    del_list = []
    for i in range(len(gt)):
        if gt[i] == '' or gt[i] is None:
            del_list.append(i)
            continue
        gt[i] = gt[i].split(',')
        del gt[i][-3:]  # del conf and other useless info
        gt[i] = np.array(list(map(int, gt[i])))
    gtFile.close()
    for i in del_list:
        del gt[i]

    # converts gt numpy and sort based on frame id
    gt_np = np.array(gt)
    gt_np = gt_np[gt_np[:, 0].argsort()]
    # show_boxes_test(img_files, gt_np, num_obj)

    # go through frame by frame until num_obj objects that are visible in min_frames_visible
    all_object_ids = set()
    starting_frame = 1
    for frame in range(100, gt_np[-1, 0] + 1):
        ids_visible_in_frame = gt_np[gt_np[:, 0] == frame][:, 1]
        ids_visible_in_frame = list(filter(lambda x: x not in all_object_ids, ids_visible_in_frame))
        if len(ids_visible_in_frame) >= num_obj:
            obj_ids = ids_visible_in_frame[0:num_obj]  # pick num_obj first ids
            visible_rows_for_obj = [np.logical_and(gt_np[:, 1] == x, gt_np[:, 0] >= frame) for x in obj_ids]

            # a list of frame ids for which the object is visible for
            visible_frames_for_obj = [gt_np[x, 0] for x in visible_rows_for_obj]
            max_list = [len(x) for x in visible_frames_for_obj]
            if min(max_list) >= min_frames_visible:
                starting_frame = frame
                all_object_ids.update(obj_ids)
                for i, objid in enumerate(obj_ids):
                    nframes = len(visible_frames_for_obj[i])

                    # get gt_list
                    gt_list = []
                    obj_img_files = []
                    for row in gt_np[visible_rows_for_obj[i]]:
                        frame_id = row[0]
                        gt = row[2:]
                        if gt[2] * gt[3] < 0 * area_thresh:
                            continue
                        gt = np.array([0 if x < 0 else x for x in gt])
                        gt_list.append(gt)
                        img_file = os.path.join(imgDir, str(frame_id).zfill(6) + ".jpg")
                        obj_img_files.append(img_file)
                    object_info = {'gt': gt_list,
                                   'img_files': obj_img_files,
                                   'name': video_name,
                                   'db_name': bench_name,
                                   'nframes': nframes}
                    object_infos.append(object_info)
            else:
                continue
            break
    # show_boxes_test(img_files, gt_np, list(all_object_ids), starting_frame)
    return object_infos

# test the module
# from utils.get_train_videos import get_train_videos
# from utils.init_params import opts
# train_videos = get_train_videos(opts)
# bench_name = train_videos['bench_names'][0]
# video_name = train_videos['video_names'][0]
# video_path = train_videos['video_paths'][0]
# get_video_infos(bench_name, video_path, video_name)

# get N most visible objects
#     unique, counts = np.unique(gt_np[:, 1], return_counts=True)
#     count_arr = np.vstack((unique, counts)).T  # count = num frames object was visible/tracked for
#     count_arr = count_arr[(-count_arr[:, 1]).argsort()]  # sort in descending order of count
#     ids_to_keep = count_arr[0:opts['num_obj_mot'], 0]
#
#     # get rid of other ids
#     gt_np = gt_np[np.any([gt_np[:, 1] == x for x in ids_to_keep], axis=0)]
#
#     # check that there are no frame jumps
#     assert len(np.unique(gt_np[:, 0])) == gt_np[:, 0].max(), "There are frame skips!"
