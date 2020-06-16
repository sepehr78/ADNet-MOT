import random

import cv2
import os

from options.general import opts
from utils.get_train_videos import get_train_videos
from utils.get_video_infos import get_video_infos

pixel_variation = 0.05


def read_bag_img(obj_to_place_path='../datasets/data/cropped_bag.png'):
    image = cv2.imread(obj_to_place_path, 1)

    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    return cv2.merge(rgba, 4)


def generate_mot_dataset(vid_info, new_img_save_path, obj_to_place_path='../datasets/data/cropped_bag.png'):
    obj_img = cv2.imread(obj_to_place_path, -1)

    scale_percent = 60  # percent of original size
    width = int(obj_img.shape[1] * scale_percent / 100)
    height = int(obj_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    obj_img = cv2.resize(obj_img, dim, interpolation=cv2.INTER_AREA)

    temp = cv2.imread(vid_info['img_files'][0])
    max_x = temp.shape[1] - obj_img.shape[1]
    max_y = temp.shape[0] - obj_img.shape[0]
    x_var = pixel_variation * temp.shape[1]
    y_var = pixel_variation * temp.shape[0]

    alpha_s = obj_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    current_x = random.randint(0, max_x)
    current_y = random.randint(0, max_y)
    new_gt_arr = []  # [left top width height]
    new_im_arr = []
    for file in vid_info['img_files']:
        new_gt_arr.append((current_x, current_y, obj_img.shape[1], obj_img.shape[0]))
        y1, y2 = current_y, current_y + obj_img.shape[0]
        x1, x2 = current_x, current_x + obj_img.shape[1]
        im = cv2.imread(file)
        for c in range(0, 3):
            im[y1:y2, x1:x2, c] = (alpha_s * obj_img[:, :, c] + alpha_l * im[y1:y2, x1:x2, c])
        new_im_arr.append(im)
        cv2.imwrite(os.path.join(new_img_save_path, 'color', file[file.rfind('/') + 1:]), im)
        # cv2.imshow('hoy', im)
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord("q"):
        #     cv2.destroyAllWindows()
        #     return
        current_x += random.randint(-x_var, x_var)
        current_y += random.randint(-y_var, y_var)

        current_x = min(current_x, max_x)
        current_y = min(current_y, max_y)

        current_x = max(current_x, 0)
        current_y = max(current_y, 0)
    gt_file_path = os.path.join(new_img_save_path, 'groundtruth1.txt')
    with open(gt_file_path, "w") as text_file:
        for (x, y, w, h) in new_gt_arr:
            text_file.write("{},{},{},{}\n".format(x, y, w, h))
    return new_im_arr, new_gt_arr


if __name__ == '__main__':
    vid_info = get_video_infos('vot15', 'datasets/data/vot15', 'bag')
    generate_mot_dataset(vid_info, '../datasets/data/vot15/bagMOT')
