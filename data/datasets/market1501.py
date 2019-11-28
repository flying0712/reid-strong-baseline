# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import re

import os.path as osp
#import shutil
import shutil
from shutil import copyfile

from .bases import BaseImageDataset

# Image Loading Code used for these examples
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio

import numpy as np
from PIL import Image
from torchvision import transforms as tfs


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='/home/code/my_data/', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        root = '/root/code/my_data/'
        #self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'my_pytorch/train/train_train')
        self.query_dir = osp.join(self.dataset_dir, 'my_pytorch/train/train_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'my_pytorch/train/train_gallery')

        #self.data_dict = self._get_data_dict(self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir)
        #self._make_my_set(self.data_dict, self.dataset_dir + 'train_set', self.train_dir, self.query_dir, self.gallery_dir)
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _get_data_dict(self, data_root_path, train_train_path, train_query_path,train_gallery_path):
        img_list = []
        ID_last = '-1'
        num = 0
        data_dict = {}
        MAX_LINE =20500
        drop_num = 0

        pytorch_path = osp.join(data_root_path, 'my_pytorch')
        if not os.path.isdir(pytorch_path):
            os.makedirs(pytorch_path)
        else:
            shutil.rmtree(pytorch_path)
            os.makedirs(pytorch_path)

        if not os.path.isdir(train_train_path):
            os.makedirs(train_train_path)
        if not os.path.isdir(train_query_path):
            os.makedirs(train_query_path)
        if not os.path.isdir(train_gallery_path):
            os.makedirs(train_gallery_path)

        tag_file = data_root_path + 'train_list.txt'
        fp = open(tag_file, 'r')
        for i in range(MAX_LINE):
            line = fp.readline()
            line = line.strip().replace('\r', '').replace('\n', '')
            if (line != ""):
                name, ID = list(re.findall(r"train/(.+?) (.*)", line)[0])
                # name, ID = line.split( ' ' )
                if name == "" or ID == "" or not name[-3:] == 'png':
                    continue

                if ID != ID_last:
                    if ID_last != '-1':
                        if num >= drop_num:
                            data_dict.update({ID_last: [num, img_list]})
                    num = 1
                    ID_last = ID
                    img_list = []
                    img_list.append(name)
                else:
                    num += 1
                    img_list.append(name)
        if num >= drop_num:
            data_dict.update({ID: [num, img_list]})
        return data_dict

    def _extand_img(self,dst_full_path):
        head = dst_full_path.split('.')[0]
        img = Image.open(dst_full_path)

        # new_im = tfs.Resize((100, 200))(img)
        # imageio.imwrite('./00073233_resize.png', new_im)

        rot_im = tfs.RandomRotation(30)(img)
        imageio.imwrite(head + '_randrot.png', rot_im)

        # 亮度
        bright_im = tfs.ColorJitter(brightness=0.5)(img)  # 随机从 0 ~ 2 之间亮度变化，1 表示原图
        imageio.imwrite(head + '_bright.png', bright_im)

        contrast_im = tfs.ColorJitter(contrast=0.5, hue=0.3)(img)  # 随机从 0 ~ 2 之间对比度变化，1 表示原图
        imageio.imwrite(head + '_contrast.png', contrast_im)

        img_np = np.array(img)
        flipped_img = np.fliplr(img_np)
        imageio.imwrite(head + '_flip.png', flipped_img)

        HEIGHT = 128
        WIDTH = 256
        # Shifting Left
        for i in range(HEIGHT, 1, -1):
            for j in range(WIDTH):
                if (i < HEIGHT - 20):
                    img_np[j][i] = img_np[j][i - 20]
                elif (i < HEIGHT - 1):
                    img_np[j][i] = 0
        imageio.imwrite(head + '_ls.png', img_np)

    def _make_my_set(self, data_dict, train_path_org, train_train_path, train_query_path, train_gallery_path):
        N_LEAST = 3
        N1 = 6
        N2 = 10000
        i = 0
        for ID in data_dict:
            i += 0
            c_num = data_dict[ID][0]
            if c_num >= N_LEAST and c_num <= N1:
                for name in data_dict[ID][1]:
                    camera_id = 3  # for train_train
                    new_name = ID + '_c' + str(camera_id) + 's' + name
                    src_full_path = train_path_org + '/' + name
                    dst_full_path = train_train_path + '/' + new_name
                    copyfile(src_full_path, dst_full_path )  # train set
                    self. _extand_img(dst_full_path)
                    #train_train_dataset.append((img_full_path, int(ID), camera_id))

            elif c_num > N1 and c_num <= N2:
                name = data_dict[ID][1][0]
                camera_id = 1  # for train_query
                new_name = ID + '_c' + str(camera_id) + 's' + name
                src_full_path = train_path_org + '/' + name
                dst_full_path = train_query_path + '/' + new_name
                copyfile(src_full_path, dst_full_path )  # train_query set

                for name in data_dict[ID][1]:
                    camera_id = 2  # for train_gallery
                    new_name = ID + '_c' + str(camera_id) + 's' + name
                    src_full_path = train_path_org + '/' + name
                    dst_full_path = train_gallery_path + '/' + new_name
                    copyfile(src_full_path, dst_full_path )  # train_gallery set

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
