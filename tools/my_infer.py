from data.datasets.dataset_loader import read_image  # 图片读取方法，可以自己写，我是用的baseline里自带的
import os
import torch
import numpy as np
import json
from  utils.re_ranking import re_ranking

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 指定gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = '/root/code/my_data/'

def my_inference(model, transform, batch_size): # 传入模型，数据预处理方法，batch_size
    query_list = list()
    # with open(data_root + 'query_a_list.txt', 'r') as f:
    #             # 测试集中txt文件
    #     lines = f.readlines()
    #     for i, line in enumerate(lines):
    #         data = line.split(" ")
    #         image_name = data[0].split("/")[1]
    #         img_file = os.path.join(data_root + 'query_b', image_name)  # 测试集query文件夹
    #         query_list.append(img_file)

    query_list = [os.path.join(data_root + 'query_b', x) for x in # 测试集gallery文件夹
                    os.listdir(data_root + 'query_b')]
    gallery_list = [os.path.join(data_root + 'gallery_b', x) for x in # 测试集gallery文件夹
                    os.listdir(data_root + 'gallery_b')]
    query_num = len(query_list)
    img_list = list()
    for q_img in query_list:
        q_img = read_image(q_img)
        q_img = transform(q_img)
        img_list.append(q_img)
    for g_img in gallery_list:
        g_img = read_image(g_img)
        g_img = transform(g_img)
        img_list.append(g_img)
    # img_list = img_list[:1000]
    iter_n = int(len(img_list)/batch_size) # batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    # img_list = img_list[0:iter_n*batch_size]
    print(iter_n)

    img_data = torch.Tensor([t.numpy() for t in img_list]).cuda()
    # img_data = torch.Tensor([t.numpy() for t in img_list]).cpu
    model = model.to(device)
    model.eval()

    all_feature = list()
    for i in range(iter_n):
        print("batch ----%d----" % (i))
        batch_data = img_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            batch_feature = model(batch_data).detach().cpu()
            # print(batch_feature)
            # batch_feature = model( batch_data ).detach().cuda()
            all_feature.append(batch_feature)

    print('done')
    all_feature = torch.cat(all_feature)
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]

    distmat = re_ranking(query_feat, gallery_feat, k1=20, k2=6, lambda_value=0.3) # rerank方法
    # distmat = distmat # 如果使用 euclidean_dist，不使用rerank改为：distamt = distamt.numpy()
    num_q, num_g = distmat.shape
    print(num_q)
    indices = np.argsort(distmat, axis=1)
    max_200_indices = indices[:, :200]
    print(max_200_indices)

    res_dict = dict()
    for q_idx in range(num_q):
        print(query_list[q_idx])
        filename = query_list[q_idx][query_list[q_idx].rindex("/")+1:]
        max_200_files = [gallery_list[i][gallery_list[i].rindex("/")+1:] for i in max_200_indices[q_idx]]
        res_dict[filename] = max_200_files

    with open(r'submission_B_4.json', 'w' ,encoding='utf-8') as f: # 提交文件
        json.dump(res_dict, f)

# if __name__ == '__main__':
#         my_inference()
