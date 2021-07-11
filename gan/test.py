import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from scipy import misc
import cv2
from model.ResNet_models import ResNet_Baseline, FCDiscriminator
from data import test_dataset



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
parser.add_argument('--forward_iter', type=int, default=10, help='iteration of samplings')

opt = parser.parse_args()

dataset_path = '/home/jingzhang/jing_files/RGBD_COD/dataset/test/'
model = ResNet_Baseline()
model.load_state_dict(torch.load('./models/Model_50_gen.pth'))
dis_model = FCDiscriminator()
dis_model.load_state_dict(torch.load('./models/Model_50_dis.pth'))

model.cuda()
model.eval()

dis_model.cuda()
dis_model.eval()

test_datasets = ['CAMO','CHAMELEON','COD10K','NC4K']

for dataset in test_datasets:
    save_mean_path = './results/' + dataset + '/Mean/'
    if not os.path.exists(save_mean_path):
        os.makedirs(save_mean_path)

    save_dis_path = './results/' + dataset + '/Dis/'
    if not os.path.exists(save_dis_path):
        os.makedirs(save_dis_path)

    image_root = dataset_path + dataset + '/Imgs/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        sal_pred = list()
        for iter in range(opt.forward_iter):
            z_noise = torch.randn(image.shape[0], opt.latent_dim).cuda()
            res = model(image, z_noise)
            sal_pred.append(res.detach())
            res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
            save_path = './results/' + dataset + '/sal' + str(iter) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + name, res)
        sal_preds = torch.sigmoid(sal_pred[0]).clone()
        for iter in range(1, opt.forward_iter):
            sal_preds = torch.cat((sal_preds, torch.sigmoid(sal_pred[iter])), 1)
        var_map = torch.var(sal_preds, 1, keepdim=True)
        var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        var_map = F.upsample(var_map, size=[WW, HH], mode='bilinear', align_corners=False)
        res = var_map.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        save_path = save_dis_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path + name.replace('.png', '_var.png'), res)

        mean_map = torch.mean(sal_preds, 1, keepdim=True)
        res = mean_map
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        save_path = save_mean_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path + name, res)

        Dis_output = dis_model(image, mean_map)
        Dis_output = F.upsample(Dis_output, size=[WW, HH], mode='bilinear',align_corners=True)
        res = Dis_output
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        save_path = save_dis_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path + name, res)

