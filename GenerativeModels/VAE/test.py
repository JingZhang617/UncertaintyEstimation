import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from scipy import misc
from model.ResNet_models import Pred_endecoder
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from tqdm import tqdm
import time



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')

opt = parser.parse_args()

dataset_path = '/home/jingzhang/jing_files/RGBD_COD/dataset/test/'

generator = Pred_endecoder(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load('./models/Model_50_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['CAMO','CHAMELEON', 'COD10K', 'NC4K']
time_list = []
for dataset in test_datasets:
    save_path_base = './results/' + dataset + '/'
    # save_path = './results/ResNet50/holo/train/left/'
    if not os.path.exists(save_path_base):
        os.makedirs(save_path_base)

    image_root = dataset_path + dataset + '/Imgs/'
    for iter in range(10):
        test_loader = test_dataset(image_root, opt.testsize)
        for i in tqdm(range(test_loader.size), desc=dataset):
            image, HH, WW, name = test_loader.load_data()
            image = image.cuda()
            torch.cuda.synchronize()
            start = time.time()
            generator_pred = generator.forward(image)
            res = generator_pred
            res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            torch.cuda.synchronize()
            end = time.time()
            time_list.append(end - start)
            res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
            save_path = save_path_base + dataset + '/' + 'sal' + str(iter) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + name, res)