import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
from model.ResNet_models import Pred_endecoder, EBM_Prior
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from tqdm import tqdm
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--feat_channel', type=int, default=256, help='reduced channel of saliency feat')
parser.add_argument('--ebm_out_dim', type=int, default=1, help='ebm initial sigma')
parser.add_argument('--ebm_middle_dim', type=int, default=100, help='ebm initial sigma')
parser.add_argument('--latent_dim', type=int, default=32, help='ebm initial sigma')
parser.add_argument('--e_init_sig', type=float, default=1.0, help='ebm initial sigma')
parser.add_argument('--e_l_steps', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--e_l_step_size', type=float, default=0.4, help='ebm initial sigma')
parser.add_argument('--e_prior_sig', type=float, default=1.0, help='ebm initial sigma')
parser.add_argument('--g_l_steps', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='ebm initial sigma')
parser.add_argument('--g_l_step_size', type=float, default=0.1, help='ebm initial sigma')
parser.add_argument('--e_energy_form', type=str, default='identity', help='ebm initial sigma')
parser.add_argument('--batchsize', type=int, default=5, help='training batch size')

opt = parser.parse_args()

dataset_path = '/home/jingzhang/jing_files/RGBD_COD/dataset/test/'

generator = Pred_endecoder(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load('./models/Model_50_gen.pth'))

generator.cuda()
generator.eval()

discriminator = EBM_Prior(opt.ebm_out_dim, opt.ebm_middle_dim, opt.latent_dim)
discriminator.load_state_dict(torch.load('./models/Model_50_ebm.pth'))

discriminator.cuda()
discriminator.eval()

test_datasets = ['CAMO','CHAMELEON', 'COD10K', 'NC4K']

def sample_p_0(n=opt.batchsize, sig=opt.e_init_sig):
    return sig * torch.randn(*[n, opt.latent_dim, 1, 1]).to(device)
time_list = []
for dataset in test_datasets:
    save_path_base = './resultsM/' + dataset + '/'
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
            z_e_0 = sample_p_0(n=image.shape[0])
            ## sample langevin prior of z
            z_e_0 = Variable(z_e_0)
            z = z_e_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.e_l_steps):
                en = discriminator(z)
                z_grad = torch.autograd.grad(en.sum(), z)[0]
                z.data = z.data - 0.5 * opt.e_l_step_size * opt.e_l_step_size * (
                        z_grad + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.e_l_step_size * torch.randn_like(z).data
                # z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()

            z_e_noise = z.detach()  ## z_
            generator_pred = generator.forward(image,z_e_noise)
            res = generator_pred
            torch.cuda.synchronize()
            end = time.time()
            time_list.append(end - start)
            res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
            save_path = save_path_base + dataset + '/' + 'sal' + str(iter) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path+name, res)