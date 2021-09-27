import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Pred_endecoder, EBM_Prior
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
from tools import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--lr_ebm', type=float, default=1e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=5, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=256, help='reduced channel of saliency feat')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--focal_lamda', type=int, default=1, help='lamda of focal loss')
parser.add_argument('--bnn_steps', type=int, default=6, help='BNN sampling iterations')
parser.add_argument('--lvm_steps', type=int, default=6, help='LVM sampling iterations')
parser.add_argument('--pred_steps', type=int, default=6, help='Predictive sampling iterations')
parser.add_argument('--smooth_loss_weight', type=float, default=0.4, help='weight of the smooth loss')
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

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Pred_endecoder(channel=opt.feat_channel,latent_dim=opt.latent_dim)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

ebm_model = EBM_Prior(opt.ebm_out_dim, opt.ebm_middle_dim, opt.latent_dim)
ebm_model.cuda()
ebm_model_params = ebm_model.parameters()
ebm_model_optimizer = torch.optim.Adam(ebm_model_params, opt.lr_ebm)

print("Model based on {} have {:.4f}Mb paramerters in total".format('Generator', sum(x.numel()/1e6 for x in generator.parameters())))
print("EBM Model based on {} have {:.4f}Mb paramerters in total".format('EBM', sum(
        x.numel() / 1e6 for x in ebm_model.parameters())))

image_root = './train/Imgs/'
gt_root = './train/GT/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  # multi-scale training


def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def structure_loss_focal_loss(pred, mask, weight):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)


    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='none')
    wbce = (((1-weight)**opt.focal_lamda)*weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_pred(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred_pred.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_original_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

def no_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()

def yes_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def sample_p_0(n=opt.batchsize, sig=opt.e_init_sig):
    return sig * torch.randn(*[n, opt.latent_dim, 1, 1]).to(device)

def compute_energy(score):
    if opt.e_energy_form == 'tanh':
        energy = F.tanh(score.squeeze())
    elif opt.e_energy_form == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif opt.e_energy_form == 'softplus':
        energy = F.softplus(score.squeeze())
    else:
        energy = score.squeeze()
    return energy

print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            z_e_0 = sample_p_0(n=images.shape[0])
            z_g_0 = sample_p_0(n=images.shape[0])
            ## sample langevin prior of z
            z_e_0 = Variable(z_e_0)
            z = z_e_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.e_l_steps):
                en = ebm_model(z)
                z_grad = torch.autograd.grad(en.sum(), z)[0]
                z.data = z.data - 0.5 * opt.e_l_step_size * opt.e_l_step_size * (
                        z_grad + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.e_l_step_size * torch.randn_like(z).data
            z_e_noise = z.detach()  ## z_

            ## sample langevin post of z
            z_g_0 = Variable(z_g_0)
            z = z_g_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.g_l_steps):
                gen_res = generator(images, z)
                g_log_lkhd = 1.0 / (2.0 * opt.g_llhd_sigma * opt.g_llhd_sigma) * mse_loss(
                    torch.sigmoid(gen_res), gts)
                z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                en = ebm_model(z)
                z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                z.data = z.data - 0.5 * opt.g_l_step_size * opt.g_l_step_size * (
                        z_grad_g + z_grad_e + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.g_l_step_size * torch.randn_like(z).data

            z_g_noise = z.detach()  ## z+

            pred = generator(images,z_g_noise)
            loss_all = structure_loss(pred, gts)

            loss_all.backward()
            generator_optimizer.step()

            ## learn the ebm
            en_neg = compute_energy(ebm_model(
                z_e_noise.detach())).mean()
            en_pos = compute_energy(ebm_model(z_g_noise.detach())).mean()
            loss_e = en_pos - en_neg
            loss_e.backward()
            ebm_model_optimizer.step()

            visualize_pred(torch.sigmoid(pred))
            visualize_gt(gts)

            if rate == 1:
                loss_record.update(loss_all.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(ebm_model_optimizer, opt.lr_ebm, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        torch.save(ebm_model.state_dict(), save_path + 'Model' + '_%d' % epoch + '_ebm.pth')
