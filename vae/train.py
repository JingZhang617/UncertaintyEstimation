import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from datetime import datetime
from model.ResNet_models import ResNet_Baseline
from data import get_loader
from utils import clip_gradient, adjust_lr, AvgMeter, l2_regularisation
import cv2
from scipy import misc
from PIL import Image
import torchvision.transforms.functional as tf



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--reg_weight', type=float, default=1e-4, help='regularization weight')
parser.add_argument('--lat_weight', type=float, default=10.0, help='latent loss weight')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='vae loss weight')
# parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
# parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
# parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
model = ResNet_Baseline()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
print("Model based on {} have {:.4f}Mb paramerters in total".format('Res50', sum(x.numel()/1e6 for x in model.parameters())))


image_root = './train/Imgs/'
gt_root = './train/GT/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
size_rates = [1]  # multi-scale training

temp_path = './temp/'
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def visualize_post_pred(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_post.png'.format(kk)
        cv2.imwrite(temp_path + name, pred_edge_kk)

def visualize_prior_pred(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_prior.png'.format(kk)
        cv2.imwrite(temp_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(temp_path + name, pred_edge_kk)

print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    # adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    model.train()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()

            post_pred, prior_pred, latent_loss = model(images,gts)
            ## l2 regularizer the inference model
            reg_loss = l2_regularisation(model.xy_encoder) + \
                       l2_regularisation(model.x_encoder)
            reg_loss = opt.reg_weight * reg_loss
            latent_loss = opt.lat_weight * latent_loss
            loss_post = structure_loss(post_pred, gts)
            gen_loss_cvae = loss_post + latent_loss
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            gen_loss_gsnn = structure_loss(prior_pred, gts)
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn
            loss = gen_loss_cvae + gen_loss_gsnn + reg_loss
            loss.backward()
            # clip_gradient(optimizer, opt.clip)
            optimizer.step()

            visualize_post_pred(torch.sigmoid(post_pred))
            visualize_prior_pred(torch.sigmoid(prior_pred))
            visualize_gt(gts)

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), save_path + 'Model' + '_%d' % epoch + '.pth')
