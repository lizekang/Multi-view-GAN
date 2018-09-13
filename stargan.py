import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)
os.makedirs('summary', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--train_step', type=int, default=100000, help='epoch to start training from')
parser.add_argument('--test_step', type=int, default=10, help='epoch to start training from')
parser.add_argument('--n_train_step', type=int, default=200, help='number of epochs of training')
parser.add_argument('--data_path', type=str, default="list.txt", help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--retrain', type=bool, default=False, help='if retrain')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=15, help='number of image channels')
parser.add_argument('--c_dims', type=int, default=10, help='number of views')
parser.add_argument('--sample_interval', type=int, default=100,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
parser.add_argument('--summary_path', type=str, default='./summary', help='path for summary')
parser.add_argument('--residual_blocks', type=int, default=6, help='number of residual blocks in generator')
parser.add_argument('--n_critic', type=int, default=5, help='number of training iterations for WGAN discriminator')
opt = parser.parse_args()
print(opt)

writer = SummaryWriter(opt.summary_path)
c_dim = int(opt.c_dims)
img_shape = (opt.channels, opt.img_height, opt.img_width)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_cycle = torch.nn.L1Loss()


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10

# Initialize generator and discriminator
generator = GeneratorResNet(img_shape=img_shape, res_blocks=opt.residual_blocks, c_dim=c_dim)
discriminator = Discriminator(img_shape=(3, opt.img_height, opt.img_width), c_dim=c_dim)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_cycle.cuda()

if not opt.retrain:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator.pth'))
    discriminator.load_state_dict(torch.load('saved_models/discriminator.pth'))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
train_transforms = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(MultiViewDataset(opt.data_path, batch_size=opt.batch_size, train_step=opt.train_step, transform=train_transforms),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_transforms = [  transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

val_dataloader = DataLoader(MultiViewDataset(opt.data_path, batch_size=10, train_step=opt.test_step, transform=val_transforms),
                            batch_size=10, shuffle=True, num_workers=1)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    print(alpha.size(), real_samples.size(), fake_samples.size())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_images(steps_done):
    """Saves a generated sample of domain translations"""
    img1, img2, img3, img4, img5, real_image, label = next(iter(val_dataloader))
    val_imgs1 = Variable(img1.type(Tensor))
    val_imgs2 = Variable(img2.type(Tensor))
    val_imgs3 = Variable(img3.type(Tensor))
    val_imgs4 = Variable(img4.type(Tensor))
    val_imgs5 = Variable(img5.type(Tensor))
    val_real_images = Variable(real_image.type(Tensor))
    val_labels = Variable(label.type(Tensor))
    img_samples = None
    for i in range(10):
        img1, img2, img3, img4, img5, real_image, label = val_imgs1[i], val_imgs2[i], val_imgs3[i], val_imgs4[i], \
                                                          val_imgs5[i], val_real_images[i], val_labels[i]
        # Repeat for number of label changes
        imgs = real_image.repeat(8, 1, 1, 1)
        labels = label.repeat(8, 1)
        # Make changes to labels
        for j in range(8):
            labels[j] = random.randint(c_dim)

        # Generate translations
        gen_imgs = generator(img1, img2, img3, img4, img5, labels)
        # Concatenate images by width
        gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
        img_sample = torch.cat((real_image.data, gen_imgs), -1)
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

    save_image(img_samples.view(1, *img_samples.shape), 'images/%s.png' % steps_done, normalize=True)


# ----------
#  Training
# ----------

saved_samples = []
start_time = time.time()
for i, (img1, img2, img3, img4, img5, real_image, label) in enumerate(dataloader):

    # Model inputs
    imgs1 = Variable(img1.type(Tensor))
    imgs2 = Variable(img2.type(Tensor))
    imgs3 = Variable(img3.type(Tensor))
    imgs4 = Variable(img4.type(Tensor))
    imgs5 = Variable(img5.type(Tensor))
    real_images = Variable(real_image.type(Tensor))
    labels = Variable(label.type(Tensor))

    # Sample labels as generator inputs
    # sampled_c = Variable(Tensor(np.random.randint(0, 2, (real_images.size(0), c_dim))))
    sampled_c = Variable(Tensor(np.random.normal(0, 1, (real_images.size(0), c_dim))))
    # Generate fake batch of images
    fake_imgs = generator(imgs1, imgs2, imgs3, imgs4, imgs5, sampled_c)

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Real images
    # real_validity, pred_cls = discriminator(real_images)
    real_validity, _ = discriminator(real_images)
    # Fake images
    fake_validity, _ = discriminator(fake_imgs.detach())
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_imgs.data)
    # Adversarial loss
    loss_D_adv = - torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    # Classification loss
    # loss_D_cls = criterion_cls(pred_cls, labels)
    # Total loss
    # loss_D = loss_D_adv + lambda_cls * loss_D_cls
    loss_D = loss_D_adv
    loss_D.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()

    # Every n_critic times update generator
    if i % opt.n_critic == 0:

        # -----------------
        #  Train Generator
        # -----------------

        # Translate and reconstruct image
        gen_imgs = generator(imgs1, imgs2, imgs3, imgs4, imgs5, sampled_c)
        # recov_imgs = generator(gen_imgs, labels)
        # Discriminator evaluates translated image
        # fake_validity, pred_cls = discriminator(gen_imgs)
        fake_validity, _ = discriminator(gen_imgs)
        # Adversarial loss
        loss_G_adv = -torch.mean(fake_validity)
        # Classification loss
        # loss_G_cls = criterion_cls(pred_cls, sampled_c)
        # Reconstruction loss
        # loss_G_rec = criterion_cycle(recov_imgs, real_images)
        # Total loss
        # loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec
        loss_G = loss_G_adv

        loss_G.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        steps_done = i
        steps_left = opt.train_step - steps_done
        time_left = datetime.timedelta(seconds=steps_left * (time.time() - start_time) / (steps_done + 1))

        # Print log
        # sys.stdout.write(
        #     "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f] ETA: %s" %
        #     (epoch, opt.n_epochs,
        #      i, len(dataloader),
        #      loss_D_adv.item(), loss_D_cls.item(),
        #      loss_G.item(), loss_G_adv.item(),
        #      loss_G_cls.item(), loss_G_rec.item(),
        #      time_left))

        # If at sample interval sample and save image
        if steps_done % opt.sample_interval == 0:
            sample_images(steps_done)

    if opt.checkpoint_interval != -1 and i % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator.pth')
        torch.save(discriminator.state_dict(), 'saved_models/discriminator.pth')
