#!/usr/bin/env python
"""


2020-03-03 09:08:28
"""
import torch
import torch.nn as nn
from addict import Dict
import utils

from utils import NormalNLLLoss, noise_sample

def train_dcgan(models, iterator, optimizers, loss_fun, device, train_args, model_args, elapsed_epochs, feature_extraction=None, log_this_epoch=False):

    netD = models.d
    netG = models.g
    optimizerD = optimizers.d
    optimizerG = optimizers.g
    criterion = models.d.criterion
    real_label = 1
    fake_label = 0
    epoch_losses = Dict({"d": 0, "g": 0, "matching": 0})
    losses = Dict({"d": 0, "g": 0, "matching": 0})
    nz = model_args["noise_dim"]

    for i, data in enumerate(iterator, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real

        netD.zero_grad()
        data = [data["real_imgs"], data["metos"]]
        
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        # import pdb
        # pdb.set_trace()
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, device=device)
        fake = netG(data[1].to(device), noise)
        # fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        losses.matching = loss_fun(fake, real_cpu)
        losses.g = errG
        losses.d = errD

        if log_this_epoch:
            for k in epoch_losses.keys():
                epoch_losses[k] += losses[k].item() / len(iterator)
        else:
            epoch_losses = None

    return models, epoch_losses, optimizers


def train(models, iterator, optimizers, loss_fun, device, train_args, model_args, elapsed_epochs, feature_extraction=None, log_this_epoch=False):

    batch_size = train_args["batch_size"]
    noise_dim = model_args["noise_dim"]
    disc_step = train_args["num_D_accumulations"]

    epoch_losses = Dict({"d": 0, "g": 0, "matching": 0, "mi_dis" : 0, "mi_con" : 0, "metos" : 0})
    models.g.train()
    models.d.train()
    iterator_len = len(iterator)
    disc_noise = torch.randn((iterator_len, disc_step, batch_size, noise_dim)).to(device)
    gen_noise = torch.randn((iterator_len, batch_size, noise_dim)).to(device)

    # Loss for discrete latent code.
    criterionQ_dis = nn.CrossEntropyLoss()
    # Loss for continuous latent code.
    criterionQ_con = NormalNLLLoss()
    # Loss for metos prediction
    criterion_metos = nn.MSELoss()

    for idx, sample in enumerate(iterator):
        losses = Dict({"d": 0, "g": 0, "matching": 0, "mi_dis" : 0, "mi_con" : 0, "metos" : 0})
        x = sample["metos"]#.to(device)
        y = sample["real_imgs"]#s.to(device)

        # update discriminator
        losses.d = 0
        for k in range(disc_step):
            optimizers.d.zero_grad()
            #noise = torch.randn(x.shape[0], noise_dim).to(device)
            if model_args["infogan"]:
                noise, dis_idx = noise_sample(model_args['num_dis_c'], model_args['dis_c_dim'], model_args['num_con_c'], model_args['num_z'], x.shape[0], device)
                noise = noise.squeeze()
            else:
                noise = disc_noise[idx, k, :x.shape[0]] #x.shape[0] represents the true batch_size (useful for the last batch especially)
            if model_args["concat_noise_metos"]:
                noise = torch.cat([noise, x], dim=1)
            y_hat = models.g(x, noise)
            losses.d = models.d.compute_loss(y, 1) + models.d.compute_loss(y_hat.detach(), 0)

            losses.d.backward()
            total_steps = elapsed_epochs * iterator_len + idx * disc_step + k
            optimizers.d = utils.optim_step(
                optimizers.d, train_args["optimizer"], total_steps, idx * disc_step + k
            )

        # update generator
        optimizers.g.zero_grad()
        #noise = torch.randn(x.shape[0], noise_dim).to(device)
        #noise = gen_noise[idx, :x.shape[0]] #x.shape[0] represents the true batch_size (useful for the last batch especially)
        #y_hat = models.g(x, noise)
        losses.g = models.d.compute_loss(y_hat, 1)
        if feature_extraction is None:
            losses.matching = loss_fun(y_hat, y)
        else:
            #import pdb
            rgb_y_hat = torch.cat([y_hat] * 3, dim=1) #converts grayscale to RGB by replicating the single channel 3 times
            rgb_y = torch.cat([y] * 3, dim=1)
            rgb_normalized_y_hat = []
            rgb_normalized_y = []
            for j in range(x.shape[0]):
                rgb_normalized_y_hat.append(feature_extraction["transformations"](rgb_y_hat[j].cpu()))
                rgb_normalized_y.append(feature_extraction["transformations"](rgb_y[j].cpu()))
            #pdb.set_trace()
            rgb_y_hat, rgb_y = torch.stack(rgb_normalized_y_hat).to(rgb_y_hat.device), torch.stack(rgb_normalized_y).to(rgb_y.device)
            feature_maps_y_hat = feature_extraction["extractor"](rgb_y_hat)
            feature_maps_y = feature_extraction["extractor"](rgb_y)
            losses.matching = loss_fun(feature_maps_y_hat, feature_maps_y)

        total_loss_g = train_args["lambda_gan"] * losses.g + train_args["lambda_L"] * losses.matching

        if model_args["infogan"] or model_args["predict_metos"]:
            models.d.discriminator_head = False
            q_logits, q_mu, q_var = models.d(y_hat)
            target = torch.LongTensor(dis_idx).to(device)
            dis_loss = torch.zeros(total_loss_g.shape).to(device)
            con_loss = torch.zeros(total_loss_g.shape).to(device)
            metos_loss = torch.zeros(total_loss_g.shape).to(device)
            import pdb
            #pdb.set_trace()
            for j in range(model_args['num_dis_c']):
                left_index = j*model_args['dis_c_dim']
                dis_loss += criterionQ_dis(q_logits[:, left_index : left_index + model_args['dis_c_dim']], target[j])
            
            if (model_args['num_con_c'] != 0):
                left_index = model_args['num_z']+ model_args['num_dis_c']*model_args['dis_c_dim']
                con_loss = criterionQ_con(noise[:, left_index : left_index + model_args["num_con_c"]].view(-1, model_args['num_con_c']), q_mu[:model_args['num_con_c']], q_var[:model_args['num_con_c']])*0.1

            if model_args["predict_metos"]:
                metos_loss = criterion_metos(q_mu[model_args['num_con_c']:], x)

            losses.mi_dis, losses.mi_con, losses.metos = dis_loss, con_loss, metos_loss
            total_loss_g+= train_args["lambda_infogan"] * (dis_loss + con_loss)
            total_loss_g+= train_args["lambda_metos"] * metos_loss

        total_loss_g.backward()

        optimizers.g = utils.optim_step(
            optimizers.g, train_args["optimizer"], total_steps, idx
        )

        if log_this_epoch:
            for k in epoch_losses.keys():
                if isinstance(losses[k], (int, float)):
                    loss_val = losses[k]
                else:
                    loss_val = losses[k].item()
                epoch_losses[k] += loss_val / len(iterator)
        else:
            epoch_losses = None

    return models, epoch_losses, optimizers
