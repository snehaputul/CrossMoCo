from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
from tqdm import tqdm

import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18, resnet101, resnet152, resnet34
from torch.utils.data import DataLoader

from datasets.data import ShapeNetRender, ScanObjectNNSVM
from models.dgcnn import DGCNN, ResNet, DGCNN_partseg
from util import IOStream, AverageMeter


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


class ModelMoCo(nn.Module):
    def __init__(self, dim=256, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=False, device='cpu'):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.device = device
        print('value of m:', self.m)

        if args.model == 'dgcnn':  # model initialization for point cloud
            self.point_model_q = DGCNN(args).to(device)
            self.point_model_k = DGCNN(args).to(device)

        for param_q, param_k in zip(self.point_model_q.parameters(), self.point_model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize and equalize params
            param_k.requires_grad = False

        if args.img_model == 'resnet18':
            self.img_model_q = ResNet(resnet18(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=512, output_dim=args.output_dim).to(device)
            self.img_model_k = ResNet(resnet18(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=512, output_dim=args.output_dim).to(device)
        elif args.img_model == 'resnet50':
            self.img_model_q = ResNet(resnet50(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=2048, output_dim=args.output_dim).to(device)
            self.img_model_k = ResNet(resnet50(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=2048, output_dim=args.output_dim).to(device)
        elif args.img_model == 'resnet34':
            self.img_model_q = ResNet(resnet34(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=512, output_dim=args.output_dim).to(device)
            self.img_model_k = ResNet(resnet34(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=512, output_dim=args.output_dim).to(device)
        elif args.img_model == 'resnet101':
            self.img_model_q = ResNet(resnet101(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=2048, output_dim=args.output_dim).to(device)
            self.img_model_k = ResNet(resnet101(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=2048, output_dim=args.output_dim).to(device)
        elif args.img_model == 'resnet152':
            self.img_model_q = ResNet(resnet152(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=2048, output_dim=args.output_dim).to(device)
            self.img_model_k = ResNet(resnet152(pretrained=args.pre_trained), out_layer= args.out_layer, feat_dim=2048, output_dim=args.output_dim).to(device)

        for param_q, param_k in zip(self.img_model_q.parameters(), self.img_model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize and equalize params
            param_k.requires_grad = False

        # define loss
        self.criterion = NTXentLoss(temperature=0.1).to(device)

        # create the queue
        self.register_buffer("queue_point", torch.randn(dim, K))  # for point model
        self.queue_point = nn.functional.normalize(self.queue_point, dim=0)

        self.register_buffer("queue_img", torch.randn(dim, K))  # for img model
        self.queue_img = nn.functional.normalize(self.queue_img, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        if torch.cuda.is_available():
            idx_shuffle = torch.randperm(x.shape[0]).to(self.device)
        else:
            idx_shuffle = torch.randperm(x.shape[0])

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.point_model_q.parameters(),
                                    self.point_model_k.parameters()):  # momentum update for point model
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.img_model_q.parameters(),
                                    self.img_model_k.parameters()):  # momentum update for image model
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, k_point, k_img):
        batch_size = k_point.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_point[:, ptr:ptr + batch_size] = k_point.t()  # transpose
        self.queue_img[:, ptr:ptr + batch_size] = k_img.t()  # transpose

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, data_t1, data_t2, new_imgs):

        # compute query features for point model
        data_t1 = data_t1.transpose(2, 1).contiguous()
        _, q_point, _ = self.point_model_q(data_t1)  # queries: NxC
        q_point = nn.functional.normalize(q_point, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            data_t2 = data_t2.transpose(2, 1).contiguous()
            data_t2_, idx_unshuffle = self._batch_shuffle_single_gpu(data_t2)

            _, k_point, _ = self.point_model_k(data_t2_)  # keys: NxC
            k_point = nn.functional.normalize(k_point, dim=1)  # already normalized

            # undo shuffle
            k_point = self._batch_unshuffle_single_gpu(k_point, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_point, k_point]).unsqueeze(
            -1)  # torch.Size([512, 128]), torch.Size([512, 128])
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk',
                             [q_point,
                              self.queue_point.clone().detach()])  # torch.Size([512, 128]), torch.Size([128, 4096])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        if torch.cuda.is_available():
            labels = labels.to(self.device)

        if torch.cuda.is_available():
            loss_point = nn.CrossEntropyLoss().to(self.device)(logits, labels)
        else:
            loss_point = nn.CrossEntropyLoss()(logits, labels)

        # compute query features for img model
        q_img = self.img_model_q(new_imgs[0])  # queries: NxC
        q_img = nn.functional.normalize(q_img, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            data_t2_, idx_unshuffle = self._batch_shuffle_single_gpu(new_imgs[1])

            k_img = self.img_model_k(data_t2_)  # keys: NxC
            k_img = nn.functional.normalize(k_img, dim=1)  # already normalized

            # undo shuffle
            k_img = self._batch_unshuffle_single_gpu(k_img, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_img, k_img]).unsqueeze(
            -1)  # torch.Size([512, 128]), torch.Size([512, 128])
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk',
                             [q_img,
                              self.queue_img.clone().detach()])  # torch.Size([512, 128]), torch.Size([128, 4096])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        if torch.cuda.is_available():
            labels = labels.to(self.device)

        if torch.cuda.is_available():
            loss_img = nn.CrossEntropyLoss().to(self.device)(logits, labels)
        else:
            loss_img = nn.CrossEntropyLoss()(logits, labels)

        loss_multi = self.criterion(q_point, q_img)  # multi modal loss

        loss = loss_point + loss_img + loss_multi

        return loss, q_point, k_point, q_img, k_img

    def forward(self, data_t1, data_t2, new_imgs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # symmetric loss
            # loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            # loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            # loss = loss_12 + loss_21
            # k = torch.cat([k1, k2], dim=0)
            pass
        else:  # asymmetric loss
            loss, q_point, k_point, q_img, k_img = self.contrastive_loss(data_t1, data_t2, new_imgs)

        self._dequeue_and_enqueue(k_point, k_img)

        return loss


def train(args, io):
    wandb.init(project="CrossPoint", name=args.exp_name)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_loader = DataLoader(ShapeNetRender(transform, n_imgs=2), num_workers=10,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
    moco = ModelMoCo(K=args.K, device=device, dim=args.output_dim, m=args.m)
    if torch.cuda.is_available():
        moco = moco.to(device)

    wandb.watch(moco.point_model_q)

    if args.resume:
        # model.load_state_dict(torch.load(args.model_path))
        print("Model Loaded !!")

    parameters = list(moco.point_model_q.parameters()) + list(moco.img_model_q.parameters())

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature=0.1).to(device)

    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()

        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        # train_imid_losses = AverageMeter()
        # train_cmid_losses = AverageMeter()
        # train_multi_losses = AverageMeter()

        moco.point_model_q.train()
        moco.img_model_q.train()
        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for i, ((data_t1, data_t2), imgs) in enumerate(tqdm(train_loader)):
            data_t1, data_t2 = data_t1.to(device), data_t2.to(device)
            new_imgs = []
            for x in imgs:
                new_imgs.append(x.to(device))
            batch_size = data_t1.size()[0]

            opt.zero_grad()
            loss = moco(data_t1, data_t2, new_imgs)
            loss.backward()

            opt.step()

            train_losses.update(loss.item(), batch_size)
            # train_imid_losses.update(loss_imid.item(), batch_size)
            # train_cmid_losses.update(loss_cmid.item(), batch_size)
            # train_multi_losses.update(multimodal_loss.item(), batch_size)

            if i % args.print_freq == 0:
                print('Epoch (%d), Batch(%d/%d), loss: %.6f ' % (
                    epoch, i, len(train_loader), train_losses.avg))

        wandb_log['Train Loss'] = train_losses.avg
        # wandb_log['Train IMID Loss'] = train_imid_losses.avg
        # wandb_log['Train CMID Loss'] = train_cmid_losses.avg

        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        io.cprint(outstr)

        # Testing

        train_val_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=1024), batch_size=20, shuffle=True)
        test_val_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=1024), batch_size=20, shuffle=True)

        feats_train = []
        labels_train = []
        moco.point_model_q.eval()

        for i, (data, label) in enumerate(train_val_loader):
            labels = label  # list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = moco.point_model_q(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (data, label) in enumerate(test_val_loader):
            labels = label  # list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = moco.point_model_q(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)

        model_tl = SVC(C=0.1, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        wandb_log['Linear Accuracy'] = test_accuracy
        print(f"Linear Accuracy : {test_accuracy}")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model.pth'.format(epoch=epoch))
            torch.save([moco.point_model_q.state_dict(), moco.point_model_k.state_dict()], save_file)

            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                               'img_model_best.pth')
            torch.save([moco.img_model_q.state_dict(), moco.img_model_k.state_dict()], save_img_model_file)

        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model.pth'.format(epoch=epoch))
            torch.save([moco.point_model_q.state_dict(), moco.point_model_k.state_dict()], save_file)

            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                               'img_model_best.pth')
            torch.save([moco.img_model_q.state_dict(), moco.img_model_k.state_dict()], save_img_model_file)
        wandb_log['Best Acc'] = best_acc
        wandb.log(wandb_log)

    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(moco.point_model_q.state_dict(), save_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                       'img_model_last.pth')
    torch.save(moco.img_model_q.state_dict(), save_img_model_file)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='which gpu to use')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--K', type=int, default=4096, help='Momentum queue size')
    parser.add_argument('--img_model', type=str, default='resnet50')
    parser.add_argument('--pre_trained', action = 'store_true')
    parser.add_argument('--output_dim', type=int, default=256, help='output dimension of model')
    parser.add_argument('--m', type=float, default=0.99, help='momentum value for momentum update')
    parser.add_argument('--out_layer', type=int, default=2, help='number of output layer')
    args = parser.parse_args()


    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
