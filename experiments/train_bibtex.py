import sys

import torch
import skimage.io
import math
import random
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.optim.lr_scheduler
import torch.cuda
import argparse, os
import time
from gspen.models import *
from experiment_utils import TensorBoard
from collections import defaultdict

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

TRAIN = 0
VAL = 1
COMBINED=3
TEST = 2

def labels2onehot(labels, gpu):
    if gpu:
        tensor_mod = torch.cuda
    else:
        tensor_mod = torch
    new_labels = labels.view(-1,1)
    onehot = tensor_mod.FloatTensor(len(new_labels), 2).fill_(0.)
    onehot.scatter_(1, new_labels, 1)
    return onehot.view(labels.size(0), -1)

def labels2beliefs(labels, pairs, gpu):
    if gpu:
        tensor_mod = torch.cuda
    else:
        tensor_mod = torch
    new_labels = labels.view(-1,1)
    unary = tensor_mod.FloatTensor(len(new_labels), 2).fill_(0.)
    unary.scatter_(1, new_labels, 1)
    pair_bel = tensor_mod.FloatTensor(len(pairs)*4).fill_(0.)
    for pair_ind,pair in enumerate(pairs):
        label_1 = labels[pair[0]]
        label_2 = labels[pair[1]]
        ind = pair_ind*4 + label_1 + label_2*2
        pair_bel[ind] = 1

    return torch.cat([unary.view(-1), pair_bel])

class BibtexDataset(Dataset):
    def __init__(self, mode, features_path, labels_path, pairs, normalize_features=False, mean=None, std=None, train_ratio=0.95, flip_prob=None):
        self.features = torch.load(features_path).type('torch.FloatTensor')
        self.labels = torch.load(labels_path).type('torch.LongTensor')
        self.flip_prob = None
        if normalize_features:
            if mean is None:
                self.feature_mean = self.features.mean(dim=0).view(1, -1)
                self.feature_std = self.features.std(dim=0).view(1, -1)
            else:
                self.feature_mean = mean
                self.feature_std = std
            self.features = (self.features - self.feature_mean)/(self.feature_std + 1e-6)
        else:
            self.feature_mean = None
            self.feature_std = None

        ind = int(len(self.features)*train_ratio)
        if mode == TRAIN:
            self.features = self.features[:ind, :]
            self.labels = self.labels[:ind, :]
            self.flip_prob = flip_prob
            if flip_prob is not None:
                print("USING FLIP PROB: ",flip_prob)
                self.bernoulli = torch.distributions.bernoulli.Bernoulli(probs=self.flip_prob)
        elif mode == VAL:
            self.features = self.features[ind:, :]
            self.labels = self.labels[ind:, :]

        if pairs is None:
            self.onehot_labels = labels2onehot(self.labels, False)
        else:
            self.onehot_labels = torch.stack([labels2beliefs(self.labels[i,:], pairs, False).squeeze() for i in range(self.labels.size(0))])

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        feat = self.features[idx, :]
        if self.flip_prob is not None:
            mask = self.bernoulli.sample(torch.Size([1836]))
            feat = feat*(1-mask) + (1-feat)*mask
        return feat, self.labels[idx, :], self.onehot_labels[idx, :]


class TDropout(nn.Module):
    def __init__(self, drop_factor):
        super(TDropout, self).__init__()
        self.drop_factor = drop_factor
        self.bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1-self.drop_factor)
        self.dropout_mask = None

    def train(self, mode=True):
        super(TDropout, self).train(mode)
        self.dropout_mask = None


    def eval(self):
        super(TDropout, self).eval()
        self.dropout_mask = None
        

    def forward(self, inp):
        if self.training and self.drop_factor > 0:
            if self.dropout_mask is None:
                self.dropout_mask = self.bernoulli.sample(inp.size())
                if inp.is_cuda:
                    self.dropout_mask = self.dropout_mask.cuda(async=True)
            inp = inp*self.dropout_mask
        return inp


class BibtexUnaryModel(nn.Module):
    def __init__(self, params):
        super(BibtexUnaryModel, self).__init__()
        dropout = params.get('unary_dropout', 0.5)
        num_layers = params.get('num_layers', 3)
        ignore_first_dropout = params.get('ignore_first_dropout', False)
        layers = [nn.Linear(1836, 150), nn.ReLU(inplace=True)]
        layers[0].weight.data.normal_(std=np.sqrt(2. / 1836))
        layers[0].bias.data.fill_(0.)
        if dropout is not None and not ignore_first_dropout:
            layers.insert(0, nn.Dropout(dropout))
        for layer in range(num_layers-2):
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(150, 150))
            layers[-1].weight.data.normal_(std=np.sqrt(2.0/150))
            layers[-1].bias.data.fill_(0.)
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(150, 159))
        layers[-1].weight.data.normal_(std=np.sqrt(2.0/150))
        layers[-1].bias.data.fill_(0.)
        self.model = nn.Sequential(*layers)
        

        self.finetune = params.get('unary_finetune', False)
        if self.finetune:
            for param in self.model[0].parameters():
                param.requires_grad=False
            for param in self.model[2].parameters():
                param.requires_grad=False

    def forward(self, inputs):
        result = torch.cuda.FloatTensor(inputs.size(0)*159, 2)
        result[:, 0].fill_(0.)
        result[:, 1].copy_(self.model(inputs).view(-1))
        return result.view(inputs.size(0), -1)

class BibtexUnaryTModel(nn.Module):
    def __init__(self, inp_size, params):
        super(BibtexUnaryTModel, self).__init__()
        first_dropout = params.get('t_first_dropout', 0.)
        dropout = params.get('t_dropout', 0.)
        num_layers = params.get('t_num_layers', 2)
        hidden_size = params.get('t_hidden_size', 16)
        layers = [nn.Linear(159, hidden_size, bias=False), nn.Softplus()]
        if first_dropout is not None and first_dropout > 0:
            layers.insert(0, TDropout(first_dropout))
        for _ in range(num_layers-2):
            if dropout > 0:
                layers.append(TDropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_size, 1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, beliefs):
        beliefs = beliefs.contiguous().view(-1, 2)[:, 1].contiguous().view(beliefs.size(0), 159)
        return self.model(beliefs).squeeze()


class BibtexPairModel(nn.Module):
    def __init__(self, params):
        super(BibtexPairModel, self).__init__()
        self.num_pairs = num_pairs = params.get('num_pairs')
        ignore_first_dropout = params.get('pair_ignore_first_dropout', False)
        num_layers = params.get('pair_num_layers', 2)
        dropout = params.get('pair_dropout', 0.5)
        input_size = 1836
        output_size = 4*num_pairs
        hidden_size = params.get('pair_hidden_size', 1836)
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True)]
        if dropout is not None and not ignore_first_dropout:
            layers.insert(0, nn.Dropout(dropout))
        for idx in range(num_layers-2):
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_size, 4*num_pairs))
        self.model = nn.Sequential(*layers)

        
    def forward(self, inputs):
        return self.model(inputs)


def test(model, dataset, params, threshold=None):
    batch_size = params.get('batch_size', 1000)
    gpu = params.get('gpu', False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    hamming = 0
    num_data = 0.0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    sanity_check_fs = []
    total_correct = 0
    total_vars = 0
    for inputs,labels, onehot_labels in dataloader:
        if gpu:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
        if threshold is None:
            predictions = model.predict(inputs)
        else:
            node_beliefs = model.calculate_beliefs(inputs)[:, :2*159].contiguous().view(-1, 2)[:, 1].view(-1, 159)
            predictions = (node_beliefs > threshold).long()
        correct = (predictions * labels).sum(1)
        total_correct += (predictions == labels).float().sum().item()
        total_vars += labels.numel()
        precision = (correct.float() / (predictions.sum(1).float()+0.000001))
        recall = (correct.float()/ (labels.sum(1).float()+0.000001))
        total_f1 = total_f1 + ((2*precision*recall)/(precision + recall + 1e-6)).sum()
        total_precision = total_precision + precision.sum()
        total_recall = total_recall + recall.sum()
    acc = total_correct/total_vars
    f1 = total_f1/len(dataset)
    precision = total_precision/len(dataset)
    recall = total_recall/len(dataset)
    return acc, precision.item(), recall.item(), f1.item()


def test_with_thresholds(model, dataset, params):
    batch_size = params.get('batch_size', 1000)
    gpu = params.get('gpu', False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    thresholds = np.arange(0.05, 0.80, 0.05)
    best_thresh = -1
    best_prec = best_recall = best_f1 = 0.
    total_accs = np.zeros(len(thresholds))
    num_vars = 0
    total_precisions = np.zeros(len(thresholds))
    total_recalls = np.zeros(len(thresholds))
    total_f1s = np.zeros(len(thresholds))
    hamming = 0
    num_data = 0.0
    for inputs,labels, onehot_labels in dataloader:
        if gpu:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
        num_vars += labels.numel()
        node_beliefs = model.calculate_beliefs(inputs)[:, :2*159].contiguous().view(-1, 2)[:, 1]
        for ind,threshold in enumerate(thresholds):
            predictions = (node_beliefs > threshold).long().view(-1, 159)
            correct = (predictions * labels).sum(1)
            precision = (correct.float() / (predictions.sum(1).float()+0.000001))
            recall = (correct.float()/ (labels.sum(1).float()+0.000001))
            total_accs[ind] += (predictions == labels).float().sum().item()
            total_recalls[ind] += recall.sum()
            total_precisions[ind] += precision.sum()
            total_f1s[ind] += ((2*precision*recall)/(precision + recall + 1e-6)).sum()
    accs = total_accs/num_vars
    precisions = total_precisions/len(dataset)
    recalls = total_recalls/len(dataset)
    f1s = total_f1s/len(dataset)
    best_ind = f1s.argmax()
    return (accs[best_ind], precisions[best_ind], recalls[best_ind], f1s[best_ind]), thresholds[best_ind]


def train(model, train_data, val_data, params, loggers):
    train_logger = loggers['train']
    val_logger = loggers['val']
    all_vals = params.get('return_all_vals', False)
    if all_vals:
        pred_logger = loggers['pred']
        data_logger = loggers['data']
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    verbose = params.get('verbose', False)
    training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    val_interval = params.get('val_interval', 5)
    working_dir = params['working_dir']
    train_data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=gpu, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=gpu, num_workers=1)
    clip_grad = params.get('clip_grad', None)
    clip_grad_norm = params.get('clip_grad_norm', None)
    tune_thresholds = params.get('tune_thresholds', False)
    opt = model.get_optimizer(params)
    if training_scheduler is not None:
        training_scheduler = training_scheduler(model_optimizer)
    end = start = 0 
    train_results = []
    val_results = []
    train_obj_vals = []
    train_inf_obj_vals = []
    use_cross_ent = params.get('use_cross_ent', False)
    best_acc = (0, 0,0,0)
    best_acc_epoch = -1
    best_thresh = -1
    for epoch in range(num_epochs):
        print("EPOCH", epoch+1, (end-start))
        train_logger.update_epoch()
        val_logger.update_epoch()
        if all_vals:
            pred_logger.update_epoch()
            data_logger.update_epoch()
        
        if epoch%val_interval == 0:
            if tune_thresholds:
                new_results, new_thresh = test_with_thresholds(model, train_data, params)
                train_results.append(new_results)
                print("TRAIN RESULTS: ",train_results[-1])
                train_logger.plot_for_current_epoch('F1', train_results[-1][-1])
                train_logger.plot_for_current_epoch('Accuracy', train_results[-1][0])
                print("BEST TRAIN THRESH: ", new_thresh)
                if val_data is not None:
                    new_results, new_thresh = test_with_thresholds(model, val_data, params)
                    val_results.append(new_results)
                    print("VAL RESULTS: ",val_results[-1])
                    print("BEST VAL THRESH: ", new_thresh)
                    val_logger.plot_for_current_epoch('F1', val_results[-1][-1])
                    val_logger.plot_for_current_epoch('Accuracy', val_results[-1][0])
            else:
                train_results.append(test(model, train_data, params))
                print("TRAIN RESULTS: ",train_results[-1])
                train_logger.plot_for_current_epoch('F1', train_results[-1][-1])
                if val_data is not None:
                    val_results.append(test(model, val_data, params))
                    print("VAL RESULTS: ",val_results[-1])
                    val_logger.plot_for_current_epoch('F1', val_results[-1][-1])
            if val_data is not None:
                if val_results[-1][-1] > best_acc[-1]:
                    best_acc = val_results[-1]
                    best_acc_epoch = epoch
                    if tune_thresholds:
                        best_thresh = new_thresh
                    print("NEW BEST ACCURACY FOUND, SAVING MODEL")
                    save_model(model, working_dir, 'model', params)
            else:
                if train_results[-1][-1] > best_acc[-1]:
                    best_acc = train_results[-1]
                    best_acc_epoch = epoch
                    if tune_thresholds:
                        best_thresh = new_thresh
                    print("NEW BEST ACCURACY FOUND, SAVING MODEL")
                    save_model(model, working_dir, 'model', params)


            save_model(model, working_dir, 'model_checkpoint', params)
            if val_data is not None:
                if tune_thresholds:
                    print("BEST VAL RESULTS (EPOCH %d, THRESH %f): "%(best_acc_epoch, best_thresh), best_acc)
                else:
                    print("BEST VAL RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
            else:
                if tune_thresholds:
                    print("BEST TRAIN RESULTS (EPOCH %d, THRESH %f): "%(best_acc_epoch, best_thresh), best_acc)
                else:
                    print("BEST TRAIN RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
        
        if training_scheduler is not None:
            training_scheduler.step()
        start = time.time() 
        avg_obj = 0
        obj_count = 0
        for batch_ind, all_inputs in enumerate(train_data_loader):
            inputs, labels, onehot_labels = all_inputs
            if use_cross_ent:
                if gpu:
                    inputs = inputs.cuda(async=True)
                    labels = labels.cuda(async=True)
                obj, inf_obj = model.calculate_obj(epoch, inputs, labels)
            else:
                if gpu:
                    inputs = inputs.cuda(async=True)
                    onehot_labels = onehot_labels.cuda(async=True)

                if all_vals:
                    obj, inf_obj, data_obj, num_iters = model.calculate_obj(epoch, inputs, onehot_labels)
                else:
                    obj, inf_obj = model.calculate_obj(epoch, inputs, onehot_labels)
            if verbose:
                print("\tBATCH %d OF %d: %f"%(batch_ind+1, len(train_data_loader), obj.item()))
            if batch_ind == 0:
                if all_vals:
                    data_logger.plot_for_current_epoch('Average Predicted Score', data_obj.item())
                    pred_logger.plot_for_current_epoch('Average Predicted Score', inf_obj.item())
                    train_logger.plot_for_current_epoch('Iterations Required for Inference Convergence', num_iters)
            obj.backward()
            avg_obj += obj.item()
            obj_count += 1
            if clip_grad is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            elif clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            opt.step()
        end = time.time()
        train_logger.plot_obj_val(avg_obj/obj_count)
        
    if tune_thresholds:
        new_results, new_thresh = test_with_thresholds(model, train_data, params)
        train_results.append(new_results)
        print("TRAIN RESULTS: ",train_results[-1])
        print("BEST TRAIN THRESH: ", new_thresh)
        train_logger.plot_for_current_epoch('F1', train_results[-1][-1])
        train_logger.plot_for_current_epoch('Accuracy', train_results[-1][0])
        if val_data is not None:
            new_results, new_thresh = test_with_thresholds(model, val_data, params)
            val_results.append(new_results)
            print("VAL RESULTS: ",val_results[-1])
            print("BEST VAL THRESH: ", new_thresh)
            val_logger.plot_for_current_epoch('F1', val_results[-1][-1])
            val_logger.plot_for_current_epoch('Accuracy', val_results[-1][0])
    else:
        train_results.append(test(model, train_data, params))
        print("FINAL TRAIN RESULTS: ",train_results[-1])
        train_logger.plot_for_current_epoch('F1', train_results[-1][-1])
        if val_data is not None:
            val_results.append(test(model, val_data, params))
            print("FINAL VAL RESULTS: ",val_results[-1])
            val_logger.plot_for_current_epoch('F1', val_results[-1][-1])
    if val_data is not None:
        if val_results[-1][-1] > best_acc[-1]:
            best_acc = val_results[-1]
            best_acc_epoch = epoch
            if tune_thresholds:
                best_thresh = new_thresh
            print("NEW BEST ACCURACY FOUND, SAVING MODEL")
            save_model(model, working_dir, 'model', params)
    else:
        if train_results[-1][-1] > best_acc[-1]:
            best_acc = train_results[-1]
            best_acc_epoch = epoch
            if tune_thresholds:
                best_thresh = new_thresh
            print("NEW BEST ACCURACY FOUND, SAVING MODEL")
            save_model(model, working_dir, 'model', params)
    save_model(model, working_dir, 'model_checkpoint', params)
    if val_data is not None:
        if tune_thresholds:
            print("BEST VAL RESULTS (EPOCH %d, THRESH %f): "%(best_acc_epoch, best_thresh), best_acc)
        else:
            print("BEST VAL RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
    else:
        if tune_thresholds:
            print("BEST TRAIN RESULTS (EPOCH %d, THRESH %f): "%(best_acc_epoch, best_thresh), best_acc)
        else:
            print("BEST TRAIN RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
    return train_obj_vals, train_inf_obj_vals, train_results, val_results

def save_model(model, path, base_name, params):
    model_path = os.path.join(path, 'unary_%s'%base_name)
    model.save_unary(model_path)

    if params['model'] in ['struct', 'gspen']:
        model_path = os.path.join(path, 'pair_%s'%base_name)
        model.save_pair(model_path)

    if params['model'] in ['spen', 'gspen']:
        if params.get('t_version') in ['full_t_v1', 'full_t_v2', 't_v2', 't_v3'] and params['model'] != 'spen':
            model_path = os.path.join(path, 't_%s'%base_name)
            model.save_t(model_path)
        else:
            model_path = os.path.join(path, 'unary_t_%s'%base_name)
            model.save_unary_t(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training models on Bibtex dataset')
    parser.add_argument('model', choices=['unary', 'struct', 'spen', 'gspen'])
    parser.add_argument('feature_file')
    parser.add_argument('label_file')
    parser.add_argument('working_dir')
    parser.add_argument('--t_version', choices=['t_v1', 'full_t_v1', 't_v2', 'full_t_v2', 't_v3'])
    parser.add_argument('--t', choices=['linear', 'quad', 'skinny_quad', 'mlp'])
    parser.add_argument('--t_unary', choices=['linear', 'quad', 'skinny_quad', 'mlp'])
    parser.add_argument('--t_pair', choices=['linear', 'quad', 'skinny_quad', 'mlp'])
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--h_lr', type=float)
    parser.add_argument('--unary_lr', type=float)
    parser.add_argument('--unary_mom', type=float, default=0.)
    parser.add_argument('--combined_mom', type=float, default=0.)
    parser.add_argument('--pair_mom', type=float, default=0.)
    parser.add_argument('--unary_num_layers', type=int, default=2)
    parser.add_argument('--pair_num_layers', type=int, default=2)
    parser.add_argument('--pair_lr', type=float)
    parser.add_argument('--t_unary_lr', type=float)
    parser.add_argument('--t_unary_mom', type=float, default=0.)
    parser.add_argument('--t_pair_lr', type=float)
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--unary_hidden_size', type=int, default=1836)
    parser.add_argument('--pair_hidden_size', type=int, default=1836)
    parser.add_argument('--num_inf_itrs', type=int, default=100)
    parser.add_argument('--inf_lr', type=float, default=0.05)
    parser.add_argument('--use_sqrt_decay', action='store_true')
    parser.add_argument('--use_linear_decay', action='store_true')
    parser.add_argument('--use_relu', action='store_true')
    parser.add_argument('--inf_method', choices=['fw', 'emd', 'gd'], default='fw')
    parser.add_argument('--use_loss_aug', action='store_true')
    parser.add_argument('--pretrain_unary')
    parser.add_argument('--pretrain_pair')
    parser.add_argument('--pretrain_t')
    parser.add_argument('--pretrain_unary_t')
    parser.add_argument('--pretrain_pair_t')
    parser.add_argument('--unary_wd', type=float, default=0.)
    parser.add_argument('--pair_wd', type=float, default=0.)
    parser.add_argument('--t_unary_wd', type=float, default=0.)
    parser.add_argument('--t_pair_wd', type=float, default=0.)
    parser.add_argument('--mp_eps', type=float, default=0.)
    parser.add_argument('--mp_itrs', type=int)
    parser.add_argument('--inf_mode', choices=['lp', 'mp', 'md', 'ilp'])
    parser.add_argument('--use_entropy', action='store_true')
    parser.add_argument('--t_dropout', type=float, default=0)
    parser.add_argument('--unary_dropout', type=float)
    parser.add_argument('--pair_dropout', type=float)
    parser.add_argument('--inf_eps', type=float)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_feature_file')
    parser.add_argument('--test_label_file')
    parser.add_argument('--no_t_pots', action='store_true')
    parser.add_argument('--unary_finetune', action='store_true')
    parser.add_argument('--t_use_softplus', action='store_true')
    parser.add_argument('--clip_grad', type=float)
    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--use_cross_ent', action='store_true')
    parser.add_argument('--tune_thresholds', action='store_true')
    parser.add_argument('--fix_features', action='store_true')
    parser.add_argument('--fix_unaries', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=0.95)
    parser.add_argument('--freeze_unary', action='store_true')
    parser.add_argument('--num_pairs', type=int)
    parser.add_argument('--use_sigmoid', action='store_true')
    parser.add_argument('--load_from_unary')
    parser.add_argument('--ignore_first_dropout', action='store_true')
    parser.add_argument('--pair_ignore_first_dropout', action='store_true')
    parser.add_argument('--entropy_coef', type=float, default=1)
    parser.add_argument('--return_all_vals', action='store_true')
    parser.add_argument('--t_hidden_size', type=int, default=16)
    parser.add_argument('--t_num_layers', type=int, default=2)
    parser.add_argument('--t_first_dropout', type=float)
    parser.add_argument('--flip_prob', type=float)
    parser.add_argument('--loss_aug_coef', type=float, default=1.)
    parser.add_argument('--use_random_pairs', action='store_true')
    parser.add_argument('--t_lr', type=float)
    parser.add_argument('--t_mom', type=float, default=0.)
    parser.add_argument('--t_normalize_pots', action='store_true')
    parser.add_argument('--pot_scaling_factor', type=float, default=1.)

    args = parser.parse_args()
    
    if args.test:
        args.flip_prob = None
    params = vars(args)

    unary_model = BibtexUnaryModel(params)
    num_nodes = 159
    num_vals = 2

    if args.model in ['struct', 'gspen']:
        max_label = 134 #Found in another script
        pairs = []
        for node in range(159):
            if node == max_label:
                continue
            elif node < max_label:
                pairs.append((node, max_label))
            else:
                pairs.append((max_label, node))
        params['num_pairs'] = len(pairs)
    else:
        pairs = None
    test_data = None
    train_data = BibtexDataset(TRAIN, args.feature_file, args.label_file, pairs, normalize_features=args.normalize_features, train_ratio=args.train_ratio, flip_prob=args.flip_prob)
    val_data = BibtexDataset(VAL, args.feature_file, args.label_file, pairs, normalize_features=args.normalize_features, mean=train_data.feature_mean, std=train_data.feature_std, train_ratio=args.train_ratio)
    if args.test:
        test_data = BibtexDataset(TEST, args.test_feature_file, args.test_label_file, pairs, normalize_features=args.normalize_features, mean=train_data.feature_mean, std=train_data.feature_std)
    if args.model in ['struct', 'gspen']:
        pair_model = BibtexPairModel(params)
    if args.model in ['spen', 'gspen']:
        unary_t_constructor = BibtexUnaryTModel
        params['no_t_pots'] = True
        pair_t_constructor = None

        if args.t_version == 't_v1':
            t_model = SplitTModelV1(unary_t_constructor, pair_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 'full_t_v1':
            t_model = TModelV1(unary_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v2':
            if args.mode == 'unary_t':
                t_model = TModelV2(unary_t_constructor, num_nodes, pairs, num_vals, params)
            else:
                t_model = SplitTModelV2(unary_t_constructor, pair_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 'full_t_v2':
            t_model = TModelV2(unary_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v3':
            t_model = TModelV3(t_constructor, num_nodes, pairs, num_vals, params)

        if args.model == 'spen':
            model = SPENModel(unary_model, t_model, num_nodes, num_vals, params)
        elif args.model == 'gspen':
            model = GSPENModel(unary_model, pair_model, t_model, num_nodes, pairs, num_vals, params)

    elif args.model == 'unary':
        model = UnaryModel(unary_model, num_nodes, num_vals, params)
    elif args.model == 'struct':
        model = StructModel(unary_model, pair_model, num_nodes, pairs, num_vals, params)
    print("MODEL: ",model)
    if args.pretrain_unary is not None:
        model.load_unary(args.pretrain_unary)
        if not args.unary_finetune:
            for param in model.parameters():
                param.requires_grad=True
    if args.pretrain_pair is not None:
        model.load_pair(args.pretrain_pair)
    if args.pretrain_t is not None:
        model.load_t(args.pretrain_t)
    if args.pretrain_unary_t is not None:
        model.load_unary_t(args.pretrain_unary_t)
    if args.pretrain_pair_t is not None:
        model.load_pair_t(args.pretrain_pair_t)

    

    if args.test:
        print("FINDING BEST THRESHOLD ON VALIDATION DATA...")
        (val_acc, val_p,val_r,val_f1), thresh = test_with_thresholds(model, val_data, params)
        print("TESTING ON TRAINING DATA...")
        train_acc, train_p,train_r,train_f1 = test(model, train_data, params, threshold=thresh)
        print("TESTING ON TEST DATA...")
        test_acc, test_p,test_r,test_f1 = test(model, test_data, params, threshold=thresh)

        print("TESTING ON TRAINING DATA (THRESH 0.5)...")
        train_acc_5, train_p_5,train_r_5,train_f1_5 = test(model, train_data, params, threshold=0.5)
        print("FINDING BEST THRESHOLD ON VALIDATION DATA (THRESH 0.5)...")
        val_acc_5, val_p_5,val_r_5,val_f1_5 = test(model, val_data, params, threshold=0.5)
        print("TESTING ON TEST DATA (THRESH 0.5)...")
        test_acc_5, test_p_5,test_r_5,test_f1_5 = test(model, test_data, params, threshold=0.5)
        print("FINAL RESULTS: (THRESHOLD %f)"%thresh)
        print("\tTRAIN: %f, %f, %f, %f"%(train_acc, train_p, train_r, train_f1))
        print("\tVAL:   %f, %f, %f, %f"%(val_acc, val_p, val_r, val_f1))
        print("\tTEST:  %f, %f, %f, %f"%(test_acc, test_p, test_r, test_f1))

        print("FINAL RESULTS: (THRESHOLD 0.5)")
        print("\tTRAIN: %f, %f, %f, %f"%(train_acc_5, train_p_5, train_r_5, train_f1_5))
        print("\tVAL:   %f, %f, %f, %f"%(val_acc_5, val_p_5, val_r_5, val_f1_5))
        print("\tTEST:  %f, %f, %f, %f"%(test_acc_5, test_p_5, test_r_5, test_f1_5))
            

    else:
        train_log_dir = os.path.join(args.working_dir, 'train')
        val_log_dir = os.path.join(args.working_dir, 'val')
        pred_log_dir = os.path.join(args.working_dir, 'pred')
        data_log_dir = os.path.join(args.working_dir, 'data')
        train_logger = TensorBoard(train_log_dir)
        val_logger = TensorBoard(val_log_dir)
        pred_logger = TensorBoard(pred_log_dir)
        data_logger = TensorBoard(data_log_dir)
        loggers = {
            'train':train_logger,
            'val':val_logger,
            'pred':pred_logger,
            'data':data_logger,
        }
        train_obj_vals, train_inf_obj_vals, train_scores, val_scores = train(model, train_data, val_data, params, loggers)

        train_logger.close()
        val_logger.close()
        pred_logger.close()
        data_logger.close()