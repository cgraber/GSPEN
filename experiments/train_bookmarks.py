import sys

import torch
import torchfile
import math
import random
from collections import defaultdict
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

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

TRAIN = 0
VAL = 1
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

class BookmarksDataset(Dataset):
    def __init__(self, mode, data_dir, pairs, flip_prob=None):
        self.flip_prob = None
        features = np.zeros((0, 2150))
        labels = np.zeros((0, 208))
        if mode == TRAIN:
            self.flip_prob = flip_prob
            if flip_prob is not None:
                print("USING FLIP PROB: ",flip_prob)
                self.bernoulli = torch.distributions.bernoulli.Bernoulli(probs=self.flip_prob)
            for i in range(1, 6):
                path = os.path.join(data_dir, 'bookmarks-train-%d.torch'%i)
                data = torchfile.load(path)
                labels = np.concatenate((labels, data[b'labels']), axis=0)
                features = np.concatenate((features, data[b'data'][:, 0:2150]), axis=0)
        elif mode == VAL:
            path = os.path.join(data_dir, 'bookmarks-dev.torch')
            data = torchfile.load(path)
            labels = np.concatenate((labels, data[b'labels']), axis=0)
            features = np.concatenate((features, data[b'data'][:, 0:2150]), axis=0)
        elif mode == TEST:
            for i in range(1, 4):
                path = os.path.join(data_dir, 'bookmarks-test-%d.torch'%i)
                data = torchfile.load(path)
                labels = np.concatenate((labels, data[b'labels']), axis=0)
                features = np.concatenate((features, data[b'data'][:, 0:2150]), axis=0)

        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

        if pairs is None:
            self.onehot_labels = labels2onehot(self.labels, False)
        else:
            self.onehot_labels = torch.stack([labels2beliefs(self.labels[i,:], pairs, False).squeeze() for i in range(self.labels.size(0))])

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        feat = self.features[idx, :]
        if self.flip_prob is not None:
            mask = self.bernoulli.sample(torch.Size([2150]))
            feat = feat*(1-mask) + (1-feat)*mask
        return feat, self.labels[idx, :], self.onehot_labels[idx, :]


class BookmarksUnaryModel(nn.Module):
    def __init__(self, params):
        super(BookmarksUnaryModel, self).__init__()
        dropout = params.get('unary_dropout', 0.5)
        self.model = nn.Sequential(
                nn.Linear(2150, 150),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(150, 150),
                nn.ReLU(inplace=True),
                nn.Linear(150, 208)
            )
        self.finetune = params.get('unary_finetune', False)
        if self.finetune:
            for param in self.model[0].parameters():
                param.requires_grad=False
            for param in self.model[2].parameters():
                param.requires_grad=False

    
    def forward(self, inputs):
        result = torch.cuda.FloatTensor(inputs.size(0)*208, 2)
        result[:, 0].fill_(0.)
        result[:, 1].copy_(self.model(inputs).view(-1))
        return result.view(inputs.size(0), -1)




class BookmarksPairModel(nn.Module):
    def __init__(self, params):
        super(BookmarksPairModel, self).__init__()
        self.num_pairs = num_pairs = params.get('num_pairs')
        self.gpu = params.get('gpu', False)
        dropout = params.get('pair_dropout', 0.0)
        num_layers = params.get('pair_num_layers', 2)
        hidden_size = params.get('pair_hidden_size', 1836)
        self.only_ones = params.get('pair_only_ones', False)
        output_size = 4*num_pairs
        layers = [nn.Linear(2150, hidden_size), nn.ReLU(inplace=True)]
        for i in range(num_layers - 2):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, inputs):
        return self.model(inputs)
        

class BookmarksTModel(nn.Module):
    def __init__(self, inp_size, params):
        super(BookmarksTModel, self).__init__()
        self.dropout = params.get('t_dropout', 0.)
        self.model = nn.Sequential(
                nn.Linear(208, 15),
                nn.Softplus(),
                nn.Linear(15, 1),
            )
        self.bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1-self.dropout)

    def train(self, mode=True):
        super(BookmarksTModel, self).train(mode)
        self.dropout_mask = None

    def eval(self):
        super(BookmarksTModel, self).eval()
        self.dropout_mask = None

    def forward(self, beliefs):
        beliefs = beliefs.contiguous().view(-1, 2)[:, 1].contiguous().view(beliefs.size(0), 208)
        if self.training and self.dropout > 0:
            if self.dropout_mask is None:
                self.dropout_mask = self.bernoulli.sample(torch.Size((beliefs.size(0), beliefs.size(1))))
                if next(self.parameters()).is_cuda:
                    self.dropout_mask = self.dropout_mask.cuda(async=True)
            beliefs = beliefs*self.dropout_mask/(1-self.dropout)
        return self.model(beliefs).squeeze()


def test(model, dataset, params, threshold=None):
    batch_size = params.get('batch_size', 1000)
    gpu = params.get('gpu', False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    hamming = 0
    num_data = 0.0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for inputs,labels, onehot_labels in dataloader:
        if gpu:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
        if threshold is None:
            predictions = model.predict(inputs)
        else:
            node_beliefs = model.calculate_beliefs(inputs)[:, :2*208].contiguous().view(-1, 2)[:, 1]
            predictions = (node_beliefs > threshold).long().view(-1, 208)
        correct = (predictions * labels).sum(1)
        precision = (correct.float() / (predictions.sum(1).float()+0.000001))
        recall = (correct.float()/ (labels.sum(1).float()+0.000001))
        total_f1 = total_f1 + ((2*precision*recall)/(precision + recall + 1e-6)).sum()
        total_precision = total_precision + precision.sum()
        total_recall = total_recall + recall.sum()
    f1 = total_f1/len(dataset)
    precision = total_precision/len(dataset)
    recall = total_recall/len(dataset)
    return precision, recall, f1

def test_with_thresholds(model, dataset, params):
    batch_size = params.get('batch_size', 1000)
    gpu = params.get('gpu', False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    thresholds = np.arange(0.05, 0.80, 0.05)
    best_thresh = -1
    best_prec = best_recall = best_f1 = 0.
    total_precisions = np.zeros(len(thresholds))
    total_recalls = np.zeros(len(thresholds))
    total_f1s = np.zeros(len(thresholds))
    hamming = 0
    num_data = 0.0
    for inputs,labels, onehot_labels in dataloader:
        if gpu:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
        node_beliefs = model.calculate_beliefs(inputs)[:, :2*208].contiguous().view(-1, 2)[:, 1]
        for ind,threshold in enumerate(thresholds):
            predictions = (node_beliefs > threshold).long().view(-1, 208)
            correct = (predictions * labels).sum(1)
            total_precisions[ind] += (correct.float() / (predictions.sum(1).float()+0.000001)).sum().item()
            total_recalls[ind] += (correct.float()/ (labels.sum(1).float()+0.000001)).sum().item()
            correct = (predictions * labels).sum(1)
            precision = (correct.float() / (predictions.sum(1).float()+0.000001))
            recall = (correct.float()/ (labels.sum(1).float()+0.000001))
            total_f1s[ind] += ((2*precision*recall)/(precision + recall + 1e-6)).sum()
    precisions = total_precisions/len(dataset)
    recalls = total_recalls/len(dataset)
    f1s = total_f1s/len(dataset)
    best_ind = f1s.argmax()
    return (precisions[best_ind], recalls[best_ind], f1s[best_ind]), thresholds[best_ind]


def train(model, train_data, val_data, params, loggers):
    train_logger = loggers['train']
    val_logger = loggers['val']
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    val_interval = params.get('val_interval', 5)
    working_dir = params['working_dir']
    verbose = params.get('verbose', False)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=gpu, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=gpu)
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
    best_acc = (0,0,0)
    best_acc_epoch = -1
    best_thresh = -1
    for epoch in range(num_epochs):
        print("EPOCH", epoch+1, (end-start))
        train_logger.update_epoch()
        val_logger.update_epoch()
        if epoch%val_interval == 0:
            if tune_thresholds:
                new_results, new_thresh = test_with_thresholds(model, train_data, params)
                train_results.append(new_results)
                
                print("TRAIN RESULTS: ",train_results[-1])
                print("BEST TRAIN THRESH: ", new_thresh)
                new_results, new_thresh = test_with_thresholds(model, val_data, params)
                val_results.append(new_results)
                print("VAL RESULTS: ",val_results[-1])
                print("BEST VAL THRESH: ", new_thresh)
                
            else:
                train_results.append(test(model, train_data, params))
                print("TRAIN RESULTS: ",train_results[-1])
                val_results.append(test(model, val_data, params))
                print("VAL RESULTS: ",val_results[-1])
            if val_results[-1][2] > best_acc[2]:
                best_acc = val_results[-1]
                best_acc_epoch = epoch
                if tune_thresholds:
                    best_thresh = new_thresh
                print("NEW BEST ACCURACY FOUND, SAVING MODEL")
                save_model(model, working_dir, 'model', params)
            save_model(model, working_dir, 'model_checkpoint', params)
            train_logger.plot_for_current_epoch('F1', train_results[-1][2])
            val_logger.plot_for_current_epoch('F1', val_results[-1][2])
            if tune_thresholds:
                print("BEST VAL RESULTS (EPOCH %d, THRESH %f): "%(best_acc_epoch, best_thresh), best_acc)
            else:
                print("BEST VAL RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
        if training_scheduler is not None:
            training_scheduler.step()
        start = time.time() 
        avg_obj = 0
        obj_count = 0
        for batch_ind, (inputs, labels, onehot_labels) in enumerate(train_data_loader):
            if use_cross_ent:
                if gpu:
                    inputs = inputs.cuda(async=True)
                    labels = labels.cuda(async=True)
                obj, inf_obj = model.calculate_obj(epoch, inputs, labels)
            else:
                if gpu:
                    inputs = inputs.cuda(async=True)
                    onehot_labels = onehot_labels.cuda(async=True)
                obj, inf_obj = model.calculate_obj(epoch, inputs, onehot_labels)
            if verbose:
                print("\tBATCH %d OF %d: %f, %f"%(batch_ind+1, len(train_data_loader), obj.item(), inf_obj.item()))
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
        new_results, new_thresh = test_with_thresholds(model, val_data, params)
        val_results.append(new_results)
        print("VAL RESULTS: ",val_results[-1])
        print("BEST VAL THRESH: ", new_thresh)
    else:
        train_results.append(test(model, train_data, params))
        print("FINAL TRAIN RESULTS: ",train_results[-1])
        val_results.append(test(model, val_data, params))
        print("FINAL VAL RESULTS: ",val_results[-1])
    if val_results[-1][2] > best_acc[2]:
        best_acc = val_results[-1]
        best_acc_epoch = epoch
        if tune_thresholds:
            best_thresh = new_thresh
        print("NEW BEST ACCURACY FOUND, SAVING MODEL")
        save_model(model, working_dir, 'model', params)
    save_model(model, working_dir, 'model_checkpoint', params)
    train_logger.plot_for_current_epoch('F1', train_results[-1][2])
    val_logger.plot_for_current_epoch('F1', val_results[-1][2])
    if tune_thresholds:
        print("BEST VAL RESULTS (EPOCH %d, THRESH %f): "%(best_acc_epoch, best_thresh), best_acc)
    else:
        print("BEST VAL RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
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
    parser = argparse.ArgumentParser(description = 'Training non-structured words models')
    parser.add_argument('model', choices=['unary', 'struct', 'spen', 'gspen'])
    parser.add_argument('data_path')
    parser.add_argument('working_dir')
    parser.add_argument('--t_version', choices=['t_v1', 'full_t_v1', 't_v2', 't_v3'])
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--unary_lr', type=float)
    parser.add_argument('--unary_mom', type=float, default=0.)
    parser.add_argument('--unary_num_layers', type=int, default=2)
    parser.add_argument('--pair_lr', type=float)
    parser.add_argument('--pair_mom', type=float, default=0.)
    parser.add_argument('--t_unary_lr', type=float)
    parser.add_argument('--t_lr', type=float)
    parser.add_argument('--t_unary_mom', type=float, default=0.)
    parser.add_argument('--t_pair_lr', type=float)
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--t_hidden_size', type=int, default=130)
    parser.add_argument('--unary_hidden_size', type=int, default=1836)
    parser.add_argument('--pair_hidden_size', type=int, default=1836)
    parser.add_argument('--t_num_layers', type=int, default=2)
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
    parser.add_argument('--unary_wd', type=float, default=0.)
    parser.add_argument('--pair_wd', type=float, default=0.)
    parser.add_argument('--t_unary_wd', type=float, default=0.)
    parser.add_argument('--t_pair_wd', type=float, default=0.)
    parser.add_argument('--mp_eps', type=float, default=0.)
    parser.add_argument('--mp_itrs', type=int)
    parser.add_argument('--inf_mode', choices=['lp', 'mp', 'md'])
    parser.add_argument('--use_entropy', action='store_true')
    parser.add_argument('--t_dropout', type=float, default=0)
    parser.add_argument('--unary_dropout', type=float)
    parser.add_argument('--pair_dropout', type=float, default=0.)
    parser.add_argument('--pair_num_layers', type=int, default=2)
    parser.add_argument('--inf_eps', type=float)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_t_pots', action='store_true')
    parser.add_argument('--unary_finetune', action='store_true')
    parser.add_argument('--t_use_softplus', action='store_true')
    parser.add_argument('--clip_grad', type=float)
    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--use_cross_ent', action='store_true')
    parser.add_argument('--tune_thresholds', action='store_true')
    parser.add_argument('--fix_features', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_pairs', type=int)
    parser.add_argument('--flip_prob', type=float)
    parser.add_argument('--fix_unaries', action='store_true')
    parser.add_argument('--entropy_coef', type=float, default=1)

    
    args = parser.parse_args()
    params = vars(args)
  
    unary_model = BookmarksUnaryModel(params)
    num_nodes = 208
    num_vals = 2

    if args.model in ['struct', 'gspen']:
        max_label = 192 #Found in another script
        pairs = []
        for node in range(208):
            if node == max_label:
                continue
            elif node < max_label:
                pairs.append((node, max_label))
            else:
                pairs.append((max_label, node))
        params['num_pairs'] = len(pairs)
        
    else:
        pairs = None
    train_data = BookmarksDataset(TRAIN, args.data_path, pairs, flip_prob=args.flip_prob)
    val_data = BookmarksDataset(VAL, args.data_path, pairs)
    if args.test:
        test_data = BookmarksDataset(TEST, args.data_path, pairs)
    
    if args.model in ['struct', 'gspen']:
        pair_model = BookmarksPairModel(params)

    if args.model in ['spen', 'gspen']:
        unary_t_constructor = BookmarksTModel
        params['no_t_pots'] = True
        pair_t_constructor = None


        if args.t_version == 't_v1':
            t_model = SplitTModelV1(unary_t_constructor, pair_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 'full_t_v1':
            t_model = TModelV1(unary_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v2':
            if args.model == 'unary_t':
                t_model = TModelV2(unary_t_constructor, num_nodes, pairs, num_vals, params)
            else:
                t_model = SplitTModelV2(unary_t_constructor, pair_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v3':
            t_model = TModelV3(t_constructor, num_nodes, None, num_vals, params)
        if args.model == 'spen':
            model = SPENModel(unary_model, t_model, num_nodes, num_vals, params)
        elif args.model in ['gspen']:
            model = GSPENModel(unary_model, pair_model, t_model, num_nodes, pairs, num_vals, params)
    elif args.model == 'unary':
        model = UnaryModel(unary_model, num_nodes, num_vals, params)
    elif args.model == 'struct':
        model = StructModel(unary_model, pair_model, num_nodes, pairs, num_vals, params)
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
    print('MODEL: ', model)
    if args.test:
        print("FINDING BEST THRESHOLD ON VALIDATION DATA...")
        (_,_,val_f1), thresh = test_with_thresholds(model, val_data, params)
        print("TESTING ON TRAINING DATA...")
        _,_,train_f1 = test(model, train_data, params, threshold=thresh)
        print("TESTING ON TEST DATA...")
        _,_,test_f1 = test(model, test_data, params, threshold=thresh)

        print("FINAL RESULTS: (THRESHOLD %f)"%thresh)
        print("\tTRAIN: %f"%(train_f1))
        print("\tVAL:   %f"%(val_f1))
        print("\tTEST:  %f"%(test_f1))

    else:
        train_log_dir = os.path.join(args.working_dir, 'train')
        val_log_dir = os.path.join(args.working_dir, 'val')
        pred_log_dir = os.path.join(args.working_dir, 'pred')
        data_log_dir = os.path.join(args.working_dir, 'data')
        train_logger = TensorBoard(train_log_dir)
        val_logger = TensorBoard(val_log_dir)
        
        loggers = {
            'train':train_logger,
            'val':val_logger,
        }
        train_obj_vals, train_inf_obj_vals, train_scores, val_scores = train(model, train_data, val_data, params, loggers)   
        train_logger.close()
        val_logger.close()
        
