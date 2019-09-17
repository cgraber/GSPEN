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
import argparse, os, sys
import time
from gspen.models import *

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
    onehot = tensor_mod.FloatTensor(len(new_labels), 26).fill_(0.)
    onehot.scatter_(1, new_labels, 1)
    return onehot.view(labels.size(0), -1)

def labels2beliefs(labels, pairs, gpu):
    if gpu:
        tensor_mod = torch.cuda
    else:
        tensor_mod = torch
    new_labels = labels.view(-1,1)
    unary = tensor_mod.FloatTensor(len(new_labels), 26).fill_(0.)
    unary.scatter_(1, new_labels, 1)
    pair_bel = tensor_mod.FloatTensor(len(pairs)*26*26).fill_(0.)
    for pair_ind,pair in enumerate(pairs):
        label_1 = labels[pair[0]]
        label_2 = labels[pair[1]]
        ind = pair_ind*26*26 + label_1 + label_2*26
        pair_bel[ind] = 1

    return torch.cat([unary.view(-1), pair_bel])


class WordsDataset(Dataset):
    def __init__(self, data_dir, mode, pairs, data_size, save=False, load=False):
        super(WordsDataset, self).__init__()
        if mode == TRAIN:
            path = os.path.join(data_dir, 'train/')
            if data_size == 'huge':
                data_len = 1000000
            elif data_size == 'lesshuge':
                data_len = 100000
            elif data_size == 'full':
                data_len = 10000
            elif data_size == 'verysmall':
                data_len = 200
            elif isinstance(data_size, int):
                data_len = data_size
                print("USING DATA SIZE: ",data_len)
            else:
                data_len = 1000
        elif mode == VAL:
            path = os.path.join(data_dir, 'val/')
            if data_size == 'huge':
                data_len = 200000
            elif data_size == 'lesshuge':
                data_len = 20000
            elif data_size == 'full':
                data_len = 2000
            elif data_size == 'verysmall':
                data_len = 200
            else:
                data_len = 200
        elif mode == TEST:
            if data_size == 'huge':
                data_len = 200000
            elif data_size == 'lesshuge':
                data_len = 20000
            elif data_size == 'full':
                data_len = 2000
            elif data_size == 'verysmall':
                data_len = 200
            else:
                data_len = 200
            path = os.path.join(data_dir, 'test/')
        self.data_len = data_len
        if load:
            load_path = os.path.join(path, 'preprocessed_data')
            if pairs is None:
                onehot_path = os.path.join(path, 'preprocessed_data_unary')
            else:
                onehot_path = os.path.join(path, 'preprocessed_data_pair')
            self.observations, self.labels = torch.load(load_path)
            self.observations = self.observations[:data_len]
            self.labels = self.labels[:data_len]
            self.onehot_labels = torch.load(onehot_path)
            self.onehot_labels = self.onehot_labels[:data_len]
            return
        self.observations = []
        self.labels = []
        self.onehot_labels = []
        for i in range(data_len):
            if (i+1)%1000 == 0:
                print("\t%d out of %d"%(i, data_len))
            tmp_path = os.path.join(path, str(i))
            label_path = os.path.join(tmp_path, 'label.txt')
            with open(label_path, 'r') as fin:
                label = torch.from_numpy(np.array([int(label.strip()) for label in fin.readlines()]))
                self.labels.append(label)
                if pairs is None:
                    self.onehot_labels.append(labels2onehot(label.unsqueeze(0), False).squeeze())
                else:
                    self.onehot_labels.append(labels2beliefs(label, pairs, False).squeeze())
            datum = []
            for j in range(5):
                img_path = os.path.join(tmp_path, '%d.png'%j)
                img = torch.from_numpy(skimage.io.imread(img_path, as_grey=True).flatten()).float()
                img.div_(255)
                datum.append(img)
            self.observations.append(torch.stack(datum))
        if save:
            save_path = os.path.join(path, 'preprocessed_data')
            torch.save([self.observations, self.labels], save_path)
            if pairs is None:
                onehot_path = os.path.join(path, 'preprocessed_data_unary')
            else:
                onehot_path = os.path.join(path, 'preprocessed_data_pair')
            torch.save(self.onehot_labels, onehot_path)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx], self.onehot_labels[idx]

    def __len__(self):
        return self.data_len

def find_pair_statistics(train_data, buffer, pair_init_div):
    unary_offset = 5*26
    p_vals=26*26
    result = torch.zeros(4*p_vals)
    for _,_,belief_labels in train_data:
        for node in range(4):
            result[p_vals*node:p_vals*(node+1)] += belief_labels[unary_offset+node*p_vals:unary_offset+(node+1)*p_vals]
    result = result/len(train_data)
    return torch.log(result + buffer)/pair_init_div

class WordsUnaryModel(nn.Module):
    def __init__(self, params):
        super(WordsUnaryModel, self).__init__()
        num_layers = params.get('unary_num_layers', 2)
        hidden_size = params.get('unary_hidden_size', 28*28)
        dropout = params.get('unary_dropout')
        layers = []
        layers.append(nn.Linear(28*28, hidden_size))
        layers.append(nn.ReLU())
        for layer in range(num_layers-2):
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 26))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(inputs.size(0), -1)


class PairModel(nn.Module):
    def __init__(self, params, pair_statistics):
        super(PairModel, self).__init__()
        self.use_separate_pairs = params.get('use_separate_pairs', False)
        self.use_better_pairs = params.get('use_better_pairs', False)
        self.finetune = params.get('finetune', False)
        if self.use_better_pairs:
            num_layers = params.get('pair_num_layers', 2)
            hidden_size = params.get('pair_hidden_size', 2*28*28)
            dropout = params.get('pair_dropout')
            layers = []
            layers.append(nn.Linear(2*28*28, hidden_size))
            layers.append(nn.ReLU())
            for layer in range(num_layers-2):
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, 26*26))
            self.model = nn.Sequential(*layers)
        elif self.use_separate_pairs:
            if params.get('pair_ones_init', False):
                self.model = torch.nn.Parameter(torch.FloatTensor(26*26*4).fill_(1.))
            elif params.get('pair_zeros_init', False):
                self.model = torch.nn.Parameter(torch.FloatTensor(26*26*4).fill_(0.))
            else:
                self.model = torch.nn.Parameter(torch.FloatTensor(26*26*4).uniform_(-0.5, 0.5))
        else:
            if params.get('pair_ones_init', False):
                self.model = torch.nn.Parameter(torch.FloatTensor(26*26).fill_(1.))
            elif params.get('pair_zeros_init', False):
                self.model = torch.nn.Parameter(torch.FloatTensor(26*26).fill_(0.))
            else:
                self.model = torch.nn.Parameter(torch.FloatTensor(26*26).uniform_(-0.5, 0.5))

    def forward(self, inputs):
        if self.use_better_pairs:
            pairs = []
            for i in range(4):
                inp = inputs[:, i:i+2, :].view(inputs.size(0), -1)
                pairs.append(self.model(inp))
            return torch.cat(pairs, dim=1)
        elif self.use_separate_pairs:
            return self.model.repeat(inputs.size(0), 1)
        else:
            return self.model.repeat(inputs.size(0), 4)

       
def test(model, dataset, params):
    batch_size = params.get('batch_size', 1000)
    gpu = params.get('gpu', False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    char_acc, word_acc = 0,0
    num_data = 0.0
    record_inf = params.get('record_inf', False)
    if record_inf:
        all_inf_objs = []
    for inputs,labels, onehot_labels in dataloader:
        if gpu:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
            onehot_labels = onehot_labels.cuda(async=True)
        if record_inf:
            predictions, inf_objs = model.predict(inputs)
            all_inf_objs.append(inf_objs)
        else:
            predictions = model.predict(inputs)
        result = (predictions == labels).float()
        char_acc += result.sum()
        word_acc += (result.sum(1) == 5).sum()
        num_data += len(inputs)
    if record_inf:
        return (char_acc.float()/(num_data*5)).item(), (word_acc.float()/num_data).item(), all_inf_objs
    else:
        return (char_acc.float()/(num_data*5)).item(), (word_acc.float()/num_data).item()


def train(model, train_data, val_data, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    val_interval = params.get('val_interval', 5)
    val_start = params.get('val_start', 0)
    clip_grad = params.get('clip_grad', None)
    clip_grad_norm = params.get('clip_grad_norm', None)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=gpu, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=gpu)
    opt = model.get_optimizer(params)
    working_dir = params['working_dir']
    use_cross_ent = params.get('use_cross_ent', False)
    best_model_path = os.path.join(working_dir, 'model')
    checkpoint_path = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    if training_scheduler is not None:
        training_scheduler = training_scheduler(model_optimizer)
    end = start = 0 
    train_results = []
    val_results = []
    train_obj_vals = []
    train_inf_obj_vals = []
    best_acc = (0,0)
    best_word_acc = (0,0)
    best_word_acc_epoch = 0
    best_acc_epoch = -1
    for epoch in range(num_epochs):
        print("EPOCH", epoch+1, (end-start))
        if epoch%val_interval == 0:
            if epoch >= val_start:
                train_results.append(test(model, train_data, params))
                print("TRAIN RESULTS: ",train_results[-1])
                val_results.append(test(model, val_data, params))
                print("VAL RESULTS: ",val_results[-1])
                if val_results[-1][0] > best_acc[0]:
                    best_acc = val_results[-1]
                    best_acc_epoch = epoch
                    print("NEW BEST ACCURACY FOUND, SAVING MODEL")
                    save_model(model, working_dir, 'model', params)
                print("BEST VAL RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
                if val_results[-1][1] > best_word_acc[1] or (val_results[-1][1] == best_word_acc[1] and val_results[-1][0] > best_word_acc[0]):
                    best_word_acc = val_results[-1]
                    best_word_acc_epoch = epoch
                    print("NEW BEST WORD ACC FOUND, SAVING MODEL")
                    save_model(model, working_dir, 'wordacc_model', params)
                print("BEST VAL WORD ACC (EPOCH %d): "%best_word_acc_epoch, best_word_acc)
            save_model(model, working_dir, 'model_checkpoint', params)
            torch.save({
                        'epoch':epoch,
                        'optimizer':opt.state_dict(),
                        'best_acc':best_acc,
                        'best_word_acc':best_word_acc,
                        'best_acc_epoch':best_acc_epoch,
                        'best_word_acc_epoch':best_word_acc_epoch,
                    }, training_path)


        if training_scheduler is not None:
            training_scheduler.step()
        start = time.time() 
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
                obj, inf_obj = model.calculate_obj(epoch, inputs, onehot_labels)
            print("\tBATCH %d OF %d: %f, %f"%(batch_ind+1, len(train_data_loader), obj.item(), inf_obj.item()))
            obj.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            elif clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            opt.step()
            train_obj_vals.append(obj.item())
            train_inf_obj_vals.append(inf_obj.item())
        end = time.time()
    train_results.append(test(model, train_data, params))
    print("FINAL TRAIN RESULTS: ",train_results[-1])
    val_results.append(test(model, val_data, params))
    print("FINAL VAL RESULTS: ",val_results[-1])
    if val_results[-1][0] > best_acc[0]:
        best_acc = val_results[-1]
        best_acc_epoch = epoch
        print("NEW BEST ACCURACY FOUND, SAVING MODEL")
        save_model(model, working_dir, 'model', params)
    if val_results[-1][1] > best_word_acc[1] or (val_results[-1][1] == best_word_acc[1] and val_results[-1][0] > best_word_acc[0]):
        best_word_acc = val_results[-1]
        best_word_acc_epoch = epoch
        print("NEW BEST WORD ACC FOUND, SAVING MODEL")
        save_model(model, working_dir, 'wordacc_model', params)
    print("BEST VAL WORD ACC (EPOCH %d): "%best_word_acc_epoch, best_word_acc)
    save_model(model, working_dir, 'model_checkpoint', params)
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
    parser = argparse.ArgumentParser(description = 'Training words models')
    parser.add_argument('model', choices=['unary', 'struct', 'spen', 'gspen'])
    parser.add_argument('data_size', choices=['huge', 'lesshuge', 'full', 'small', 'verysmall'])
    parser.add_argument('data_directory')
    parser.add_argument('working_dir')
    parser.add_argument('--t_version', choices=['t_v1', 'full_t_v1', 'full_t_v2', 't_v2', 't_v3'])
    parser.add_argument('--t', choices=['linear', 'quad', 'skinny_quad', 'mlp'])
    parser.add_argument('--t_unary', choices=['linear', 'quad', 'skinny_quad', 'mlp'])
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--unary_lr', type=float)
    parser.add_argument('--unary_mom', type=float, default=0.)
    parser.add_argument('--pair_mom', type=float, default=0.)
    parser.add_argument('--pair_lr', type=float)
    parser.add_argument('--t_unary_lr', type=float)
    parser.add_argument('--t_pair_lr', type=float)
    parser.add_argument('--t_lr', type=float)
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--val_start', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--t_hidden_size', type=int, default=130)
    parser.add_argument('--unary_hidden_size', type=int, default=28*28)
    parser.add_argument('--t_num_layers', type=int, default=2)
    parser.add_argument('--unary_num_layers', type=int, default=2)
    parser.add_argument('--pair_num_layers', type=int, default=2)
    parser.add_argument('--num_inf_itrs', type=int, default=100)
    parser.add_argument('--inf_lr', type=float, default=0.05)
    parser.add_argument('--use_sqrt_decay', action='store_true')
    parser.add_argument('--use_linear_decay', action='store_true')
    parser.add_argument('--use_relu', action='store_true')
    parser.add_argument('--inf_method', choices=['fw', 'emd'], default='fw')
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
    parser.add_argument('--pair_ones_init', action='store_true')
    parser.add_argument('--pair_zeros_init', action='store_true')
    parser.add_argument('--inf_mode', choices=['lp', 'mp', 'md'])
    parser.add_argument('--use_entropy', action='store_true')
    parser.add_argument('--use_better_pairs', action='store_true')
    parser.add_argument('--t_first_dropout', type=float)
    parser.add_argument('--t_last_dropout', type=float)
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--inf_eps', type=float)
    parser.add_argument('--use_separate_pairs', action='store_true')
    parser.add_argument('--pair_hidden_size', type=int, default=2*28*28)
    parser.add_argument('--pair_dropout', type=float)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--clip_grad', type=float)
    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--no_t_pots', action='store_true')
    parser.add_argument('--use_cross_ent', action='store_true')
    parser.add_argument('--t_use_softplus', action='store_true')
    parser.add_argument('--t_use_relu', action='store_true')
    parser.add_argument('--t_normalize_pots', action='store_true')
    parser.add_argument('--unary_dropout', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    params = vars(args)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    unary_model = WordsUnaryModel(params)
    num_nodes = 5
    num_vals = 26
    params['num_vals'] = num_vals

    if args.model in ['struct', 'gspen']:
        pairs = [(0,1), (1,2), (2,3), (3,4)]
    else:
        pairs = None
    if args.save_data:
        print("PROCESSING TRAINING DATA...")
        train_data = WordsDataset(args.data_directory, TRAIN, pairs, args.data_size, save=True)
        print("PROCESSING VALIDATION DATA...")
        val_data = WordsDataset(args.data_directory, VAL, pairs, args.data_size, save=True)
        print("DONE.")
        sys.exit(0)
    else:
        print("LOADING TRAINING DATA...")
        train_data = WordsDataset(args.data_directory, TRAIN, pairs, args.data_size, load=args.load_data)
        print("LOADING VALIDATION DATA...")
        val_data = WordsDataset(args.data_directory, VAL, pairs, args.data_size, load=args.load_data)
        if args.test:
            print("LOADING TEST DATA...")
            test_data = WordsDataset(args.data_directory, TEST, pairs, 'full', load=args.load_data)
    if args.model in ['struct', 'gspen']:
        pair_model = PairModel(params, None)

    if args.model in ['spen', 'gspen']:
        if args.t_unary == 'linear':
            unary_t_constructor = LinearTModel
        elif args.t_unary == 'quad':
            unary_t_constructor = QuadTModel
        elif args.t_unary == 'skinny_quad':
            unary_t_constructor = SkinnyQuadTModel
        elif args.t_unary == 'mlp':
            unary_t_constructor = MLPTModel
        else:
            unary_t_constructor = None


        if args.t_version == 't_v1':
            t_model = SplitTModelV1(unary_t_constructor, None, num_nodes, pairs, num_vals, params)
        elif args.t_version == 'full_t_v1':
            t_model = TModelV1(unary_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v2':
            if args.model == 'unary_t':
                t_model = TModelV2(unary_t_constructor, num_nodes, pairs, num_vals, params)
            else:
                t_model = SplitTModelV2(unary_t_constructor, None, num_nodes, pairs, num_vals, params)
        elif args.t_version == 'full_t_v2':
            t_model = TModelV2(unary_t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v3':
            t_model = TModelV3(unary_t_constructor, num_nodes, pairs, num_vals, params)
        if args.model == 'spen':
            model = SPENModel(unary_model, t_model, num_nodes, num_vals, params)
        elif args.model == 'gspen':
            model = GSPENModel(unary_model, pair_model, t_model, num_nodes, pairs, num_vals, params)
    elif args.model == 'unary':
        model = UnaryModel(unary_model, num_nodes, num_vals, params)
    elif args.model == 'struct':
        model = StructModel(unary_model, pair_model, num_nodes, pairs, num_vals, params)
    if args.pretrain_unary is not None:
        model.load_unary(args.pretrain_unary)
    if args.pretrain_pair is not None:
        model.load_pair(args.pretrain_pair)
    if args.pretrain_t is not None:
        model.load_t(args.pretrain_t)
    if args.pretrain_unary_t is not None:
        model.load_unary_t(args.pretrain_unary_t)
    print(model)
    
    if args.test:
        print("TESTING ON TRAINING DATA...")
        train_char_acc, train_word_acc = test(model, train_data, params)
        print("TESTING ON VALIDATION DATA...")
        val_char_acc, val_word_acc = test(model, val_data, params)
        print("TESTING ON TEST DATA...")
        test_char_acc, test_word_acc = test(model, test_data, params)

        print("FINAL RESULTS:")
        print("\tTRAIN: (%f, %f)"%(train_char_acc, train_word_acc))
        print("\tVAL:   (%f, %f)"%(val_char_acc, val_word_acc))
        print("\tTEST:  (%f, %f)"%(test_char_acc, test_word_acc))
    else:
        train_obj_vals, train_inf_obj_vals, train_scores, val_scores = train(model, train_data, val_data, params)