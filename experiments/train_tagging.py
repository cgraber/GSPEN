import torch
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
from torchvision import transforms
import torchvision.models
from PIL import Image
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

NUM_TRAIN = 10000
NUM_TEST = 10000
NUM_VAL = 5000

order_full = {'structures.txt':0,
        'animals.txt':1,
        'transport.txt':2,
'food.txt':3,
'portrait.txt':4,
'sky.txt':5,
'female.txt':6,
'male.txt':7,
'flower.txt':8,
'people.txt':9,
'river.txt':10,
'sunset.txt':11,
'baby.txt':12,
'plant_life.txt':13,
'indoor.txt':14,
'car.txt':15,
'bird.txt':16,
'dog.txt':17,
'tree.txt':18,
'sea.txt':19,
'night.txt':20,
'lake.txt':21,
'water.txt':22,
'clouds.txt':23}
reverse_order = [None for _ in range(24)]
for key in order_full:
    reverse_order[order_full[key]] = key


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


class FlickrTaggingDataset(Dataset):
    def __init__(self, images_folder, annotations_folder, mode, save_file, pairs, load=False):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        if load:
            print("LOADING PRECOMPUTED IMAGES")
            self.images, self.labels, self.onehot_labels = torch.load(save_file)
        else:
            print("LOADING IMAGES")
            self.annotations = [None]*24
            for annotation_file in os.listdir(annotations_folder):
                if '_r1' in annotation_file or 'README' in annotation_file:
                    continue
                vals = set()
                fin = open(os.path.join(annotations_folder, annotation_file), 'r')
                for line in fin:
                    vals.add(int(line.strip())-1)
                self.annotations[order_full[annotation_file]] = vals
            self.img_folder = images_folder
            self.img_files = [img_file for img_file in os.listdir(images_folder) if (os.path.isfile(os.path.join(images_folder, img_file)) and 'jpg' in img_file)]
            self.img_files.sort(key=lambda name: int(name[2:name.find('.jpg')]))

            if mode == TRAIN:
                self.img_files = self.img_files[:NUM_TRAIN]
            elif mode == TEST:
                self.img_files = self.img_files[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
            else:
                self.img_files = self.img_files[NUM_TRAIN+NUM_TEST:]
            self.images = [None]*len(self.img_files)
            self.labels = []
            self.onehot_labels = []
            for img_file in self.img_files:
                path = os.path.join(self.img_folder, img_file)
                with open(path, 'rb') as f:
                    with Image.open(f) as raw_img:
                        img = self.transform(raw_img.convert('RGB'))
                img_no = int(img_file[2:img_file.find('.jpg')]) - 1
                if mode == TRAIN:
                    img_ind = img_no
                elif mode == TEST:
                    img_ind = img_no - NUM_TRAIN
                else:
                    img_ind = img_no - NUM_TRAIN - NUM_TEST
                label = [0]*len(self.annotations)
                for i,annotation in enumerate(self.annotations):
                    if img_no in annotation:
                        label[i] = 1
                self.images[img_ind] = img
                label = torch.LongTensor(label)
                self.labels.append(label)
                if pairs is None:
                    self.onehot_labels.append(labels2onehot(label.unsqueeze(0), False).squeeze())
                else:
                    self.onehot_labels.append(labels2beliefs(label, pairs, False).squeeze())
            if save_file is not None:
                torch.save([self.images, self.labels, self.onehot_labels], save_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.onehot_labels[idx]


class UnaryModel(nn.Module):
    def __init__(self, params):
        super(UnaryModel, self).__init__()
        self.finetune = params.get('finetune', False)
        self.model = torchvision.models.alexnet(pretrained=True)
        self.simple = params.get('unary_simple')
            
        if self.simple:
            self.model.classifier = nn.Sequential(
                    nn.Linear(256*6*6, 2*24),
                    nn.ReLU(inplace=True),
                    nn.Linear(2*24, 2*24),
            )
        else:
            tmp = list(self.model.classifier)
            tmp[-1] = nn.Linear(4096, 2*24)
            self.model.classifier = nn.Sequential(*tmp)
        if self.finetune:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def load_from_old(self, path):
        new_linear = torch.load(path)
        tmp = list(self.model.classifier)
        tmp[-1] = new_linear
        self.model.classifier = nn.Sequential(*tmp)

    def forward(self, inputs):
        return self.model(inputs)

    
class UnaryModel_Old(nn.Module):
    def __init__(self, params):
        super(UnaryModel_Old, self).__init__()
        self.finetune = params.get('finetune', False)
        self.model = torchvision.models.alexnet(pretrained=True)
        self.simple = params.get('unary_simple')
            
        if self.simple:
            self.model.classifier = nn.Sequential(
                    nn.Linear(256*6*6, 2*24),
                    nn.ReLU(inplace=True),
                    nn.Linear(2*24, 2*24),
            )
        else:
            tmp = list(self.model.classifier)
            tmp[-1] = nn.Linear(4096, 24)
            self.model.classifier = nn.Sequential(*tmp)

    def load_from_old(self, path):
        self.model = torch.load(path)

    def forward(self, inputs):
        if inputs.is_cuda:
            tmp_result = torch.cuda.FloatTensor(inputs.size(0)*24, 1).fill_(0.)
        result = self.model(inputs).view(-1, 1)*-1
        return torch.cat([result, tmp_result], dim=1).view(inputs.size(0), -1)

    def parameters(self):
        if self.finetune:
            if self.simple:
                return self.model.classifier.parameters()
            else:
                return self.model.classifier[-1].parameters()
        else:
            return super(UnaryModel_Old, self).parameters()


class PairModel(nn.Module):
    def __init__(self, params, pair_statistics=None):
        super(PairModel, self).__init__()
        self.finetune = params.get('pair_finetune', False)
        self.simple = params.get('pair_simple', False)
        self.use_better_pairs = params.get('use_better_pairs', False)
        if self.use_better_pairs:
            self.model = torchvision.models.alexnet(pretrained=True)
            if self.simple:
                hidden_size = params.get('pair_hidden_size', 276*4)
                dropout = params.get('pair_dropout', False)
                if dropout:
                    self.model.classifier = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(256*6*6, hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, 276*4),
                        )
                else:
                    self.model.classifier = nn.Sequential(
                            nn.Linear(256*6*6, hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, 276*4),
                        )
            else:
                tmp = list(self.model.classifier)
                tmp[-1] = nn.Linear(4096, 276*4)
                self.model.classifier = nn.Sequential(*tmp)
        elif pair_statistics is not None:
            self.model = torch.nn.Parameter(pair_statistics)
        elif params.get('pair_ones_init', False):
            self.model = torch.nn.Parameter(torch.FloatTensor(276*4).fill_(1.))
        elif params.get('pair_zeros_init', False):
            self.model = torch.nn.Parameter(torch.FloatTensor(276*4).fill_(0.))
        else:
            self.model = torch.nn.Parameter(torch.FloatTensor(276*4).uniform_(-0.5, 0.5))

    def parameters(self):
        if self.finetune:
            if self.simple:
                return self.model.classifier.parameters()
            else:
                return self.model.classifier[-1].parameters()
        else:
            return super(PairModel, self).parameters()


    def forward(self, inputs):
        if self.use_better_pairs:
            return self.model(inputs)
        else:
            return self.model.repeat(inputs.size(0), 1)


class CombinedPotModel(nn.Module):
    def __init__(self, params, pairs):
        super(CombinedPotModel, self).__init__()
        self.finetune = params.get('finetune', False)
        self.model = torchvision.models.alexnet(pretrained=True)
        self.fix_unaries = params.get('fix_unaries', False)
        self.model.classifier = nn.Sequential(*(list(self.model.classifier)[:-1]))
        self.unary_layer = nn.Linear(4096, 2*24)
        self.pair_layer = nn.Linear(4096, 4*len(pairs))
        if self.fix_unaries:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.unary_layer.parameters():
                param.requires_grad = False

    def load_from_unary(self, path):
        unary_state_dict = torch.load(path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in unary_state_dict[0].items():
            name = k[6:]
            new_state_dict[name] = v
        tmp_model = torchvision.models.alexnet()
        tmp = list(tmp_model.classifier)
        tmp[-1] = nn.Linear(4096, 2*24)
        tmp_model.classifier = nn.Sequential(*tmp)
        tmp_model.load_state_dict(new_state_dict)
        self.model = tmp_model
        self.unary_layer = self.model.classifier[-1]
        self.model.classifier = nn.Sequential(*(list(self.model.classifier)[:-1]))
        if self.fix_unaries:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.unary_layer.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        feats = self.model(inputs)
        unary_pots = self.unary_layer(feats)
        pair_pots = self.pair_layer(feats)
        return torch.cat([unary_pots, pair_pots], dim=1)


class CombinedPotModel_Old(nn.Module):
    def __init__(self, params, pairs):
        super(CombinedPotModel_Old, self).__init__()
        self.unary_model = torchvision.models.alexnet(pretrained=False)
        tmp = list(self.unary_model.classifier)
        tmp[-1] = nn.Linear(4096, 2*24)
        self.unary_model.classifier = nn.Sequential(*tmp)

        self.pair_model = torch.nn.Parameter(torch.FloatTensor(len(pairs)*4).fill_(0.0))

    def forward(self, inputs):
        unary_pots = self.unary_model(inputs)
        pair_pots = self.pair_model.repeat(inputs.size(0), 1)
        return torch.cat([unary_pots, pair_pots], dim=1)

    def load_from_old(self, path):
        self.load_state_dict(torch.load(path))



def test(model, dataset, params):
    batch_size = params.get('batch_size', 1000)
    gpu = params.get('gpu', False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    hamming = 0
    num_data = 0.0
    micro_correct = 0
    micro_predicted = 0
    micro_labels = 0
    total_f1 = 0
    for inputs,labels, onehot_labels in dataloader:
        if gpu:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
        predictions = model.predict(inputs)
        hamming += torch.abs(predictions-labels).sum().item()
        num_data += len(inputs)
        correct = (predictions * labels).sum(1)
        micro_correct += correct.sum().item()
        micro_predicted += predictions.sum().item()
        micro_labels += labels.sum().item()
        precision = (correct.float() / (predictions.sum(1).float()+0.000001))
        recall = (correct.float()/ (labels.sum(1).float()+0.000001))
        total_f1 = total_f1 + ((2*precision*recall)/(precision + recall + 1e-6)).sum()
    macro_f1 = total_f1/len(dataset)
    micro_p = micro_correct/(micro_predicted + 1e-6)
    micro_r = micro_correct/(micro_labels + 1e-6)
    micro_f1 = 2*micro_p*micro_r/(micro_p + micro_r + 1e-6)
    return hamming/num_data, macro_f1, micro_f1

def expanded_test(model, dataset, params):
    batch_size = params.get('batch_size', 1000)
    gpu = params.get('gpu', False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    hamming = 0
    num_data = 0.0
    micro_correct = 0
    micro_predicted = 0
    micro_labels = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    label_correct = torch.cuda.FloatTensor(24).fill_(0.)
    label_counts = torch.cuda.FloatTensor(24).fill_(0.)
    label_pred = torch.cuda.FloatTensor(24).fill_(0.)
    for inputs,labels, onehot_labels in dataloader:
        if gpu:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
        predictions = model.predict(inputs)
        hamming += torch.abs(predictions-labels).sum().item()
        num_data += len(inputs)
        correct = (predictions * labels).sum(1)
        label_correct += (predictions*labels).sum(dim=0).float()
        label_counts += labels.sum(dim=0).float()
        label_pred += predictions.sum(dim=0).float()
        micro_correct += correct.sum().item()
        micro_predicted += predictions.sum().item()
        micro_labels += labels.sum().item()
        precision = (correct.float() / (predictions.sum(1).float()+0.000001))
        total_precision += precision.sum()
        recall = (correct.float()/ (labels.sum(1).float()+0.000001))
        total_recall += recall.sum()
        total_f1 = total_f1 + ((2*precision*recall)/(precision + recall + 1e-6)).sum()
    macro_p = total_precision/len(dataset)
    macro_r = total_recall/len(dataset)
    macro_f1 = total_f1/len(dataset)
    micro_p = micro_correct/micro_predicted
    micro_r = micro_correct/micro_labels
    micro_f1 = 2*micro_p*micro_r/(micro_p + micro_r + 1e-6)
    label_p = label_correct/label_pred
    label_r = label_correct/label_counts
    label_f1 = 2*label_p*label_r/(label_p+label_r+1e-6)
    return hamming/num_data, (macro_p, macro_r, macro_f1), (micro_p, micro_r, micro_f1), (label_p.cpu().numpy(), label_r.cpu().numpy(), label_f1.cpu().numpy())


def train(model, train_data, val_data, params, loggers):
    train_logger = loggers['train']
    val_logger = loggers['val']
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    val_interval = params.get('val_interval', 5)
    working_dir = params['working_dir']
    use_cross_ent = params.get('use_cross_ent', False)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=gpu, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=gpu)
    opt = model.get_optimizer(params)
    if training_scheduler is not None:
        training_scheduler = training_scheduler(model_optimizer)
    end = start = 0 
    train_results = []
    val_results = []
    train_obj_vals = []
    best_acc = [float('inf'), 0, 0]
    best_acc_epoch = -1
    for epoch in range(num_epochs):
        print("EPOCH", epoch+1, (end-start))
        train_logger.update_epoch()
        val_logger.update_epoch()
        if epoch%val_interval == 0:
            train_results.append(test(model, train_data, params))
            print("TRAIN RESULTS: ",train_results[-1])
            val_results.append(test(model, val_data, params))
            print("VAL RESULTS: ",val_results[-1])
            if val_results[-1][0] < best_acc[0]:
                best_acc = val_results[-1]
                best_acc_epoch = epoch
                print("NEW BEST ACCURACY FOUND, SAVING MODEL")
                save_model(model, working_dir, 'model', params)
            print("BEST VAL RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)
            save_model(model, working_dir, 'model_checkpoint', params)
            train_logger.plot_for_current_epoch('Hamming Loss', train_results[-1][0])
            val_logger.plot_for_current_epoch('Hamming Loss', val_results[-1][0])
        if training_scheduler is not None:
            training_scheduler.step()
        start = time.time() 
        avg_obj = 0
        obj_count = 0
        for batch_ind, all_inputs in enumerate(train_data_loader):
            if use_cross_ent:
                inputs, labels, onehot_labels = all_inputs
                if gpu:
                    inputs = inputs.cuda(async=True)
                    labels = labels.cuda(async=True)
                obj, inf_obj = model.calculate_obj(epoch, inputs, labels)
            else:
                inputs, labels, onehot_labels = all_inputs
                if gpu:
                    inputs = inputs.cuda(async=True)
                    onehot_labels = onehot_labels.cuda(async=True)
                obj,_ = model.calculate_obj(epoch, inputs, onehot_labels)
            print("\tBATCH %d OF %d: %f"%(batch_ind+1, len(train_data_loader), obj.item()))
            obj.backward()
            opt.step()
            train_obj_vals.append(obj.item())
            avg_obj += obj.item()
            obj_count += 1
        end = time.time()
        train_logger.plot_obj_val(avg_obj/obj_count)
    train_results.append(test(model, train_data, params))
    print("FINAL TRAIN RESULTS: ",train_results[-1])
    val_results.append(test(model, val_data, params))
    print("FINAL VAL RESULTS: ",val_results[-1])
    if val_results[-1][0] < best_acc[0]:
        best_acc = val_results[-1]
        best_acc_epoch = epoch
        print("NEW BEST ACCURACY FOUND, SAVING MODEL")
        save_model(model, working_dir, 'model', params)
    train_logger.plot_for_current_epoch('Hamming Loss', train_results[-1][0])
    val_logger.plot_for_current_epoch('Hamming Loss', val_results[-1][0])
    save_model(model, working_dir, 'model_checkpoint', params)
    print("BEST VAL RESULTS (EPOCH %d): "%best_acc_epoch, best_acc)

    return train_obj_vals, train_results, val_results

def save_model(model, path, base_name, params):
    model_path = os.path.join(path, 'unary_%s'%base_name)
    model.save_unary(model_path)

    if params['model'] in ['struct', 'gspen'] and params['model'] != 'combined_struct' and not params.get('use_combined_struct_old') and not params.get('use_combined_struct'):
        model_path = os.path.join(path, 'pair_%s'%base_name)
        model.save_pair(model_path)

    if params['model'] in ['spen', 'gspen']:
        if params.get('t_version') in ['full_t_v1', 't_v2', 't_v3'] and params['model'] != 'spen':
            model_path = os.path.join(path, 't_%s'%base_name)
            model.save_t(model_path)
        else:
            model_path = os.path.join(path, 'unary_t_%s'%base_name)
            model.save_unary_t(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training tagging models')
    parser.add_argument('model', choices=['unary', 'spen', 'gspen'])
    parser.add_argument('working_dir')
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--t', choices=['linear', 'quad', 'skinny_quad', 'mlp'])
    parser.add_argument('--t_unary', choices=['linear', 'quad', 'skinny_quad', 'mlp'])
    parser.add_argument('--t_version', choices=['t_v1', 'full_t_v1', 't_v2', 'full_t_v2', 't_v3'])
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--combined_lr', type=float)
    parser.add_argument('--combined_wd', type=float, default=0.)
    parser.add_argument('--combined_mom', type=float, default=0.)
    parser.add_argument('--unary_lr', type=float)
    parser.add_argument('--unary_mom', type=float, default=0.)
    parser.add_argument('--pair_lr', type=float)
    parser.add_argument('--pair_mom', type=float, default=0.)
    parser.add_argument('--t_unary_lr', type=float)
    parser.add_argument('--t_pair_lr', type=float)
    parser.add_argument('--t_lr', type=float)
    parser.add_argument('--t_mom', type=float, default=0.)
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--t_hidden_size', type=int, default=130)
    parser.add_argument('--t_num_layers', type=int, default=2)
    parser.add_argument('--unary_num_layers', type=int, default=2)
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
    parser.add_argument('--pretrain_pair_t')
    parser.add_argument('--unary_wd', type=float, default=0.)
    parser.add_argument('--pair_wd', type=float, default=0.)
    parser.add_argument('--t_unary_wd', type=float, default=0.)
    parser.add_argument('--t_unary_mom', type=float, default=0.)
    parser.add_argument('--t_pair_wd', type=float, default=0.)
    parser.add_argument('--t_wd', type=float, default=0.)
    parser.add_argument('--train_data_file')
    parser.add_argument('--val_data_file')
    parser.add_argument('--test_data_file')
    parser.add_argument('--img_dir')
    parser.add_argument('--labels_dir')
    parser.add_argument('--process_data', action='store_true')
    parser.add_argument('--mp_eps', type=float, default=0.)
    parser.add_argument('--mp_itrs', type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--pair_ones_init', action='store_true')
    parser.add_argument('--pair_zeros_init', action='store_true')
    parser.add_argument('--inf_mode', choices=['lp', 'mp', 'md'])
    parser.add_argument('--ignore_dropout', action='store_true', default=False)
    parser.add_argument('--no_t_pots', action='store_true')
    parser.add_argument('--unary_simple', action='store_true')
    parser.add_argument('--pair_simple', action='store_true')
    parser.add_argument('--pair_hidden_size', type=int, default=276*4)
    parser.add_argument('--use_softmax', action='store_true')
    parser.add_argument('--inf_eps', type=float)
    parser.add_argument('--use_better_pairs', action='store_true')
    parser.add_argument('--pair_finetune', action='store_true')
    parser.add_argument('--pair_dropout', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--interleaved_mp_itrs', type=int, default=1)
    parser.add_argument('--interleaved_itrs', type=int, default=1)
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--load_old_struct')
    parser.add_argument('--load_old_unary')
    parser.add_argument('--use_combined_struct', action='store_true')
    parser.add_argument('--use_combined_struct_old', action='store_true')
    parser.add_argument('--use_old_unary', action='store_true')
    parser.add_argument('--inf_region_eps', type=float)
    parser.add_argument('--use_cross_ent', action='store_true')
    parser.add_argument('--ignore_unary_dropout', action='store_true')
    parser.add_argument('--t_use_hardtanh', action='store_true')
    parser.add_argument('--t_use_softplus', action='store_true')
    parser.add_argument('--entropy_coef', type=float, default=1)
    parser.add_argument('--t_normalize_pots', action='store_true')
    parser.add_argument('--t_identity_init', action='store_true')
    parser.add_argument('--manual_init_pairs', action='store_true')
    parser.add_argument('--fix_unaries', action='store_true')
    parser.add_argument('--load_combined_unary')

    args = parser.parse_args()
    params = vars(args)
    if args.process_data:
        if args.model == 'unary':
            print("PROCESSING UNARY TRAINING DATA")
            train_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, TRAIN, args.train_data_file, None)
            print("PROCESSING UNARY VALIDATION DATA")
            val_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, VAL, args.val_data_file, None)
            print("PROCESSING UNARY TESTING DATA")
            test_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, TEST, args.test_data_file, None)
        elif args.model == 'struct':
            pairs = []
            num_nodes = 24
            for n1 in range(num_nodes):
                for n2 in range(n1+1, num_nodes):
                    pairs.append((n1, n2))
            print("PROCESSING STRUCT TRAINING DATA")
            train_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, TRAIN, args.train_data_file, pairs)
            print("PROCESSING STRUCT VALIDATION DATA")
            val_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, VAL, args.val_data_file, pairs)
            print("PROCESSING STRUCT TESTING DATA")
            test_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, TEST, args.test_data_file, pairs)

        sys.exit(0)

    num_nodes = 24
    num_vals = 2
    if args.model in ['struct', 'gspen']:
        pairs = []
        for n1 in range(num_nodes):
            for n2 in range(n1+1, num_nodes):
                pairs.append((n1, n2))
        if not args.use_combined_struct and not args.use_combined_struct_old:
            pair_model = PairModel(params)
    else:
        pairs = None
    train_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, TRAIN, args.train_data_file, pairs, load=args.load_data)
    val_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, VAL, args.val_data_file, pairs, load=args.load_data)
    if args.test:
        test_data = FlickrTaggingDataset(args.img_dir, args.labels_dir, TEST, args.test_data_file, pairs, load=args.load_data)
    if args.model == 'combined_struct' or args.use_combined_struct:
        unary_model = combined_model = CombinedPotModel(params, pairs)
        pair_model = None
        if args.load_combined_unary:
            unary_model.load_from_unary(args.load_combined_unary)
    elif args.model == 'combined_struct_old' or args.use_combined_struct_old:
        unary_model = combined_model = CombinedPotModel_Old(params, pairs)
        if args.load_old_struct is not None:
            combined_model.load_from_old(args.load_old_struct)
        pair_model = None
    elif args.use_old_unary:
        unary_model = UnaryModel_Old(params)
        if args.load_old_unary is not None:
            unary_model.load_from_old(args.load_old_unary)
    else:
        unary_model = UnaryModel(params)
        if args.model in ['struct', 'gspen']:
            pair_model = PairModel(params)
        else:
            pair_model = None
        
    
    
    if args.model in ['spen', 'gspen']:
        if args.t == 'linear':
            t_constructor = LinearTModel
        elif args.t == 'quad':
            t_constructor = QuadTModel
        elif args.t == 'skinny_quad':
            t_constructor = SkinnyQuadTModel
        elif args.t == 'mlp':
            t_constructor = MLPTModel

        if args.t_version == 't_v1':
            t_model = SplitTModelV1(t_constructor, None, num_nodes, pairs, num_vals, params)
        elif args.t_version == 'full_t_v1':
            t_model = TModelV1(t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v2' or 'full_t_v2':
            t_model = TModelV2(t_constructor, num_nodes, pairs, num_vals, params)
        elif args.t_version == 't_v3':
            t_model = TModelV3(t_constructor, num_nodes, pairs, num_vals, params)
        if args.model == 'spen':
            model = SPENModel(unary_model, t_model, num_nodes, num_vals, params)
        elif args.model == 'gspen':
            model = GSPENModel(unary_model, pair_model, t_model, num_nodes, pairs, num_vals, params)
    elif args.model in ['combined_struct', 'combined_struct_old']:
        model = StructModel(combined_model, None, num_nodes, pairs, num_vals, params)

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
    print("MODEL: ",model)

    if args.test:
        print("TESTING ON TRAINING DATA...")
        train_hamming, train_macros, train_micros, train_label_results = expanded_test(model, train_data, params)
        print("TESTING ON VALIDATION DATA...")
        val_hamming, val_macros, val_micros, val_label_results = expanded_test(model, val_data, params)
        print("TESTING ON TEST DATA...")
        test_hamming, test_macros, test_micros, test_label_results = expanded_test(model, test_data, params)

        print("FINAL RESULTS:")
        print("Hamming Results: ")
        print("\tTrain: %f"%train_hamming)
        print("\tVal  : %f"%val_hamming)
        print("\tTest : %f"%test_hamming)
        print("Macro Results:")
        print("\tTrain: %f, %f, %f"%train_macros)
        print("\tVal  : %f, %f, %f"%val_macros)
        print("\tTest : %f, %f, %f"%test_macros)
        print("Micro Results:")
        print("\tTrain: %f, %f, %f"%train_micros)
        print("\tVal  : %f, %f, %f"%val_micros)
        print("\tTest : %f, %f, %f"%test_micros)
        print("Label Results:")
        for i in range(24):
            print("\tLabel %d: %s"%(i, reverse_order[i][:-4]))
            print("\t\tTrain: %f, %f, %f"%(train_label_results[0][i], train_label_results[1][i], train_label_results[2][i]) )
            print("\t\tVal  : %f, %f, %f"%(val_label_results[0][i], val_label_results[1][i], val_label_results[2][i]) )
            print("\t\tTest : %f, %f, %f"%(test_label_results[0][i], test_label_results[1][i], test_label_results[2][i]) )

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
        train_obj_vals, train_scores, val_scores = train(model, train_data, val_data, params, loggers)
        
        train_logger.close()
        val_logger.close()