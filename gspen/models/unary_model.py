import torch
from torch import nn

def labels2onehot(labels, num_vals, gpu):
    if gpu:
        tensor_mod = torch.cuda
    else:
        tensor_mod = torch
    new_labels = labels.view(-1,1)
    onehot = tensor_mod.FloatTensor(len(new_labels), num_vals).fill_(0.)
    onehot.scatter_(1, new_labels, 1)
    return onehot.view(labels.size(0), -1)

class UnaryModel(nn.Module):
    def __init__(self, unary_model, num_nodes, num_vals, params):
        super(UnaryModel, self).__init__()
        self.unary_model = unary_model
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.gpu = params.get('gpu', False)
        self.use_loss_aug = params.get('use_loss_aug', False)
        self.use_cross_ent = params.get('use_cross_ent', False)
        if self.gpu:
            self.unary_model.cuda()

    def get_optimizer(self, params):
        lr = params['unary_lr']
        wd = params.get('unary_wd', 0.)
        mom = params.get('unary_mom', 0.)
        nesterov = params.get('use_nesterov', False)
        model_params = filter(lambda p: p.requires_grad, self.unary_model.parameters())
        if params.get('use_adam', False):
            opt = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=mom, nesterov=nesterov)
        return opt

    def calculate_obj(self, epoch, inp, onehot_labels, belief_masks=None, masks=None):
        self.train()
        self.zero_grad()
        if masks is not None:
            lens = masks.sum(dim=1).int()
            pots = self.unary_model(inp, lens)
        else:
            pots = self.unary_model(inp)
        if self.use_loss_aug:
            if self.use_cross_ent:
                actual_onehot = labels2onehot(onehot_labels, self.num_vals, self.gpu)
                pots = pots + (1 - actual_onehot)
            else:
                pots = pots + (1 - onehot_labels)
        if self.use_cross_ent:
            if belief_masks is not None:
                return (nn.CrossEntropyLoss(reduce=False, size_average=False)(pots.view(-1, self.num_vals), onehot_labels.view(-1))*masks.view(-1)).sum()/masks.sum(), torch.FloatTensor([0.])
            else:
                return nn.CrossEntropyLoss()(pots.view(-1, self.num_vals), onehot_labels.view(-1)), torch.FloatTensor([0.])
        else:
            prediction = pots.view(-1, self.num_vals).argmax(dim=1).view(pots.size(0), -1)
            onehot_prediction = labels2onehot(prediction, self.num_vals, self.gpu)
            if belief_masks is not None:
                pots = pots * belief_masks
            return (pots*(onehot_prediction - onehot_labels)).sum()/pots.size(0), (pots*onehot_prediction).sum()/pots.size(0)

    def calculate_beliefs(self, inp, belief_masks=None):
        self.unary_model.eval()
        pred = self.unary_model(inp)
        return nn.Softmax(dim=1)(pred.view(-1, self.num_vals)).view(inp.size(0), -1)

    def predict(self, inp, masks=None, belief_masks=None, threshold=None):
        self.eval()
        if masks is not None:
            lens = masks.sum(dim=1).int()
            pred = self.unary_model(inp, lens)
        else:
            pred = self.unary_model(inp)
        if threshold is not None:
            return (nn.Softmax(dim=1)(pred.view(-1, self.num_vals))[:, 1] > threshold).long().view(-1, self.num_nodes)
        else:
            return pred.view(-1, self.num_vals).argmax(dim=1).view(pred.size(0), -1)

    def save_unary(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.unary_model.state_dict()]
            torch.save(result, fout)

    def load_unary(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.unary_model.load_state_dict(params[0])