import torch
from torch import nn
import numpy as np
from gspen.fastmp import fastmp
try:
    from gspen.fastmp import ilpinf
    has_ilpinf = True
except:
    has_ilpinf = False



class StructModel(nn.Module):
    def __init__(self, unary_model, pair_model, num_nodes, pairs, num_vals, params):
        super(StructModel, self).__init__()
        self.unary_model = unary_model
        self.pair_model = pair_model
        self.combined_pots = self.pair_model is None
        self.num_nodes = num_nodes
        self.pairs = pairs
        self.num_vals = num_vals
        self.gpu = params.get('gpu', False)
        self.use_loss_aug = params.get('use_loss_aug', False)
        self.loss_aug_coef = params.get('loss_aug_coef', 1.)
        self.mp_eps = params.get('mp_eps', 0.)
        self.mp_itrs = params.get('mp_itrs')
        self.ignore_dropout = params.get('ignore_dropout', False)
        self.inf_mode = params.get('inf_mode', 'mp')
        if self.gpu:
            self.unary_model.cuda()
            if not self.combined_pots:
                self.pair_model.cuda()
        batch_size = params['batch_size']
        self.num_potentials = self.num_nodes*self.num_vals + self.num_vals*self.num_vals*len(self.pairs)
        self.beliefs = torch.FloatTensor(batch_size, self.num_potentials).fill_(0.)
        if self.inf_mode in ['lp', 'ilp'] and not has_ilpinf:
            raise ValueError('Inference Mode %s is only valid if this library is installed while a valid Gurobi install is present.'%self.inf_mode)

        if self.inf_mode == 'mp': 
            self.mp_graphs = [fastmp.FastMP(num_nodes, num_vals, pairs, np.zeros(self.num_potentials, dtype=np.float32)) for _ in range(batch_size)]
            self.num_msgs = self.mp_graphs[0].get_num_msgs()
            self.msgs = torch.FloatTensor(batch_size, self.num_msgs)
            for idx, mp_graph in enumerate(self.mp_graphs):
                mp_graph.allocate_mp_mem(self.mp_eps)
                mp_graph.update_msgs(self.msgs[idx, :].numpy())
        elif self.inf_mode == 'lp':
            self.mp_graphs = [ilpinf.ILPInf(num_nodes, num_vals, pairs, np.zeros(self.num_potentials, dtype=np.float32), True) for _ in range(batch_size)]
        elif self.inf_mode == 'ilp':
            self.mp_graphs = [ilpinf.ILPInf(num_nodes, num_vals, pairs, np.zeros(self.num_potentials, dtype=np.float32), False) for _ in range(batch_size)]

        for idx, mp_graph in enumerate(self.mp_graphs):
            mp_graph.update_beliefs_pointer(self.beliefs[idx, :].numpy())

    def get_optimizer(self, params):
        unary_lr = params.get('unary_lr', None)
        pair_lr = params.get('pair_lr', None)
        combined_lr = params.get('combined_lr', None)
        wd = params.get('wd', 0.)
        combined_mom = params.get('combined_mom', None)
        unary_wd = params.get('unary_wd', 0.)
        pair_wd = params.get('pair_wd', 0.)
        nesterov = params.get('use_nesterov', False)
        unary_mom = params.get('unary_mom', 0.)
        pair_mom = params.get('pair_mom', 0.)
        if self.combined_pots:
            param_groups = [{'params':self.unary_model.parameters(), 'lr':combined_lr, 'weight_decay':wd, 'momentum':combined_mom}]
        else:
            param_groups = [{'params':self.unary_model.parameters(), 'lr':unary_lr, 'weight_decay':unary_wd, 'momentum':unary_mom},
                    {'params':self.pair_model.parameters(), 'lr':pair_lr, 'weight_decay':pair_wd, 'momentum':pair_mom},]
        if params.get('use_adam', False):
            opt = torch.optim.Adam(param_groups)
        else:
            opt = torch.optim.SGD(param_groups, nesterov=nesterov)
        return opt

    def _do_inf(self, pots):
        mp_graphs = self.mp_graphs[:pots.size(0)]
        for mp_graph_ind, mp_graph in enumerate(mp_graphs):
            graph_pots = pots[mp_graph_ind, :].data.cpu().numpy()
            mp_graph.update_potentials(graph_pots)
        if self.inf_mode == 'mp':
            self.msgs[:pots.size(0), :].fill_(0.)
            fastmp.runmp(mp_graphs, self.mp_itrs, self.mp_eps)
            fastmp.update_beliefs(mp_graphs, self.mp_eps)
        else:
            ilpinf.runilp(mp_graphs)
        return self.beliefs[:pots.size(0), :]

    def _get_loss_aug(self, belief_labels):
        if self.gpu:
            loss_aug = torch.cuda.FloatTensor(belief_labels.size(0), belief_labels.size(1)).fill_(0.)
        num_unary = self.num_nodes*self.num_vals
        loss_aug[:, :num_unary] = 1-belief_labels[:, :num_unary]
        return self.loss_aug_coef*loss_aug

    def calculate_pots(self, inp):
        if self.combined_pots:
            return self.unary_model(inp)
        else:
            return torch.cat([self.unary_model(inp), self.pair_model(inp)], dim=1)

    def calculate_obj(self, epoch, inp, belief_labels, belief_masks=None, masks=None):
        self.train()
        self.zero_grad()
        pots = self.calculate_pots(inp)
        if self.use_loss_aug:
            loss_aug = self._get_loss_aug(belief_labels)
            pots = pots + loss_aug
        beliefs = self._do_inf(pots)
        if self.gpu:
            beliefs = beliefs.cuda()
            
        if self.use_loss_aug:
            inf_obj = ((pots+loss_aug)*beliefs).sum(dim=1)
            if self.mp_eps > 0:
                inf_obj = inf_obj - (beliefs*torch.log(beliefs+1e-6)).sum(dim=1)
            return (inf_obj - (pots*belief_labels).sum(dim=1)).mean(), inf_obj.mean()
        inf_obj = (pots*beliefs).sum(dim=1)
        if self.mp_eps > 0:
            inf_obj = inf_obj - (beliefs*torch.log(beliefs+1e-6)).sum(dim=1)
        return (inf_obj - (pots*belief_labels).sum(dim=1)).mean(), inf_obj.mean()

    def calculate_beliefs(self, inp):
        self.eval()
        pots = self.calculate_pots(inp)
        beliefs = self._do_inf(pots)

        if self.gpu:
            beliefs = beliefs.cuda()
        return beliefs

    def predict(self, inp):
        self.eval()
        pots = self.calculate_pots(inp)
        beliefs = self._do_inf(pots)
        predictions = beliefs[:, :self.num_nodes*self.num_vals].contiguous().view(-1, self.num_vals).argmax(dim=1).view(len(inp), -1)
        if self.gpu:
            predictions = predictions.cuda()
        return predictions

    def save_unary(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.unary_model.state_dict()]
            torch.save(result, fout)

    def load_unary(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.unary_model.load_state_dict(params[0])

    def save_pair(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.pair_model.state_dict()]
            torch.save(result, fout)

    def load_pair(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.pair_model.load_state_dict(params[0])