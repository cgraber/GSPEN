import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class SplitUnaryTModelV1(nn.Module):
    def __init__(self, unary_t_model_constructor, params):
        super(SplitUnaryTModelV1, self).__init__()
        self.no_t_pots = params.get('no_t_pots', False)
        self.use_t_input = params.get('use_t_input', False)
        if self.no_t_pots:
            self.unary_t_model = unary_t_model_constructor(params)
        else:
            self.unary_t_model = unary_t_model_constructor(params)
    
    def forward(self, beliefs, pots, inputs=None):
        obj = (beliefs*pots).sum(dim=1)
        if self.use_t_input:
            obj = obj + self.unary_t_model(beliefs, pots, inputs)
        elif self.no_t_pots:
            obj = obj + self.unary_t_model(beliefs)
        else:
            obj = obj + self.unary_t_model(beliefs, pots)
            #print("AVG UNARY T SCORE: ",(self.unary_t_model(unary_inp).squeeze().mean()))
        return obj



class SplitTModelV1(nn.Module):
    def __init__(self, unary_t_model_constructor, pair_t_model_constructor, num_nodes, pairs, num_vals, params):
        super(SplitTModelV1, self).__init__()
        self.no_t_pots = params.get('no_t_pots', False)
        self.use_t_input = params.get('use_t_input', False)
        self.num_unary_pots = num_nodes*num_vals
        if unary_t_model_constructor is not None:
            self.use_unary = True
            if self.no_t_pots:
                self.unary_t_model = unary_t_model_constructor(self.num_unary_pots, params)
            else:
                self.unary_t_model = unary_t_model_constructor(self.num_unary_pots*2, params)
        else:
            self.use_unary = False
        if pair_t_model_constructor is not None:
            self.use_pair = True
            self.num_pair_pots = num_vals*num_vals*len(pairs)
            if self.no_t_pots:
                self.pair_t_model = pair_t_model_constructor(self.num_pair_pots, params)
            else:
                self.pair_t_model = pair_t_model_constructor(self.num_pair_pots*2, params)
        else:
            self.use_pair = False

    def forward(self, beliefs, pots, inputs=None):
        obj = (beliefs*pots).sum(dim=1)
        if self.use_unary:
            if self.use_t_input:
                obj = obj + self.unary_t_model(beliefs[:, :self.num_unary_pots], pots[:, :self.num_unary_pots], inputs)
            else:
                if self.no_t_pots:
                    unary_inp = beliefs[:, :self.num_unary_pots]
                else:
                    unary_inp = torch.cat([beliefs[:, :self.num_unary_pots], pots[:, :self.num_unary_pots]], dim=1)
                obj = obj + self.unary_t_model(unary_inp).squeeze()
        if self.use_pair:
            if self.no_t_pots:
                unary_inp = beliefs[:, self.num_unary_pots:]
            else:
                pair_inp = torch.cat([beliefs[:, self.num_unary_pots:], pots[:, self.num_unary_pots:]], dim=1)
            obj = obj + self.pair_t_model(pair_inp).squeeze()
        return obj

    def get_t_input_gradient(self, beliefs, pots, inputs):
        if self.use_unary:
            if self.use_t_input:
                obj = self.unary_t_model(beliefs[:, :self.num_unary_pots], pots[:, :self.num_unary_pots], inputs)
            else:
                if self.no_t_pots:
                    unary_inp = beliefs[:, :self.num_unary_pots]
                else:
                    unary_inp = torch.cat([beliefs[:, :self.num_unary_pots], pots[:, :self.num_unary_pots]], dim=1)
                obj = self.unary_t_model(unary_inp).squeeze()
        pred_grads = torch.autograd.grad(obj.sum(), beliefs, create_graph=True)[0]
        return pred_grads
        
class InputSplitTModelV1(nn.Module):
    def __init__(self, unary_t_model_constructor, pair_t_model_constructor, num_nodes, pairs, num_vals, params):
        super(InputSplitTModelV1, self).__init__()
        self.no_t_pots = params.get('no_t_pots', False)
        self.num_unary_pots = num_nodes*num_vals
        if unary_t_model_constructor is not None:
            self.use_unary = True
            if self.no_t_pots:
                self.unary_t_model = unary_t_model_constructor(self.num_unary_pots, params)
            else:
                self.unary_t_model = unary_t_model_constructor(self.num_unary_pots*2, params)
        else:
            self.use_unary = False
        if pair_t_model_constructor is not None:
            self.use_pair = True
            self.num_pair_pots = num_vals*num_vals*len(pairs)
            if self.no_t_pots:
                self.pair_t_model = pair_t_model_constructor(self.num_pair_pots, params)
            else:
                self.pair_t_model = pair_t_model_constructor(self.num_pair_pots*2, params)
        else:
            self.use_pair = False

    def forward(self, beliefs, pots, inputs):
        obj = (beliefs*pots).sum(dim=1)
        if self.use_unary:
            unary_beliefs = beliefs[:, :self.num_unary_pots]
            unary_pots = pots[:, :self.num_unary_pots]
            obj = obj + self.unary_t_model(unary_beliefs, unary_pots, inputs).squeeze()
        if self.use_pair:
            pair_beliefs = beliefs[:, self.num_unary_pots:]
            pair_pots = pots[:, self.num_unary_pots:]
            obj = obj + self.pair_t_model(pair_beliefs, pair_pots, inputs).squeeze()
        return obj

class InputSplitTModelV2(nn.Module):
    def __init__(self, unary_t_model_constructor, pair_t_model_constructor, num_nodes, pairs, num_vals, params):
        super(InputSplitTModelV2, self).__init__()
        self.no_t_pots = params.get('no_t_pots', False)
        self.num_unary_pots = num_nodes*num_vals
        if unary_t_model_constructor is not None:
            self.use_unary = True
            if self.no_t_pots:
                self.unary_t_model = unary_t_model_constructor(self.num_unary_pots, params)
            else:
                self.unary_t_model = unary_t_model_constructor(self.num_unary_pots*2, params)
        else:
            self.use_unary = False
        if pair_t_model_constructor is not None:
            self.use_pair = True
            self.num_pair_pots = num_vals*num_vals*len(pairs)
            if self.no_t_pots:
                self.pair_t_model = pair_t_model_constructor(self.num_pair_pots, params)
            else:
                self.pair_t_model = pair_t_model_constructor(self.num_pair_pots*2, params)
        else:
            self.use_pair = False

    def forward(self, beliefs, pots, inputs):
        obj = 0
        if self.use_unary:
            unary_beliefs = beliefs[:, :self.num_unary_pots]
            unary_pots = pots[:, :self.num_unary_pots]
            obj = obj + self.unary_t_model(unary_beliefs, unary_pots, inputs).squeeze()
        if self.use_pair:
            pair_beliefs = beliefs[:, self.num_unary_pots:]
            pair_pots = pots[:, self.num_unary_pots:]
            obj = obj + self.pair_t_model(pair_beliefs, pair_pots, inputs).squeeze()
        return obj



class SplitTModelV2(nn.Module):
    def __init__(self, unary_t_model_constructor, pair_t_model_constructor, num_nodes, pairs, num_vals, params):
        super(SplitTModelV2, self).__init__()
        self.num_unary_pots = num_nodes*num_vals
        if unary_t_model_constructor is not None:
            self.use_unary = True
            self.unary_t_model = unary_t_model_constructor(self.num_unary_pots*2, params)
        else:
            self.use_unary = False
        if pair_t_model_constructor is not None:
            self.use_pair = True
            self.num_pair_pots = num_vals*num_vals*len(pairs)
            self.pair_t_model = pair_t_model_constructor(self.num_pair_pots*2, params)
        else:
            self.use_pair = False

    def forward(self, beliefs, pots):
        obj = (beliefs*pots).sum(dim=1)
        if self.use_unary:
            unary_inp = torch.cat([beliefs[:, :self.num_unary_pots], pots[:, :self.num_unary_pots]], dim=1)
            obj = obj + self.unary_t_model(unary_inp).squeeze()
        else:
            obj = (beliefs[:, :self.num_unary_pots]*pots[:, :self.num_unary_pots]).sum(dim=1)
        if self.use_pair:
            pair_inp = torch.cat([beliefs[:, self.num_unary_pots:], pots[:, self.num_unary_pots:]], dim=1)
            obj = obj + self.pair_t_model(pair_inp).squeeze()
        else:
            obj = obj + (beliefs[:, self.num_unary_pots:]*pots[:, self.num_unary_pots:]).sum(dim=1)
        return obj

            


class TModelV1(nn.Module):
    def __init__(self, t_model_constructor, num_nodes, pairs, num_vals, params):
        super(TModelV1, self).__init__()
        self.no_t_pots = params.get('no_t_pots', False)
        self.use_t_input = params.get('use_t_input', False)
        if pairs is None:
            num_pots = num_nodes*num_vals
        else:
            num_pots = num_nodes*num_vals + num_vals*num_vals*len(pairs)
        if self.no_t_pots:
            self.t_model = t_model_constructor(num_pots, params)
        else:
            self.t_model = t_model_constructor(num_pots*2, params)

    def forward(self, beliefs, pots, inputs=None):
        if self.use_t_input:
            t_part = self.t_model(beliefs, pots, inputs)
        else:
            if self.no_t_pots:
                inputs = beliefs
            else:
                inputs = torch.cat([beliefs, pots], dim=1)
            t_part = self.t_model(inputs).squeeze()
        return (beliefs*pots).sum(dim=1) + t_part


class TModelV2(nn.Module):
    def __init__(self, t_model_constructor, num_nodes, pairs, num_vals, params):
        super(TModelV2, self).__init__()
        self.normalize_pots = params.get('t_normalize_pots', False)
        self.use_t_input = params.get('use_t_input', False)
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.num_unary = num_nodes*num_vals
        if pairs is None:
            self.use_pairs = False
            num_pots = num_nodes*num_vals
            self.unary_t_model = self.t_model = t_model_constructor(num_pots*2, params)
        else:
            self.use_pairs = True
            self.num_pairs = len(pairs)
            num_pots = num_nodes*num_vals + num_vals*num_vals*len(pairs)
            self.t_model = t_model_constructor(num_pots*2, params)

    def forward(self, beliefs, pots, inputs=None):
        if self.use_t_input:
            return self.t_model(beliefs, pots, inputs).squeeze()
        else:
            if self.normalize_pots:
                if self.use_pairs:
                    unaries = F.softmax(pots[:, :self.num_unary].view(pots.size(0), self.num_nodes, self.num_vals), dim=-1)
                    pairs = F.softmax(pots[:, self.num_unary:].view(pots.size(0), self.num_pairs, -1), dim=-1)
                    pots = torch.cat([unaries.view(pots.size(0), -1), pairs.view(pots.size(0), -1)], dim=1)
                else:
                    pots = F.softmax(pots.view(pots.size(0), self.num_nodes, self.num_vals), dim=-1).view(pots.size(0), -1)
            inputs = torch.cat([beliefs, pots], dim=1)
            return self.t_model(inputs).squeeze()


class TModelV3(nn.Module):
    def __init__(self, t_model_constructor, num_nodes, pairs, num_vals, params):
        super(TModelV3, self).__init__()
        if pairs is None:
            num_pots = num_nodes*num_vals
        else:
            num_pots = num_nodes*num_vals + num_vals*num_vals*len(pairs)
        self.normalize_pots = params.get('t_normalize_pots', False)
        self.t_model = t_model_constructor(num_pots, params)
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.num_unary = num_nodes*num_vals
        self.num_pairs = len(pairs)

    def forward(self, beliefs, pots):
        if self.normalize_pots:
            unaries = F.softmax(pots[:, :self.num_unary].view(pots.size(0), self.num_nodes, self.num_vals), dim=-1)
            pairs = F.softmax(pots[:, self.num_unary:].view(pots.size(0), self.num_pairs, -1), dim=-1)
            pots = torch.cat([unaries.view(pots.size(0), -1), pairs.view(pots.size(0), -1)], dim=1)
        return self.t_model(beliefs*pots).squeeze()

class QuadTModel(nn.Module):
    def __init__(self, input_size, params):
        super(QuadTModel, self).__init__()
        self.quad_part = nn.Parameter(torch.FloatTensor(input_size, input_size))
        val = 1/np.sqrt(input_size)
        self.quad_part.data.uniform_(-val, val)
        self.linear_part = nn.Linear(input_size, 1)

    def forward(self, inp):
        tmp_inp = inp.unsqueeze(1)
        result = torch.matmul(tmp_inp, self.quad_part)
        result = -1* torch.matmul(result, result.transpose(1,2)).squeeze(2) + self.linear_part(inp)
        return result

class SkinnyQuadTModel(nn.Module):
    def __init__(self, input_size, params):
        super(SkinnyQuadTModel, self).__init__()
        self.quad_part = nn.Parameter(torch.FloatTensor(input_size, 1))
        val = 1/np.sqrt(input_size)
        self.quad_part.data.uniform_(-val, val)
        self.linear_part = nn.Linear(input_size, 1)

    def forward(self, inp):
        tmp_inp = inp.unsqueeze(1)
        quad_mat = torch.matmul(self.quad_part, self.quad_part.t())
        result = torch.matmul(tmp_inp, quad_mat)
        result = -1*torch.matmul(result, tmp_inp.transpose(1,2)).squeeze(2) + self.linear_part(inp)
        return result.squeeze()


class MLPTModel(nn.Module):
    def __init__(self, input_size, params):
        super(MLPTModel, self).__init__()
        hidden_size = params.get('t_hidden_size', 2)
        num_layers = params.get('t_num_layers', 2)
        t_first_dropout = params.get('t_first_dropout', None)
        t_last_dropout = params.get('t_last_dropout', None)
        use_softplus = params.get('t_use_softplus', False)
        use_hardtanh = params.get('t_use_hardtanh', False)
        use_relu = params.get('t_use_relu', False)
    
        if use_softplus:
            layers = [nn.Linear(input_size, hidden_size), nn.Softplus()]
        elif use_hardtanh:
            layers = [nn.Linear(input_size, hidden_size), nn.Hardtanh()]
        elif use_relu:
            layers = [nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True)]
        else:
            layers = [nn.Linear(input_size, hidden_size), nn.LeakyReLU(negative_slope=0.25)]
        if t_first_dropout is not None:
            layers.insert(0, nn.Dropout(p=t_first_dropout))
        for layer in range(num_layers-2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_softplus:
                layers.append(nn.Softplus())
            elif use_hardtanh:
                layers.append(nn.Hardtanh())
            elif use_relu:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.25))
        if t_last_dropout is not None:
            layers.append(nn.Dropout(p=t_last_dropout))
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        return self.model(inp).squeeze()

LinearTModel = lambda size, params: nn.Linear(size, 1)
