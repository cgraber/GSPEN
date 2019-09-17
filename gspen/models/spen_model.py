import torch
import torch.nn as nn
import numpy as np

def labels2onehot(labels, num_vals, gpu):
    if gpu:
        tensor_mod = torch.cuda
    else:
        tensor_mod = torch
    new_labels = labels.view(-1,1)
    onehot = tensor_mod.FloatTensor(len(new_labels), num_vals).fill_(0.)
    onehot.scatter_(1, new_labels, 1)
    return onehot.view(labels.size(0), -1)

class SPENModel(nn.Module):
    def __init__(self, unary_model, unary_t_model, num_nodes, num_vals, params):
        super(SPENModel, self).__init__()
        self.unary_model = unary_model
        self.unary_t_model = unary_t_model
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.num_inf_itrs = params.get('num_inf_itrs', 100)
        self.inf_lr = params.get('inf_lr', 0.05)
        self.use_sqrt_decay = params.get('use_sqrt_decay', False)
        self.use_linear_decay = params.get('use_linear_decay', False)
        self.use_relu = params.get('use_relu', False)
        self.inf_method = params.get('inf_method')
        self.gpu = params.get('gpu', False)
        self.use_loss_aug = params.get('use_loss_aug', False)
        self.ignore_unary_h_dropout = params.get('ignore_unary_h_dropout', False)
        self.use_entropy = params.get('use_entropy', False)
        self.inf_eps = params.get('inf_eps', None)
        self.inf_region_eps = params.get('inf_region_eps', None)
        self.inf_fw_eps = params.get('inf_fw_eps', None)
        self.use_t_input = params.get('use_t_input', False)
        self.verbose = params.get('verbose', False)
        self.record_inf = params.get('record_inf', False)
        self.use_t_dropout = params.get('use_t_dropout', False)
        self.inf_uniform_init = params.get('inf_uniform_init', False)
        self.accumulate_beliefs = params.get('accumulate_beliefs', False)
        self.return_all_vals = params.get('return_all_vals', False)
        self.entropy_coef = params.get('entropy_coef', 1.)
        if self.gpu:
            self.tensor_mod = torch.cuda
            self.unary_model.cuda()
            self.unary_t_model.cuda()
        else:
            self.tensor_mod = torch

    def get_optimizer(self, params):
        unary_lr = params['unary_lr']
        unary_wd = params.get('unary_wd', 0.)
        unary_mom = params.get('unary_mom', 0.)
        t_lr = params['t_unary_lr']
        t_wd = params.get('t_unary_wd', 0.)
        t_mom = params.get('t_unary_mom', 0.)
        unary_params = filter(lambda p: p.requires_grad, self.unary_model.parameters())
        unary_t_params = filter(lambda p: p.requires_grad, self.unary_t_model.parameters())
        if params.get('use_adam', False):
            opt = torch.optim.Adam([
                {'params':unary_params, 'lr':unary_lr, 'weight_decay':unary_wd},
                {'params':unary_t_params, 'lr':t_lr, 'weight_decay':t_wd},
                ])
        else:
            opt = torch.optim.SGD([
                {'params':unary_params, 'lr':unary_lr, 'weight_decay':unary_wd, 'momentum':unary_mom},
                {'params':unary_t_params, 'lr':t_lr, 'weight_decay':t_wd, 'momentum':t_mom},
                ])
        return opt

    def get_random_probabilities(self, num_samples, num_nodes=None):
        if num_nodes is None:
            num_nodes = self.num_nodes
        all_probs = []
        for _ in range(num_nodes):
            vals = [np.random.rand() for i in range(self.num_vals-1)]
            vals.sort()
            vals.append(1)
            vals.insert(0, 0)
            probs = [vals[i+1] - vals[i] for i in range(self.num_vals)]
            probabilities = torch.FloatTensor(probs)
            all_probs.append(probabilities)
        result = torch.cat(all_probs).expand(num_samples, num_nodes*self.num_vals).contiguous()
        if self.gpu:
            result = result.cuda(async=True)
        return result

    def _call_unary_t(self, predictions, pots, inputs, belief_masks):
        if belief_masks is None:
            if self.use_t_input:
                return self.unary_t_model(predictions, pots, inputs)
            else:
                return self.unary_t_model(predictions, pots)
        else:
            if predictions.size(1) != belief_masks.size(1):
                predictions = torch.cat([predictions, self.tensor_mod.FloatTensor(predictions.size(0), belief_masks.size(1) - predictions.size(1)).fill_(0.)], dim=1)
            if pots.size(1) != belief_masks.size(1):
                pots = torch.cat([pots, self.tensor_mod.FloatTensor(pots.size(0), belief_masks.size(1) - pots.size(1)).fill_(0.)], dim=1)
            return self.unary_t_model(predictions*belief_masks, pots*belief_masks)

    def _find_predictions(self, is_train, epoch, inputs, lossaug, labels, belief_masks, pots=None):
        if self.inf_method == 'fw':
            return self._fw_inf(is_train, epoch, inputs, lossaug, labels, belief_masks, pots)
        elif self.inf_method == 'emd':
            return self._emd_inf(is_train, inputs, lossaug, belief_masks, pots)
        else:
            raise Exception("Inference method not recognized: ",self.inf_method)

    def _emd_inf(self, is_train, inputs, lossaug, belief_masks, pots):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            batch_size = inputs[0].size(0)
        else:
            batch_size = inputs.size(0)
        
        if pots is not None:
            pots = pots.detach()
        else:
            pots = self.unary_model(inputs).detach()
        if belief_masks is not None:
            num_nodes = int(pots.size(1)/self.num_vals)
            prediction = self.get_random_probabilities(batch_size, num_nodes)
        else:
            prediction = self.get_random_probabilities(batch_size)
        pots.requires_grad=False
        if belief_masks is not None:
            if pots.size(1) != belief_masks.size(1):
                pots = torch.cat([pots, self.tensor_mod.FloatTensor(pots.size(0), belief_masks.size(1) - pots.size(1)).fill_(0.)], dim=1)
            pots = pots * belief_masks
            prediction = prediction*belief_masks

        prev_obj = self.tensor_mod.FloatTensor(batch_size).fill_(-float('inf'))
        if self.record_inf:
            inf_objs = []
        prev_preds = None
        prev_pred = prediction
        prediction_eps = float('inf')
        region_prediction_eps = float('inf')
        if is_train and self.accumulate_beliefs:
            all_preds = [prediction.clone()]
        for itr in range(self.num_inf_itrs):
            self.unary_t_model.zero_grad()
            pred_var = torch.autograd.Variable(prediction, requires_grad=True)
            vector_obj = self._call_unary_t(pred_var, pots, inputs, belief_masks)

            if lossaug is not None:
                vector_obj = vector_obj + (pred_var*lossaug).sum(dim=1)
            if self.use_entropy:
                vector_obj = vector_obj - self.entropy_coef*(pred_var*torch.log(pred_var+1e-6)).sum(dim=1)
            if self.record_inf:
                inf_objs.append(vector_obj[0].item())
            eps = torch.abs(vector_obj - prev_obj).max().item()
            if self.verbose:
                print("\t\t\t%d: %f, %f, %f, %f"%(itr, vector_obj[0].item(), eps, prediction_eps, region_prediction_eps))
                if prev_preds is None:
                    prev_preds = prediction.view(-1, self.num_vals)
                else:
                    norm_diff = torch.norm(prev_preds -prediction.view(-1, self.num_vals), dim=1)
                    max_diff_ind = torch.argmax(norm_diff)
                    print("\t\t\t\t\tPREV: ",prev_preds[max_diff_ind,:])
                    print("\t\t\t\t\tNEW: ",prediction.view(-1, self.num_vals)[max_diff_ind,:])
                    prev_preds = prediction.view(-1, self.num_vals)

            if self.inf_eps is not None and eps < self.inf_eps:
                if self.verbose:
                    print("\t\t\tBREAKING AFTER %d ITRS"%itr)
                break
            prev_obj = vector_obj.detach()

            obj = vector_obj.sum()
            obj.backward()
            if self.use_linear_decay:
                lr = self.inf_lr/(itr+1)
            else:
                lr = self.inf_lr/np.sqrt(itr+1)
            if self.gpu:
                lr = torch.cuda.FloatTensor([lr])
            grad = pred_var.grad
            result = lr*grad.view(-1, self.num_vals)
            max_r, max_inds = torch.max(result, dim=1)
            if self.verbose:
                print("\t\t\t\tNEXT ARGMAX: ",max_inds[0])
            result = prediction.view(-1, self.num_vals)*torch.exp(result - max_r.view(-1, 1))
            prediction = (result/(result.sum(dim=1, keepdim=True)+1e-6)).view(batch_size, -1)
            if belief_masks is not None:
                prediction = prediction*belief_masks
            if is_train and self.accumulate_beliefs:
                all_preds.append(prediction.clone())
            prediction_eps = torch.norm(prev_pred - prediction, dim=1).max().item()
            region_prediction_eps = torch.norm(prev_pred.view(-1, self.num_vals) - prediction.view(-1, self.num_vals), dim=1).max().item()
            if self.inf_region_eps is not None and region_prediction_eps < self.inf_region_eps:
                break
            prev_pred = prediction
        if self.verbose:
            print("\t\t\tFINAL EPS: ",eps)
        if is_train and self.accumulate_beliefs:
            prediction = torch.cat(all_preds, dim=0)
        if self.record_inf:
            return [prediction, inf_objs], itr
        else:
            return prediction, itr

    def _fw_inf(self, is_train, epoch, inputs, lossaug, labels, belief_masks, pots):
        lr = self.inf_lr
        if self.use_sqrt_decay or self.use_linear_decay:
            start_lr = lr
        if self.inf_uniform_init:
            prediction = self.tensor_mod.FloatTensor(inputs.size(0), self.num_nodes*self.num_vals).fill_(1/self.num_vals)
        prediction = self.get_random_probabilities(len(inputs))
        if pots is not None:
            pots = pots.detach()
        else:
            pots = self.unary_model(inputs).detach()
        pots.requires_grad=False
        prev_obj = self.tensor_mod.FloatTensor(inputs.size(0)).fill_(-float('inf'))
        region_prediction_eps = float('inf')
        if self.record_inf:
            inf_objs = []
        prev_pred = prediction
        prediction_eps = float('inf')
        region_prediction_eps = float('inf')
        max_region = 0
        eps = float('inf')
        if is_train and self.accumulate_beliefs:
            all_preds = [prediction.clone()]
        for itr in range(self.num_inf_itrs):
            self.unary_t_model.zero_grad()
            pred_var = torch.autograd.Variable(prediction, requires_grad=True)
            vector_obj = self._call_unary_t(pred_var, pots, inputs, belief_masks)
            
            if lossaug is not None:
                vector_obj = vector_obj + (pred_var*lossaug).sum(dim=1)
            if self.use_entropy:
                vector_obj = vector_obj - self.entropy_coef*(pred_var*torch.log(pred_var+1e-6)).sum(dim=1)
            if self.record_inf:
                inf_objs.append(vector_obj[0].item())
            eps = torch.abs(vector_obj - prev_obj).max().item()
            if self.inf_eps is not None and eps < self.inf_eps:
                break
            prev_obj = vector_obj
            obj = vector_obj.sum()
            
            obj.backward()
            grad = pred_var.grad
            max_p = grad.view(-1, self.num_vals).argmax(dim=1).view(len(inputs), -1)
            onehot_max_p = labels2onehot(max_p, self.num_vals, self.gpu)
            if self.gpu:
                onehot_max_p = onehot_max_p.cuda(async=True)
            fw_dir = onehot_max_p - prediction
            if self.inf_fw_eps is not None:
                fw_eps = (fw_dir*grad).sum(dim=1).max().item()
                if fw_eps < self.inf_fw_eps:
                    print("\t\t\tBREAKING AFTER %d ITRS"%itr)
                    break
            if self.use_sqrt_decay:
                lr = self.inf_lr/np.sqrt(itr+1)
            elif self.use_linear_decay:
                lr = self.inf_lr/(itr+1)
            if self.gpu:
                lr = torch.cuda.FloatTensor([lr])
            prediction = prediction + lr*fw_dir
            prediction_eps = torch.norm(prev_pred - prediction, dim=1).max().item()
            region_prediction_eps = torch.norm(prev_pred.view(-1, self.num_vals) - prediction.view(-1, self.num_vals), dim=1)
            region_prediction_eps, max_region = torch.max(region_prediction_eps, dim=0)
            region_prediction_eps, max_region = region_prediction_eps.item(), max_region.item()
            if self.inf_region_eps is not None and region_prediction_eps < self.inf_region_eps:
                break
            prev_pred = prediction
            if is_train and self.accumulate_beliefs:
                all_preds.append(prediction.clone())
        if self.inf_fw_eps is not None and self.verbose:
            print("\t\t\tFINAL FW EPS: ",fw_eps)
        if is_train and self.accumulate_beliefs:
            prediction = torch.cat(all_preds, dim=0)
        if self.record_inf:
            return prediction, inf_objs, itr
        else:
            return prediction, itr

    def calculate_obj(self, epoch, inputs, belief_labels, masks=None, belief_masks=None):
        onehot_labels = belief_labels
        self.unary_t_model.train()
        if self.ignore_unary_h_dropout:
            self.unary_model.eval()
        else:
            self.unary_model.train()
        if masks is not None:
            lens = masks.sum(dim=1).int()
            pots = self.unary_model(inputs, lens)
        else:
            pots = self.unary_model(inputs)
        if self.use_loss_aug:
            lossaug = 1 - onehot_labels
        else:
            lossaug = None
        results, num_iters = self._find_predictions(True, epoch, inputs, lossaug, onehot_labels, belief_masks, pots=pots)
        if self.record_inf:
            predictions, inf_objs = results
        else:
            predictions = results
        self.zero_grad()
        if self.accumulate_beliefs:
            input_dims = [int(predictions.size(0)/inputs.size(0))] + [1 for _ in range(len(inputs.shape)-1)]
            inputs = inputs.repeat(*input_dims)
            pots = pots.repeat(int(predictions.size(0)/pots.size(0)), 1)
            onehot_labels = onehot_labels.repeat(int(predictions.size(0)/onehot_labels.size(0)), 1)
            if lossaug is not None:
                lossaug = lossaug.repeat(int(predictions.size(0)/lossaug.size(0)), 1)
        inf_obj = self._call_unary_t(predictions, pots, inputs, belief_masks)
        if self.use_loss_aug:
            inf_obj = inf_obj + (predictions*lossaug).sum(dim=1)
        if self.use_entropy:
            inf_obj = inf_obj - self.entropy_coef*(predictions*torch.log(predictions+1e-6)).sum(dim=1)
        label_term = self._call_unary_t(onehot_labels, pots, inputs, belief_masks)
        obj = inf_obj - label_term

        if self.use_relu:
            obj = nn.ReLU()(obj)
        obj = obj.mean()
        
        if self.return_all_vals:
            return obj, inf_obj.mean(), label_term.mean(), num_iters
        else:
            return obj, inf_obj.sum()/pots.size(0)

    def calculate_beliefs(self, inputs, belief_masks=None):
        self.eval()
        return self._find_predictions(False, None, inputs, None, None, belief_masks)[0]

    def predict(self, inputs, belief_masks=None, threshold=None, masks=None):
        self.eval()
        if masks is not None:
            lens = masks.sum(dim=1).int()
            pots = self.unary_model(inputs, lens)
        else:
            pots = self.unary_model(inputs)
        if threshold is not None:
            # This should only be the case for binary problems. In which case, we only look to see if '1' predictions are above threshold
            return (self._find_predictions(False, None, inputs, None, None, belief_masks, pots=pots)[0].view(-1, self.num_vals)[:, 1] > threshold).long().view(-1, self.num_nodes)
        else:
            predictions = self._find_predictions(False, None, inputs, None, None, belief_masks, pots=pots)[0]
            if self.record_inf:
                predictions, inf_scores = predictions
            if self.num_nodes is None:
                num_nodes = int(pots.size(1)/self.num_vals)
            else:
                num_nodes = self.num_nodes
            predictions = predictions.view(-1, self.num_vals).argmax(dim=1).view(-1, num_nodes)
            if self.record_inf:
                return predictions, inf_scores
            else:
                return predictions
            
    def save_unary(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.unary_model.state_dict()]
            torch.save(result, fout)

    def load_unary(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.unary_model.load_state_dict(params[0])

    def save_unary_t(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.unary_t_model.unary_t_model.state_dict()]
            torch.save(result, fout)

    def load_unary_t(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.unary_t_model.unary_t_model.load_state_dict(params[0])