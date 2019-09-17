import torch
from torch import nn
import numpy as np
from gspen.fastmp import fastmp
try:
    from gspen.fastmp import ilpinf
    has_ilpinf = True
except:
    has_ilpinf = False


class GSPENModel(nn.Module):
    def __init__(self, unary_model, pair_model, t_model, num_nodes, pairs, num_vals, params):
        super(GSPENModel, self).__init__()
        self.unary_model = unary_model
        self.pair_model = pair_model
        self.combined_pots = self.pair_model is None
        self.t_model = t_model
        self.num_nodes = num_nodes
        self.pairs = pairs
        self.num_vals = num_vals
        self.num_unary = self.num_nodes*self.num_vals
        self.num_pair = len(pairs)*num_vals*num_vals
        self.num_inf_itrs = params.get('num_inf_itrs', 100)
        self.inf_lr = params.get('inf_lr', 0.05)
        self.use_sqrt_decay = params.get('use_sqrt_decay', False)
        self.use_linear_decay = params.get('use_linear_decay', False)
        self.ignore_unary_dropout = params.get('ignore_unary_dropout', False)
        self.use_relu = params.get('use_relu', False)
        self.inf_method = params.get('inf_method')
        self.gpu = params.get('gpu', False)
        self.use_loss_aug = params.get('use_loss_aug', False)
        self.use_recall_loss_aug = params.get('use_recall_loss_aug', False)
        self.inf_mode = params.get('inf_mode')
        self.mp_eps = params.get('mp_eps', 0.)
        self.mp_itrs = params.get('mp_itrs')
        self.gt_belief_interpolate = params.get('gt_belief_interpolate', -1)
        self.only_h_pair = params.get('only_h_pair', False)
        self.only_h_pair_unary = params.get('only_h_pair_unary', False)
        self.use_t_input = params.get('use_t_input', False)
        self.inf_eps = params.get('inf_eps', None)
        self.inf_region_eps = params.get('inf_region_eps', None)
        self.use_entropy = params.get('use_entropy', False)
        self.inf_loss = params.get('inf_loss', None)
        self.use_log_inf_loss = params.get('use_log_inf_loss', False)
        self.inf_loss_coef = params.get('inf_loss_coef', 1.)
        self.label_sq_coef = params.get('label_sq_coef', None)
        self.entropy_coef = params.get('entropy_coef', 1.)
        self.return_all_vals = params.get('return_all_vals', False)
        self.pot_scaling_factor = params.get('pot_scaling_factor', 1.)
        if self.gpu:
            self.tensor_mod = torch.cuda
            self.unary_model.cuda()
            if not self.combined_pots:
                self.pair_model.cuda()
            self.t_model.cuda()
        else:
            self.tensor_mod = torch
        batch_size = params['batch_size']
        self.num_potentials = num_nodes*num_vals + num_vals*num_vals*len(pairs)


        if self.inf_mode == 'lp':
            if not has_ilpinf:
                raise ValueError('Inference Mode %s is only valid if this library is installed while a valid Gurobi install is present.'%self.inf_mode)
            self.inf_runners = [ilpinf.ILPInf(num_nodes, num_vals, pairs, np.zeros(self.num_potentials, dtype=np.float32), True) for _ in range(batch_size)]
        elif self.inf_mode in ['mp', 'md']:
            if self.inf_mode == 'md':
                if self.mp_eps == 0:
                    self.mp_eps = 1.
            self.inf_runners = [fastmp.FastMP(num_nodes, num_vals, pairs, np.zeros(self.num_potentials, dtype=np.float32)) for _ in range(batch_size)]
            self.num_msgs = self.inf_runners[0].get_num_msgs()
            self.msgs = torch.FloatTensor(batch_size, self.num_msgs)
            for idx, mp_graph in enumerate(self.inf_runners):
                mp_graph.allocate_mp_mem(self.mp_eps)
                mp_graph.update_msgs(self.msgs[idx, :].numpy())
        else:
            raise Exception("Inf mode not recognized: ",self.inf_mode)

        self.beliefs = torch.FloatTensor(batch_size, self.num_potentials).fill_(0.)
        if self.gpu:
            self.beliefs.pin_memory()
        for idx, inf_runner in enumerate(self.inf_runners):
            inf_runner.update_beliefs_pointer(self.beliefs[idx, :].numpy())

    def get_optimizer(self, params):
        unary_lr = params['unary_lr']
        pair_lr = params['pair_lr']
        lr = params.get('lr', None)
        t_lr = params.get('t_lr', None)
        t_unary_lr = params.get('t_unary_lr', None)
        t_pair_lr = params.get('t_pair_lr', None)
        t_wd = params.get('t_wd', 0.)
        t_unary_wd = params.get('t_unary_wd', 0.)
        t_pair_wd = params.get('t_pair_wd', 0.)
        unary_wd = params.get('unary_wd', 0.)
        pair_wd = params.get('pair_wd', 0.)
        combined_wd = params.get('combined_wd', 0.)
        unary_mom = params.get('unary_mom', 0)
        pair_mom = params.get('pair_mom', 0)
        unary_t_mom = params.get('t_unary_mom', 0)
        t_mom = params.get('t_mom', 0)
        if self.combined_pots:
            param_groups = [{'params':self.unary_model.parameters(), 'lr':combined_lr, 'weight_decay':combined_wd}]
        else:
            param_groups = [
                    {'params':self.unary_model.parameters(), 'lr':unary_lr, 'weight_decay':unary_wd, 'momentum':unary_mom},
                    {'params':self.pair_model.parameters(), 'lr':pair_lr, 'weight_decay':pair_wd, 'momentum':pair_mom}]
        if t_lr is not None:
            param_groups.append({'params':self.t_model.parameters(), 'lr':t_lr, 'weight_decay':t_wd, 'momentum':t_mom})
        else:
            if t_unary_lr is not None:
                param_groups.append({'params':self.t_model.unary_t_model.parameters(), 'lr':t_unary_lr, 'weight_decay':t_unary_wd, 'momentum':unary_t_mom})
            if t_pair_lr is not None:
                param_groups.append({'params':self.t_model.pair_t_model.parameters(), 'lr':t_pair_lr, 'weight_decay':t_pair_wd})
        if params.get('use_adam', False):
            opt = torch.optim.Adam(param_groups)
        else:
            opt = torch.optim.SGD(param_groups)
        return opt

    def get_random_beliefs(self, num_samples, nodes, pairs):
        all_probs = []
        if nodes is None:
            beliefs = torch.FloatTensor(self.num_potentials)
            for _ in range(self.num_nodes):
                vals = [np.random.rand() for i in range(self.num_vals-1)]
                vals.sort()
                vals.append(1)
                vals.insert(0, 0)
                probs = [vals[i+1] - vals[i] for i in range(self.num_vals)]
                probabilities = torch.FloatTensor(probs)
                all_probs.append(probabilities)
            beliefs[:self.num_nodes*self.num_vals] = torch.cat(all_probs)
            offset = self.num_nodes*self.num_vals
            for pair in self.pairs:
                n1, n2 = pair
                bels1 = beliefs[n1*self.num_vals:(n1+1)*self.num_vals]
                bels2 = beliefs[n2*self.num_vals:(n2+1)*self.num_vals]
                beliefs[offset:offset+self.num_vals*self.num_vals] = torch.ger(bels2, bels1).view(-1) #This is the outer product function
                offset += self.num_vals*self.num_vals
            beliefs = beliefs.expand(num_samples, self.num_potentials).contiguous()
        else:
            #TODO: need to do similar to above, but for each individual data point
            pass
        if self.gpu:
            beliefs = beliefs.cuda(async=True)
        return beliefs

    def _run_inf_runners(self, inf_runners):
        if self.inf_mode == 'lp':
            ilpinf.runilp(inf_runners)
        elif self.inf_mode in ['mp', 'md']:
            self.msgs[:len(inf_runners), :].fill_(0.)
            fastmp.runmp(inf_runners, self.mp_itrs, self.mp_eps)
            fastmp.update_beliefs(inf_runners, self.mp_eps)

    def _find_predictions(self, is_test, epoch, inputs, pots, lossaug, labels, belief_masks=None, nodes=None, pairs=None, init_predictions=None, msgs=None, log_callback=None):
        if self.inf_mode == 'md':
            return self._md_inf(is_test, inputs, pots, lossaug, labels, epoch, belief_masks, nodes, pairs)
        else:
            return self._fw_inf(is_test, inputs, pots, lossaug, labels, epoch, nodes, pairs, log_callback=log_callback)

    def calculate_pots(self, inp, belief_labels):
        if self.combined_pots:
            return self.pot_scaling_factor*self.unary_model(inp)
        else:
            return self.pot_scaling_factor*torch.cat([self.unary_model(inp), self.pair_model(inp)], dim=1)

    def _call_t(self, predictions, pots, inputs):
        if self.use_t_input:
            return self.t_model(predictions, pots, inputs)
        else:
            return self.t_model(predictions, pots)
    
    def _md_inf(self, is_test, inputs, pots, lossaug, labels, epoch, belief_masks, nodes, pairs):
        lr = self.inf_lr
        if self.use_sqrt_decay or self.use_linear_decay:
            start_lr = lr
        prediction = self.get_random_beliefs(len(inputs), nodes, pairs)
        pots = pots.detach()
        pots.requires_grad=False
        if belief_masks is not None:
            if pots.size(1) != belief_masks.size(1):
                pots = torch.cat([pots, self.tensor_mod.FloatTensor(pots.size(0), belief_masks.size(1) - pots.size(1))], dim=1)
            pots = pots * belief_masks
            prediction = prediction*belief_masks
            #TODO: initialize inf runners for this batch
        else:
            inf_runners = self.inf_runners[:pots.size(0)]
        prev_obj = self.tensor_mod.FloatTensor(inputs.size(0)).fill_(-float('inf'))
        region_prediction_eps = float('inf')
        prev_unary_pred = prediction[:, :self.num_vals*self.num_nodes].contiguous()
        prev_pair_pred = prediction[:, self.num_vals*self.num_nodes:].contiguous()
        for itr in range(self.num_inf_itrs):
            self.t_model.zero_grad()
            pred_var = torch.autograd.Variable(prediction, requires_grad=True)
            vector_obj = self._call_t(pred_var, pots, inputs)
            if lossaug is not None:
                vector_obj = vector_obj + (pred_var*lossaug).sum(dim=1)
            if self.use_entropy:
                vector_obj = vector_obj - self.entropy_coef*(pred_var*torch.log(pred_var+1e-6)).sum(dim=1)
            eps = torch.abs(vector_obj - prev_obj).max().item()
            if self.inf_eps is not None and eps < self.inf_eps:
                break
            prev_obj = vector_obj.detach()
            obj = vector_obj.sum()
            obj.backward()
            if self.use_sqrt_decay:
                lr = self.inf_lr/np.sqrt(itr+1)
            elif self.use_linear_decay:
                lr = self.inf_lr/(itr+1)
            if self.gpu:
                lr = torch.cuda.FloatTensor([lr])
            grad = pred_var.grad
            inf_coef = (1 + torch.log(prediction+1e-6) + lr*grad)
            inf_coef = inf_coef.cpu()
            for inf_runner_ind, inf_runner in enumerate(inf_runners):
                graph_coef = inf_coef[inf_runner_ind, :].numpy()
                inf_runner.update_potentials(graph_coef)
            self._run_inf_runners(inf_runners)
            prediction = self.beliefs[:pots.size(0), :]
            if self.gpu:
                prediction = prediction.cuda(async=True)
            if self.inf_region_eps is not None:
                new_unary_pred = prediction[:, :self.num_vals*self.num_nodes].contiguous()
                new_pair_pred = prediction[:, self.num_vals*self.num_nodes:].contiguous()
                unary_prediction_eps = torch.norm(prev_unary_pred.view(-1, self.num_vals) - new_unary_pred.view(-1, self.num_vals), dim=1).max().item()
                pair_prediction_eps = torch.norm(prev_pair_pred.view(-1, self.num_vals*self.num_vals) - new_pair_pred.view(-1, self.num_vals*self.num_vals), dim=1).max().item()
                if unary_prediction_eps < self.inf_region_eps and pair_prediction_eps < self.inf_region_eps:
                    break
                prev_unary_pred = new_unary_pred
                prev_pair_pred = new_pair_pred

        return prediction, itr

    def _fw_inf(self, is_test, inputs, pots, lossaug, labels, epoch, nodes, pairs, log_callback=None):
        lr = self.inf_lr
        if self.use_sqrt_decay or self.use_linear_decay:
            start_lr = lr

        prediction = self.get_random_beliefs(len(inputs), nodes, pairs)

        pots = pots.detach()
        pots.requires_grad=False
        inf_runners = self.inf_runners[:pots.size(0)]
        if self.inf_region_eps is not None:
            prev_pred = prediction.clone()

        self.t_model.zero_grad()
        pred_var = torch.autograd.Variable(prediction, requires_grad=True)
        obj = self._call_t(pred_var, pots, inputs)
        if lossaug is not None:
            obj = obj + (lossaug*pred_var).sum(dim=1)
        if self.use_entropy:
            obj = obj - self.entropy_coef*(pred_var*torch.log(pred_var+1e-6)).sum(dim=1)
        prev_obj = obj.data
        breaking=False
        for itr in range(self.num_inf_itrs):
            if log_callback is not None:
                log_callback(itr, obj[0].item())
            obj = obj.sum()
            obj.backward()
            grad = pred_var.grad
            for inf_runner_ind, inf_runner in enumerate(inf_runners):
                graph_grad = grad[inf_runner_ind, :].data.cpu().numpy()
                inf_runner.update_potentials(graph_grad)
            
            self._run_inf_runners(inf_runners)

            if self.gpu:
                max_p = self.beliefs[:inputs.size(0),:].cuda(async=True)
            else:
                max_p = self.beliefs[:inputs.size(0),:]
            if self.use_sqrt_decay:
                lr = self.inf_lr/np.sqrt(itr+1)
            elif self.use_linear_decay:
                lr = self.inf_lr/(itr+1)
            if self.gpu:
                lr = torch.cuda.FloatTensor([lr])
            prediction = prediction + lr*(max_p - prediction)

            self.t_model.zero_grad()
            pred_var = torch.autograd.Variable(prediction, requires_grad=True)
            obj = self._call_t(pred_var, pots, inputs)
            if lossaug is not None:
                obj = obj + (lossaug*pred_var).sum(dim=1)
            if self.use_entropy:
                obj = obj - self.entropy_coef*(pred_var*torch.log(pred_var+1e-6)).sum(dim=1)
            eps = torch.abs(obj.data - prev_obj).max().item()
            if self.inf_eps is not None and eps < self.inf_eps:
                breaking=True
                break
            if self.inf_region_eps is not None:
                region_eps = torch.abs(prev_pred - prediction).max().item()
                if region_eps < self.inf_region_eps:
                    break
                prev_pred = prediction.clone()


            prev_obj = obj.data
        return prediction, itr

    def _get_loss_aug(self, belief_labels):
        if self.gpu:
            loss_aug = torch.cuda.FloatTensor(belief_labels.size(0), belief_labels.size(1)).fill_(0.)
        else:
            loss_aug = torch.FloatTensor(belief_labels.size(0), belief_labels.size(1)).fill_(0.)
        num_unary = self.num_nodes*self.num_vals
        loss_aug[:, :num_unary] = 1-belief_labels[:, :num_unary]
        return loss_aug

    def calculate_obj(self, epoch, inputs, belief_labels, belief_masks=None, nodes=None, pairs=None, init_predictions=None, messages=None):
        if self.ignore_unary_dropout:
            self.unary_model.eval()
        else:
            self.unary_model.train()
        if not self.combined_pots:
            self.pair_model.train()
        self.t_model.eval()
        if self.use_loss_aug:
            lossaug = self._get_loss_aug(belief_labels)
        elif self.use_recall_loss_aug:
            lossaug = self._get_recall_loss_aug(belief_labels)
        else:
            lossaug = None
        pots = self.calculate_pots(inputs, belief_labels)
        predictions, num_iters = self._find_predictions(False, epoch, inputs, pots, lossaug, belief_labels, belief_masks, nodes, pairs, init_predictions=init_predictions, msgs=messages)
        self.t_model.zero_grad()
        self.unary_model.zero_grad()
        self.unary_model.train()
        if not self.combined_pots:
            self.pair_model.zero_grad()
            self.pair_model.train()
        self.t_model.train()
        inf_obj = self._call_t(predictions, pots, inputs)
        if lossaug is not None:
            inf_obj = inf_obj + (predictions*lossaug).sum(dim=1)
        if self.use_entropy:
            inf_obj = inf_obj - self.entropy_coef*(predictions*torch.log(predictions+1e-6)).sum(dim=1)
        label_term = self._call_t(belief_labels, pots, inputs)
        obj = inf_obj - label_term
        if self.use_relu:
            obj = nn.ReLU()(obj)
        obj = obj.mean()
            
        if self.return_all_vals:
            return obj, inf_obj.mean(), label_term.mean(), num_iters
        else:
            return obj, inf_obj.sum()/pots.size(0)

    def calculate_beliefs(self, inputs, labels=None):
        self.unary_model.eval()
        if not self.combined_pots:
            self.pair_model.eval()
        self.t_model.eval()
        pots = self.calculate_pots(inputs, labels)
        result, _ = self._find_predictions(True, None, inputs, pots, None, labels)
        return result

    def predict(self, inputs, labels=None, log_callback=None):
        self.unary_model.eval()
        if not self.combined_pots:
            self.pair_model.eval()
        self.t_model.eval()
        pots = self.calculate_pots(inputs, labels)
        return self._find_predictions(True, None, inputs, pots, None, labels, log_callback=log_callback)[0][:,:self.num_nodes*self.num_vals].contiguous().view(-1, self.num_vals).argmax(dim=1).view(-1, self.num_nodes)

    def save_unary(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.unary_model.state_dict()]
            torch.save(result, fout)

    def load_unary(self, file_path):
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.unary_model.load_state_dict(params[0])

    def save_pair(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.pair_model.state_dict()]
            torch.save(result, fout)

    def load_pair(self, file_path):
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.pair_model.load_state_dict(params[0])

    def save_t(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.t_model.state_dict()]
            torch.save(result, fout)

    def load_t(self, file_path):
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.t_model.load_state_dict(params[0])

    def save_unary_t(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.t_model.unary_t_model.state_dict()]
            torch.save(result, fout)

    def load_unary_t(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.t_model.unary_t_model.load_state_dict(params[0])

    def save_pair_t(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.t_model.pair_t_model.state_dict()]
            torch.save(result, fout)

    def load_pair_t(self, file_path):
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.t_model.pair_t_model.load_state_dict(params[0])