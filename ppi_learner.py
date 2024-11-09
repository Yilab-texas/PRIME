import torch
import random
import os
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import logging
from ppi_dataset import SDNNPPIdataset
from sklearn.model_selection import train_test_split
from tools.metric_recorder import MetricRecorder
# from models import PredModel, DiscrimModel
from losses import EditLoss

# from utils.dscript_utils import collate_paired_sequences, load_hdf5_parallel, collate_paired_sequences_2
from utils.sdnn_utils import collate_sdnn_sequences
# from backbones.dscript.embedding import FullyConnectedEmbed
# from backbones.dscript.contact import ContactCNN
# from backbones.dscript.interaction import ModelInteraction
from backbones.sdnn.sdnn_model import SdnnModel
from sklearn.metrics import average_precision_score as average_precision
from datetime import datetime

# import nni

logging.getLogger().setLevel(logging.INFO)



class PPILearner():
    """learner for ppi prediction"""
    def __init__(self, config, use_nni=False):
        """Constructor function."""
        self.use_nni = use_nni
        self.save_results = config['save_results']
        if self.save_results:
            # trial_ID = 'abcd'
            # self.nni_results_fpath = 'tmp_results/{}.csv'.format(trial_ID)
            # self.nni_results_fpath = os.path.join(config['mdl_dpath'], 'results.csv')
            self.feature_fpath = os.path.join(config['mdl_dpath'], 'features.pt')
            # os.makedirs('tmp_results', exist_ok=True)
        os.makedirs(config['mdl_dpath'], exist_ok=True)
        #initialization
        self.config = config
        self.lr_init = round(config['lr_init'], 10)
        self.pred_layer_num = config['pred_layer_num']
        self.fc_hidden = config['fc_hidden']
        self.class_num = config['class_num']
        self.patch_dpath = config['patch_dpath']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.savemodel = self.config['save_model']
        self.mdl_dpath = self.config['mdl_dpath']
        self.dropout = self.config['dropout']
        self.train_task = config['train_task']
        self.max_dist = config['max_dist']
        self.ori_coeff = config['ori_coeff']
        self.GD_coeff = config['GD_coeff']
        self.edit_coeff = config['edit_coeff']
        self.splited = config['splited']
        self.seed = config['seed']
        
        if 'pair' not in config:
            self.pair = False
        else:
            self.pair = config['pair']
            self.pair_mode = config['pair_mode']
            if self.pair:
                self.discrim = True
                # self.discrim = config['discrim']
                self.discrim_weight = config['discrim_weight']
                self.discrim_dim = 50

        
        self.bn = False

        # not use nni
        self.dropout_p = round(config['dropout_p'],10)
        self.inter_weight = 0.35
        if 'pair_weight' not in config:
            self.pair_weight = 0.05
        else:
            self.pair_weight = config['pair_weight']
        self.max_edit_dist = 2
        zero_label_weight = 1
        one_label_weight = 1
        self.discrim_weight = 1.0

        if config['esm_pool']:
            self.esm_pool = True
        else:
            self.esm_pool = False
        one_label_weight = config['one_label_weight']
        self.use_esm = True
        self.cal_disc = True
        self.ensemble_func = 'disc' # disc, both, norm
        self.CE_weights = torch.tensor([zero_label_weight*1.0, one_label_weight*1.0]).cuda() 
        # self.config['lr_scheduler'] == 'const'


    def eval(self, model_fpath, save_result_fpath, feature_fpath, fold=-1): # fold self.feature_fpath 
        # if fold != -1:
        #     self.nni_results_fpath = os.path.join(self.config['mdl_dpath'], 'case_results_fold{}.csv'.format(fold))
        #     self.feature_fpath = os.path.join(self.config['mdl_dpath'], 'case_features_fold{}.pt'.format(fold))   
        data_loader_tst = self.__sdnn_loader_eval(fold=-1)
        logging.info(f'# of iter in the test subset:{len(data_loader_tst.dataset)}')
        pred_net = self.__build_models(self.train_task)
        pred_net = self.restore_model(pred_net, model_fpath)
        criterion = nn.CrossEntropyLoss()
        epoch = -1
        embeddings = None

        with torch.no_grad():
            tst_acc, tst_loss, metrics, all_results = self.__eval_impl(pred_net, data_loader_tst, criterion, epoch, embeddings, subset= 'tst')


        if self.save_results:
            with open(save_result_fpath, 'w') as w:
                w.write('{},{},{},{},{},{},{}\n'.format('mutation', 'interactor', 'pred_change', 'pred_logic_0', 'pred_logic_1', 'label_change', 'label_before'))  
                n0_list, n1_list, p_guess, y, y_2, p_logic, merged_out_list, merged_out_2_list, esm_merged_list, merged_out_all_list = all_results
                feat_dict = {}
                for n0, n1, p, label, label_2, p_l, feat_1, feat_2, feat_esm, feat_all in zip(n0_list, n1_list, p_guess, y, y_2, p_logic, merged_out_list, merged_out_2_list, esm_merged_list, merged_out_all_list):
                    p = int(p.item())
                    label = int(label)
                    label_2 = int(label_2)
                    p_l_1 = float(p_l[0].cpu().item())
                    p_l_2 = float(p_l[1].cpu().item())

                    w.write('{},{},{},{},{},{},{}\n'.format(n0, n1, p, p_l_1, p_l_2, label, label_2)) 
                    feat_dict['{}-{}'.format(n0, n1)] = (feat_1, feat_2, label, feat_esm, feat_all)
                torch.save(feat_dict, feature_fpath)


        logging.info('tst_acc: {}'.format(tst_acc))

        # with open(self.config['log_fpath'], 'a+') as w:
        #     w.write('config_{}.yaml, validation Loss {:.5}, Acc {:.5}\n'.format(self.config['mdl_dpath'].split('/')[-1], val_loss, val_acc))     
        return tst_acc, metrics   



    def train(self, fold=-1):
        if fold != -1:
            self.nni_results_fpath = os.path.join(self.config['mdl_dpath'], 'results_fold{}.csv'.format(fold))
            self.feature_fpath = os.path.join(self.config['mdl_dpath'], 'features_fold{}.pt'.format(fold))   
            pass

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2
        torch.use_deterministic_algorithms(True)
        #build data loaders from training & validation subsets


        if self.train_task == 'sdnn':
            data_loader_trn, data_loader_val, data_loader_tst = self.__sdnn_loader(fold=fold)
            embeddings = None
        n_iters_per_epoch = len(data_loader_trn)
        logging.info(f'# of iter in the train subset:{len(data_loader_trn.dataset)}')
        logging.info(f'# of iter in the valid subset:{len(data_loader_val.dataset)}')
        logging.info(f'# of iter in the test subset:{len(data_loader_tst.dataset)}')
        
        pred_net = self.__build_models(self.train_task)
        
        optimizer, scheduler = self.__build_optimizer(pred_net)
        criterion = nn.CrossEntropyLoss()
        
        # pred_net, optimizer, scheduler, idx_epoch_last = \
        #     self.__restore_snapshot(pred_net, optimizer, scheduler)

        acc_opt, loss_opt = None, None
        tst_acc_opt, tst_loss_opt = None, None
        trn_acc_opt, tst_loss_opt = None, None
        best_epoch_idx = 0
        
        if fold != -1:
            pth_fpath_pred_opt = os.path.join(self.mdl_dpath, 'model_pred_opt_fold_{}.pth'.format(fold))
        else:
            pth_fpath_pred_opt = os.path.join(self.mdl_dpath, 'model_pred_opt.pth')
        all_results_best = None
        metrics_best = None
        # if idx_epoch_last != -1:
        #     idx_iter = n_iters_per_epoch * (idx_epoch_last + 1)
        #     acc_opt, loss_opt = self.__eval_impl(pred_net, data_loader_val, criterion, idx_epoch_last, subset = 'val')
        
        
        #update the model through multiple epochs
        for epoch in range(0, self.config['n_epochs']):
            lrn_rate = self.lr_init if scheduler is None else scheduler.get_last_lr()[0]
            logging.info(f'starting the {epoch+1}-th training epoch (LR: {lrn_rate})')
            

            self.__train_impl(pred_net, data_loader_trn, optimizer, criterion, epoch, embeddings)

            with torch.no_grad():
                val_acc, val_loss, _, _ = self.__eval_impl(pred_net, data_loader_val, criterion, epoch, embeddings, subset = 'val')

            

            if self.config['opt_rule'] == 'loss':
                if loss_opt is None or loss_opt > val_loss:
                    loss_opt = val_loss
                    if self.savemodel:
                        self.save_model(pred_net, pth_fpath_pred_opt)
            elif self.config['opt_rule'] == 'acc':
                if acc_opt is None or acc_opt <= val_acc: # gai
                    with torch.no_grad():
                        tst_acc, tst_loss, metrics, all_results = self.__eval_impl(pred_net, data_loader_tst, criterion, epoch, embeddings, subset= 'tst')

                    acc_opt = val_acc
                    tst_acc_opt = tst_acc
                    best_epoch_idx = epoch + 1
                    if self.savemodel:
                        self.save_model(pred_net, pth_fpath_pred_opt)
                    results_log = {'best_epoch':best_epoch_idx, 'val_acc_opt': acc_opt, 'tst_acc_opt': tst_acc_opt}
                    all_results_best = all_results
                    metrics_best = metrics
                    metrics_best['val_acc'] = acc_opt
 
            if scheduler:
                scheduler.step()
            #save the snapshot for fast recovery
            # self.__save_snapshot(pred_net, optimizer, scheduler, epoch)

        # nni.report_final_result(float(tst_acc_opt))
        if self.use_nni:
            with open(self.nni_log_path, 'a+') as w:
                results_log = {'best_epoch':best_epoch_idx, 'val_acc_opt': acc_opt, 'tst_acc_opt': tst_acc_opt}
                w.write('trial_id: {}, ep: {}, val_acc {:.5}, '.format(self.trial_id, results_log['best_epoch'], results_log['val_acc_opt']))
                for key in metrics_best:
                    w.write('{}: {:.5} '.format(key, metrics_best[key]))
                w.write('\n')
                # w.write('trial_id: {}, val_acc_opt {:.5}, tst_acc_opt {:.5}, best_epoch {}\n'.format(self.trial_id, results_log['val_acc_opt'], results_log['tst_acc_opt'], results_log['best_epoch'])) 
            with open(self.nni_results_fpath, 'w') as w:
                n0_list, n1_list, p_guess, y, y_2 = all_results_best
                for n0, n1, p, label, label_2 in zip(n0_list, n1_list, p_guess, y, y_2):
                    p = int(p.item())
                    label = int(label)
                    label_2 = int(label_2)
                    w.write('{},{},{},{},{}\n'.format(n0, n1, p, label, label_2))  
                # print(n0, n1, p, label)            
            with open(self.nni_parameter_fpath, 'w') as w:
                for parameter_key in self.params:
                    w.write('{}: {}\n'.format(parameter_key, self.params[parameter_key]))
        else:
            with open(self.config['log_fpath'], 'a+') as w:
                if fold == -1:
                    results_log = {'seed':self.seed, 'best_epoch':best_epoch_idx, 'val_acc_opt': acc_opt, 'tst_acc_opt': tst_acc_opt}
                else:
                    results_log = {'seed':'{}_{}'.format(self.seed, fold), 'best_epoch':best_epoch_idx, 'val_acc_opt': acc_opt, 'tst_acc_opt': tst_acc_opt}
                w.write('seed: {}, ep: {}, val_acc {:.5}, '.format(results_log['seed'], results_log['best_epoch'], results_log['val_acc_opt']))
                for key in metrics_best:
                    w.write('{}: {:.5} '.format(key, metrics_best[key]))
                w.write(',{}'.format(self.mdl_dpath))
                w.write('\n')

        if self.save_results:
            with open(self.nni_results_fpath, 'w') as w:
                w.write('{},{},{},{},{},{},{}\n'.format('mutation', 'interactor', 'pred_change', 'pred_logic_0', 'pred_logic_1', 'label_change', 'label_before'))  
                n0_list, n1_list, p_guess, y, y_2, p_logic, merged_out_list, merged_out_2_list, esm_merged_list, merged_out_all_list = all_results_best
                feat_dict = {}
                for n0, n1, p, label, label_2, p_l, feat_1, feat_2, feat_esm, feat_all in zip(n0_list, n1_list, p_guess, y, y_2, p_logic, merged_out_list, merged_out_2_list, esm_merged_list, merged_out_all_list):
                    p = int(p.item())
                    label = int(label)
                    label_2 = int(label_2)
                    p_l_1 = float(p_l[0].cpu().item())
                    p_l_2 = float(p_l[1].cpu().item())

                    w.write('{},{},{},{},{},{},{}\n'.format(n0, n1, p, p_l_1, p_l_2, label, label_2)) 
                    feat_dict['{}-{}'.format(n0, n1)] = (feat_1, feat_2, label, feat_esm, feat_all)
                torch.save(feat_dict, self.feature_fpath)


        logging.info('Seed: {}, Epoch_opt: {}, val_opt_acc: {}, tst_opt_acc: {}'.format(self.seed, best_epoch_idx, acc_opt, tst_acc_opt))
        print(metrics_best)
        # with open(self.config['log_fpath'], 'a+') as w:
        #     w.write('config_{}.yaml, validation Loss {:.5}, Acc {:.5}\n'.format(self.config['mdl_dpath'].split('/')[-1], val_loss, val_acc))     
        return tst_acc_opt, results_log, metrics_best                 
                
    def __eval_impl(self, pred_net, data_loader, criterion, idx_epoch, embeddings, subset):
        """Evaluate the model - core implementation"""
        
        #eval the model
        pred_net.eval()
        
        recorder = MetricRecorder()
        n_iters_per_epoch = len(data_loader)
        for idx_iter, (inputs) in enumerate(data_loader):
            # loss, metrics = self.__forward_pair_pred(inputs, pred_net, criterion, subset = subset)

            if self.train_task == 'sdnn':
                metrics, all_results = self.__forward_sdnn_eval(pred_net, data_loader)

            if self.train_task == 'dscript' or self.train_task == 'sdnn' or self.train_task == 'deepfe' or self.train_task == 'pipr':
               metrics = metrics
               break
            else:
                #recore evaluation metrics
                recorder.add(metrics)
                if (idx_iter + 1) % self.config['n_iters_rep'] != 0:
                    continue
                #report evaluation metrics periodically
                ratio = (idx_iter + 1) / n_iters_per_epoch
                recorder.display('Ep. #%d - %.2f%% (%s): ' % (idx_epoch + 1, 100.0 * ratio, subset))

        if self.train_task == 'dscript' or self.train_task == 'sdnn' or self.train_task == 'deepfe' or self.train_task == 'pipr':
            acc = metrics['acc']
            loss = metrics['loss']
            # mse = metrics['mse']
            pr = metrics['pr']
            re = metrics['re']
            f1 = metrics['f1']
            aupr = metrics['aupr']

            # acc = metrics[1]/(n_iters_per_epoch * self.config['batch_size_val'])
            logging.info(f"{subset} Epoch {idx_epoch + 1}: Loss={loss:.6}, Accuracy={acc:.3%}, Precision={pr:.6}, Recall={re:.6}, F1={f1:.6}, AUPR={aupr:.6}")
            return acc, loss, metrics, all_results
        else:
            #show final evaluation metrics at the end of epoch
            recorder.display('Ep. #%d - Final (%s): ' % (idx_epoch + 1, subset))
        
            return recorder.get()['Acc'], recorder.get()['Loss']

    def __forward_cls_pred(self, inputs, pred_net, criterion, subset):
        """Perform the forward pass with train inputs"""
        
        inputs['gene'] = inputs['gene'].to(self.device)
        inputs['interactor'] = inputs['interactor'].to(self.device)
        inputs['label'] = inputs['label'].to(self.device)
        
        pred = pred_net(torch.cat((inputs['gene'], inputs['interactor']), 1))

        loss, metrics = self.__calc_loss_impl(inputs['label'], pred, criterion, subset = subset)
        
        return loss, metrics

    def __forward_cont_pred(self, inputs, pred_net, criterion, subset):
        """Perform the forward pass with train inputs"""
        
        inputs['gene'] = inputs['gene'].to(self.device)
        inputs['ori_gene'] = inputs['ori_gene'].to(self.device)
        inputs['interactor'] = inputs['interactor'].to(self.device)
        inputs['label'] = inputs['label'].to(self.device)
        inputs['ori_label'] = inputs['ori_label'].to(self.device)
        inputs['change_label'] = inputs['change_label'].to(self.device)
        
        # pred = pred_net(torch.cat((inputs['gene'], inputs['interactor']), 1))
        outputs = pred_net(inputs['ori_gene'], inputs['gene'], inputs['interactor'])

        loss, metrics = self.__cont_loss_impl(inputs['label'], inputs['ori_label'], inputs['change_label'], outputs, criterion, subset = subset)
        
        return loss, metrics

    def __predict_cmap_interaction(self, model, n0, n1, tensors, use_cuda):
        """
        Predict whether a list of protein pairs will interact, as well as their contact map.

        :param model: Model to be trained
        :type model: dscript.models.interaction.ModelInteraction
        :param n0: First protein names
        :type n0: list[str]
        :param n1: Second protein names
        :type n1: list[str]
        :param tensors: Dictionary of protein names to embeddings
        :type tensors: dict[str, torch.Tensor]
        :param use_cuda: Whether to use GPU
        :type use_cuda: bool
        """
        # b batch_zie
        b = len(n0)

        p_hat = []
        c_map_mag = []
        cm_list = []
        B_list = []
        for i in range(b):
            z_a = tensors[n0[i]]
            z_b = tensors[n1[i]]
            if use_cuda:
                z_a = z_a.cuda()
                z_b = z_b.cuda()
            # z_a shape 1,52,6165
            cm, ph, B = model.map_predict(z_a, z_b)
            B_list.append(B)
            # cm shape ([1, 1, 52, 485]) , ph : 0.0015
            p_hat.append(ph)
            c_map_mag.append(torch.mean(cm))
            cm_list.append(cm)
        p_hat = torch.stack(p_hat, 0)
        c_map_mag = torch.stack(c_map_mag, 0)
        return c_map_mag, p_hat, cm_list, B_list
    
    def __forward_sdnn_pred(self, inputs, model):
        if not self.pair:
            n0, n1, y, _, _, _, _, _, _, _, esm_feat = inputs
            
        else:
            # n0, n1, y, n0_2, n1_2, y_2, same, mute_id, n0_id, n1_id, esm_feat = inputs
            n0, n1, y, n0_2, n1_2, y_2, same, esm_feat = inputs
            y_2 = y_2.cuda()
        
        y = y.cuda()
        pred, merged_out = model(n0.cuda(), n1.cuda())
        
        if self.pair:
            pred_2, merged_out_2 = model(n0_2.cuda(), n1_2.cuda())
            if self.discrim:
                if self.use_esm:
                    pred_change, esm_merged, merged_out_all = model.discriminator_esm(merged_out, merged_out_2, esm_feat.cuda())
                else:
                    pred_change = model.discriminator(merged_out, merged_out_2)
                discrim_loss = self.__discrim_loss(pred_change, y, y_2)
            pair_loss = self.__sdnn_pair_loss(merged_out, merged_out_2, same, y, y_2)
            count_not_same = torch.sum(1 - same.int())

            pair_loss = pair_loss / count_not_same

        loss_fn = nn.CrossEntropyLoss(weight=self.CE_weights)
        label = y.long()
        bce_loss = loss_fn(pred.cuda(), label.cuda())

        accuracy_loss = bce_loss
        if self.pair: 
            loss = accuracy_loss + self.pair_weight * pair_loss
            if self.discrim:
                loss += self.discrim_weight * discrim_loss.cpu()
        b = len(pred)

        with torch.no_grad():
            pred = pred.cpu()
            y = y.cpu()
            _, predicted = torch.max(pred.data, 1)
            correct = (predicted == y).sum().item()
        metrics = {'Loss': loss.item(),
                   'Acc': correct/b,
                #    'mse': mse
                    }
        # return loss, correct, mse, b
        return loss, metrics
        

    
    def __discrim_loss(self, pred_change, y, y_2):
        loss_fn = nn.CrossEntropyLoss()
        cmp = torch.eq(y, y_2)
        label = torch.logical_not(cmp).int().long()
        loss = loss_fn(pred_change.cuda(), label.cuda())
        return loss

    def __sdnn_pair_loss(self, merged_out, merged_out_2, same, y, y_2):
        ed_loss = EditLoss(func='sum')
        dist_norm_list = []
        for cm, cm_2, sa, y_fst, y_sec in zip(merged_out, merged_out_2, same, y, y_2):
            if not sa:
                if y_fst == y_sec:
                    edit_dist = 1
                else:
                    edit_dist = self.max_edit_dist
                cm = cm.squeeze()
                cm_2 = cm_2.squeeze()

                L2_dist = torch.nn.functional.pairwise_distance(cm, cm_2)
                # rep_diff = torch.abs(cm - cm_2)
                rep_diff = L2_dist
                dist_norm = torch.mean(rep_diff) / edit_dist
                dist_norm_list.append(dist_norm.unsqueeze(0))
        dist_norm = torch.cat(dist_norm_list, dim=0)
        edit_loss = ed_loss(dist_norm)
        return edit_loss

    def __predict_interaction(self, model, n0, n1, tensors, use_cuda):
        _, p_hat, cm_list, B_list = self.__predict_cmap_interaction(model, n0, n1, tensors, use_cuda)
        return p_hat

    def __forward_sdnn_eval(self, model, test_iterator):
        p_hat = []
        p_logic = []
        pred_change_prob = []
        true_y = []
        true_y_2 = []
        n0_list = []
        n1_list = []
        merged_out_list = []
        merged_out_2_list = []
        esm_merged_list, merged_out_all_list = [], []
        total_loss = 0
        for inputs in test_iterator:

            n0, n1, y, n0_2, n1_2, y_2, _, _, n0_id, n1_id, esm_feat = inputs
            pred, merged_out = model(n0.cuda(), n1.cuda())
            label = y.long()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred.cuda(), label.cuda()).item()
            total_loss += loss
            if self.pair:
                pred_2, merged_out_2 = model(n0_2.cuda(), n1_2.cuda()) # merged_out_2 = [bs, dim]
                if self.discrim:
                    if self.use_esm:
                        pred_change, esm_merged, merged_out_all = model.discriminator_esm(merged_out, merged_out_2, esm_feat.cuda())
                    else:
                        pred_change = model.discriminator(merged_out, merged_out_2)
                    # if self.ensemble_func == 'both':
                merged_out_2_list.append(merged_out_2.cpu())
                esm_merged_list.append(esm_merged.cpu())
                merged_out_all_list.append(merged_out_all.cpu())
            merged_out_list.append(merged_out.cpu())
            # if self.
            # cmp = torch.eq(y, y_2)
            # label = torch.logical_not(cmp).int().long()  
            if self.ensemble_func == 'disc':
                _, predicted = torch.max(pred_change.data, 1)
            elif self.ensemble_func == 'both':
                # print(pred_change.data.shape)
                # print(pred.data.shape)
                _, predicted = torch.max((pred_change.data-pred.data), 1) # todo
                # input('debug')
            elif self.ensemble_func == 'norm':
                _, predicted = torch.max(pred.data, 1)
            p_logic.append(pred_change.data)
            pred_change_prob.append(torch.softmax(pred_change.data.cpu(), dim=1))
            p_hat.append(predicted)
            true_y.append(y)
            true_y_2.append(y_2)
            n0_list += n0_id
            n1_list += n1_id

        merged_out_list = torch.cat(merged_out_list, dim=0)
        if self.pair:
            merged_out_2_list = torch.cat(merged_out_2_list, dim=0)
            esm_merged_list = torch.cat(esm_merged_list, dim=0)
            merged_out_all_list = torch.cat(merged_out_all_list, dim=0)
        p_logic = torch.cat(p_logic, 0)
        pred_change_prob = torch.cat(pred_change_prob, 0)
        

        y = torch.cat(true_y, 0)
        y_2 = torch.cat(true_y_2, 0)
        p_hat = torch.cat(p_hat, 0)
                
        y = y.cuda()
        
        # y = y.unsqueeze(1).cuda()

        with torch.no_grad():
            p_hat = p_hat.data.detach().cpu()
            y = y.detach().cpu()
            if self.cal_disc:
                cmp = torch.eq(y, y_2)
                label = torch.logical_not(cmp).int().long()
                y = label
            else:
                # gai: inverted_tensor
                inverted_y = 1 - y
                inverted_p_hat = 1 - p_hat

                # ppi baseline only
                y = inverted_y
                p_hat = inverted_p_hat


            correct = (p_hat == y).sum().item()
            accuracy = correct / len(p_hat)


            # mse = torch.mean((y.float() - pred) ** 2).item()

            tp = torch.sum(y * p_hat).item()
            fp = torch.sum((p_hat == 1) & (y == 0)).float()
            fn = torch.sum((p_hat == 0) & (y == 1)).float()
            pr = tp / (tp + fp + 1e-8)
            re = tp / (tp + fn + 1e-8)
            f1 = 2 * pr * re / (pr + re + 1e-8)

        y = y.numpy()
        y_2 = y_2.numpy()
        p_hat = p_hat.numpy()
        pred_change_prob = pred_change_prob.numpy()

        aupr = average_precision(y, pred_change_prob[:,1])
        metrics = {
            'loss': total_loss/len(test_iterator),
            'acc': accuracy,
            # 'mse': mse,
            'pr': pr,
            're': re,
            'f1': f1,
            'aupr': aupr,
        }
        all_results = (n0_list, n1_list, p_hat, y, y_2, p_logic, merged_out_list, merged_out_2_list, esm_merged_list, merged_out_all_list)


        return metrics, all_results
    


    def __train_impl(self, pred_net, data_loader, optimizer, criterion, idx_epoch, embeddings=None):
        """train the model - core implementation"""
        # current_lr = optimizer.param_groups[0]['lr']
        # logging.info(f'starting the {idx_epoch+1}-th training epoch (LR: {current_lr})')
        #train the model
        pred_net.train()

        recorder = MetricRecorder()
        n_iters_per_epoch = len(data_loader)
        for idx_iter, (inputs) in enumerate(data_loader):
            if self.train_task == 'sdnn':
                loss, metrics = self.__forward_sdnn_pred(inputs, pred_net)
   
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # record evaluation metrics
            recorder.add(metrics)
            if (idx_iter + 1) % self.config['n_iters_rep'] != 0:
                continue
            
            #report evaluation metrics periodically
            ratio = (idx_iter + 1) / n_iters_per_epoch
            recorder.display('Ep. #%d - %.2f%% (Train): ' % (idx_epoch + 1, 100.0 * ratio))
        
        # show final evaluation metrics at the end of epoch
        recorder.display('Ep. #%d - Final (Train): ' % (idx_epoch + 1))
        # return pred_net, optimizer 
            
    def __calc_loss_impl(self, label, pred_class, criterion, subset):
        """Calculate the loss & evaluation metrics - core implementation"""
        loss = criterion(pred_class, label)
        predicted = torch.max(pred_class, 1)[1]
        correct = (predicted == label).sum()

        # acc = correct / self.config['batch_size_%s' % subset]
        acc = correct / predicted.shape[0]
        
        metrics = {
            'Loss' : loss.item(),
            'Acc'  : acc.item(),
        }
        return loss, metrics

    def __cont_loss_impl(self, label, ori_label, change_label, outputs, criterion, subset):
        """Calculate the loss & evaluation metrics - core implementation"""
        embeds, scores, ori_pred, mute_pred = outputs

        ori_pred_loss = criterion(ori_pred, ori_label)
        mute_pred_loss = criterion(mute_pred, label)

        GD_loss, _, _ = self.__GD_loss(scores)
        edit_dist = torch.tensor([[0, i * self.max_dist + 1] for i in change_label]).to(self.device)

        dist_loss = self.__dist_loss(embeds, edit_dist)
        loss = mute_pred_loss + self.ori_coeff * ori_pred_loss + self.GD_coeff * GD_loss + self.edit_coeff * dist_loss


        mute_predicted = torch.max(mute_pred, 1)[1]
        mute_correct = (mute_predicted == label).sum()
        # if subset == 'tst':
        #     print(mute_predicted)
        # acc = correct / self.config['batch_size_%s' % subset]
        mute_acc = mute_correct / mute_predicted.shape[0]
        

        metrics = {
            'Loss' : loss.item(),
            # 'ori_pred_loss' : self.ori_coeff * ori_pred_loss.item(),
            # 'mute_pred_loss' : mute_pred_loss.item(),
            # 'GD_loss' : self.GD_coeff * GD_loss.item(),
            # 'dist_loss' : self.edit_coeff * dist_loss.item(),
            'Acc'  : mute_acc.item(),
            
        }
        return loss, metrics

    def __GD_loss(self, scores):            
        answer = torch.zeros((scores.size(0),), dtype=torch.long).to(self.device)
        
        pred = scores.argmax(dim=-1)
        correct = (pred == answer).sum().cpu().item()
        total = scores.size(0)

        GD_loss = F.cross_entropy(scores, answer)

        return GD_loss, correct, total

    def __dist_loss(self, embeds, edit_dist):
        # graph_emb : (B, num_candidates, dim)
        rep_diff = (embeds[:,1:] - embeds[:,:1]).norm(dim=-1)

        edit_dist = edit_dist[:, 1:]
        
        dist_norm = rep_diff / edit_dist
        rate = torch.ones((dist_norm.size(0),)).to(self.device)
        
        edit_loss = F.mse_loss(dist_norm, rate)

        return edit_loss

    def __build_optimizer(self, model):
        """Build a optimizer & its learning rate scheduler"""
        
        #create an Adam optimizer
        optimizer = Adam(
            model.parameters(), lr = self.lr_init, weight_decay=self.config['weight_decay'])
        
        #create a LR scheduler
        if self.config['lr_scheduler'] == 'const':
            scheduler = None
        elif self.config['lr_scheduler'] == 'mstep':
            scheduler = MultiStepLR(
                optimizer, milestones=self.config['lr_mlstn'], gamma = self.config['lr_gamma'])
        elif self.config['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, self.config['n_epochs'], eta_min = self.config['lr_min'])
        
        else:
            raise ValueError('unrecognized LR scheduler: ' + self.config['lr_scheduler'])
        return optimizer, scheduler
    
    def __build_models(self, train_task='cls'):
        """build a classification model"""
        config = {}
        num_pred_out = 2560
        # torch.manual_seed(self.seed)


        if train_task == 'sdnn':
            pred_net = SdnnModel(in_features=573, out_features=32, dropout_p=self.dropout_p, use_esm=self.use_esm).to(self.device)
        # elif

        logging.info(f'pred_net model initialized: {str(pred_net)}')
        
        return pred_net


    def __sdnn_loader(self, fold=-1):
        embed_fpath = self.patch_dpath
        trn_data = SDNNPPIdataset(folder_name = embed_fpath, csv_fpath=self.config['trn_fpath'], config=self.config, is_train=True, fold=fold, out_all=False, esm_pool=self.esm_pool)
        val_data = SDNNPPIdataset(folder_name = embed_fpath, csv_fpath=self.config['val_fpath'], config=self.config, is_train=False, fold=fold, out_all=True, esm_pool=self.esm_pool)
        tst_data = SDNNPPIdataset(folder_name = embed_fpath, csv_fpath=self.config['tst_fpath'], config=self.config, is_train=False, fold=fold, out_all=True, esm_pool=self.esm_pool)
        collate_fn = collate_sdnn_sequences
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)


        trn_loader = DataLoader(
            trn_data,
            batch_size=self.config['batch_size_trn'],
            shuffle=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=generator
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config['batch_size_val'],
            shuffle=False,
            collate_fn=collate_fn,
        )
        tst_loader = DataLoader(
            tst_data,
            batch_size=self.config['batch_size_tst'],
            shuffle=False,
            collate_fn=collate_fn,
        )
        return trn_loader, val_loader, tst_loader 

    def __sdnn_loader_eval(self, fold=-1):
        embed_fpath = self.patch_dpath
        tst_data = SDNNPPIdataset(folder_name = embed_fpath, csv_fpath=self.config['tst_fpath'], config=self.config, is_train=False, fold=fold, out_all=True)
        collate_fn = collate_sdnn_sequences
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        tst_loader = DataLoader(
            tst_data,
            batch_size=self.config['batch_size_tst'],
            shuffle=False,
            collate_fn=collate_fn,
        )
        return tst_loader 


    def save_model(cls, model, path):
        """Save the model to a PyTorch checkpoint file."""

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        logging.info('model saved to %s', path)    

    def restore_model(self, model, path):
        """Restore the model from a PyTorch checkpoint file."""

        # find the latest PyTorch checkpoint file, if not provided
        if not os.path.exists(path):
            logging.warning('checkpoint file (%s) does not exist; using the latest model ...', path)
            pth_fnames = [x for x in os.listdir(self.mdl_dpath) if x.endswith('.pth')]
            assert len(pth_fnames) > 0, 'no checkpoint file found under ' + self.mdl_dpath
            path = os.path.join(self.mdl_dpath, sorted(pth_fnames)[-1])

        # restore the model from a PyTorch checkpoint file
        model.load_state_dict(torch.load(path))
        logging.info('model restored from %s', path)

        return model 

    def __save_snapshot(self, pred_net, optimizer, scheduler, idx_epoch):
        """Save base & target models, optimizer, and LR scheduler to a checkpoint file."""

        snapshot = {
            'idx_epoch': idx_epoch,
            'pred_net': pred_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': None if scheduler is None else scheduler.state_dict(),
        }
        pth_fpath = os.path.join(self.mdl_dpath, 'snapshot.pth')
        torch.save(snapshot, pth_fpath)
        logging.info('snapshot saved to %s', pth_fpath)

    def __restore_snapshot(self,  pred_net, optimizer, scheduler):
        """Restore base & target models, optimizer, and LR scheduler from the checkpoint file."""

        pth_fpath = os.path.join(self.mdl_dpath, 'snapshot.pth')
        if not os.path.exists(pth_fpath):
            idx_epoch = -1  # to indicate that no checkpoint file is available
        else:
            snapshot = torch.load(pth_fpath)
            logging.info('snapshot restored from %s', pth_fpath)
            idx_epoch = snapshot['idx_epoch']
            pred_net.load_state_dict(snapshot['pred_net'])
            optimizer.load_state_dict(snapshot['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(snapshot['scheduler'])

        return pred_net, optimizer, scheduler, idx_epoch
