"""Main entry for training and sampling with <EbmLearner>."""
from datetime import datetime
import os
import argparse
import logging
from ppi_learner import PPILearner
import optuna
import torch
import numpy as np

def config_init(path):
    import yaml
    with open(path, 'r') as i_file:
        config = yaml.safe_load(i_file)
    for key, value in config.items():
        logging.info('%s => %s / %s', key, str(value), str(type(value)))
    return config

def main():
    """Main entry."""
    parser = argparse.ArgumentParser(description='run ebm model')
    parser.add_argument('--config_fname', required=True, help='config*.yaml file name')
    parser.add_argument('--fold', type=int, default=-1, help='Fold index for k-fold training')
    parser.add_argument('--use_optuna', action='store_true', default=False, help='use optuna or not')
    parser.add_argument('--data', type=str, default='random', help='how to split the data: random/group')
    parser.add_argument('--optuna_name', type=str, default='example', help='savefpath of the optuna outputs')
    parser.add_argument('--optuna_storage', type=str, default='sqlite:///example.db', help='savefpath of the optuna outputs')
    parser.add_argument('--optuna_trials', type=int, default=100, help='Integer argument')
    parser.add_argument('--optuna_save', action='store_true', default=False, help='save model and results')
    #parser.add_argument('--esm_pool', action='store_true', default=False, help='esm_pool or not')

    parser.add_argument('--esm_pool', action='store_true', default=None, help='Use ESM pooling')


    parser.add_argument('--no_pair_loss', action='store_true', default=False, help='no_pair_loss')

    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite the hyper parameters')
    parser.add_argument('--lr_init', type=float, default=0.003, help='init learn rate')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='dropout p')
    parser.add_argument('--seed', type=int, default=42, help='Integer argument')
    parser.add_argument('--log_fpath', type=str, default='tmp_results/tmp.log', help='savefpath of the results')
    parser.add_argument('--one_label_weight', type=int, default=1, help='one_label_weight')
    parser.add_argument('--discrim_weight', type=float, default=1.0, help='discrim_weight')

    parser.add_argument('--save_model', action='store_true', default=False, help='save_model or not')
    parser.add_argument('--save_results', action='store_true', default=False, help='save_results or not')
    parser.add_argument('--mdl_dpath', type=str, default='tmp_results/exp01', help='savedpath of the models')

    parser.add_argument('--eval', action='store_true', default=False, help='eval mode or not')
    parser.add_argument('--eval_csv', type=str, default='datasets/ppi_case/case_reports_mutation.csv', help='savefpath of the results')
    parser.add_argument('--eval_model_fpath', type=str, default='tmp_results/ODrd01_5fold/exp0/model_pred_opt_fold_0.pth', help='savefpath of the results')
    parser.add_argument('--feat_h5_fpath', type=str, default='datasets/embeddings/sdnn/sdnn_corrected_ppi.h5', help='savefpath of the results')
    parser.add_argument('--save_result_fpath', type=str, default='eval_results/ODrd01_5fold_exp0_fold0.csv', help='savefpath of the results')  
    parser.add_argument('--save_feature_fpath', type=str, default='eval_results/feat_ODrd01_5fold_exp0_fold.pt', help='savefpath of the results')
    parser.add_argument('--load_model_fpath', type=str, default=None, help='path to load pretrained model')
  
    # parser.add_argument('--k_fold', type=int, default=0, help='fold num, if train-valid-test, store 0')
    # parser.add_argument('--optuna_savepath', type=str, default='optuna', help='savedpath of the optuna outputs logs')



    args = parser.parse_args()
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    yml_fpath = os.path.join(curr_dir, args.config_fname)
    config = config_init(yml_fpath)

    if args.eval:
        config['exec_mode'] = 'eval'
        # args.mdl_dpath = args.eval_dpath

    if 'k_fold' not in config:
        config['k_fold'] = 0


    config['data_type'] = 'random'


    if args.no_pair_loss:
        config['pair_weight'] = 0.0

    if args.overwrite:
        config['lr_init'] = args.lr_init
        config['dropout_p'] = args.dropout_p
        config['seed'] = args.seed
        config['log_fpath'] = args.log_fpath
        config['one_label_weight'] = args.one_label_weight
        config['discrim_weight'] = args.discrim_weight

        config['save_model'] = args.save_model
        config['save_results'] = args.save_results
        config['mdl_dpath'] = args.mdl_dpath
    config['fold'] = args.fold
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.setdefault('log_fpath', f"./tmp_results/default_log_seed{config['seed']}_{timestamp}.log")
    if args.optuna_save:
        config['save_model'] = True
        config['save_results'] = True     
    if args.load_model_fpath:
        config['load_model_fpath'] = args.load_model_fpath


  #  if args.esm_pool:
  #      config['esm_pool'] = True
  #  else:
  #      config['esm_pool'] = False
    if args.esm_pool is not None:
        config['esm_pool'] = args.esm_pool


        # config['k_fold'] = args.k_fold
    if args.mdl_dpath:
        config['mdl_dpath'] = args.mdl_dpath

    # train a missense mutation PPI model or inference with this model
    print('k_fold', config['k_fold'])
    if config['exec_mode'] == 'train':
        if not args.use_optuna:
            learner = PPILearner(config)
            
            if config['k_fold']:
                for fold in range(int(config['k_fold'])):
                    learner.train(fold=fold)
            else:
                learner.train()
        else:
            def train(trial):
                optuna_savepath = 'optuna_{}'.format(args.optuna_name)
                os.makedirs(optuna_savepath, exist_ok=True)

                

                # define searching space for optuna
                optuna_dict = {}
                # optuna_dict['dropout_p'] = trial.suggest_uniform('dropout_p', 0.05, 0.5)
                # optuna_dict['lr_init'] = trial.suggest_uniform('lr_init', 0.001, 0.01)
                optuna_dict['dropout_p'] = trial.suggest_discrete_uniform('dropout_p', 0.05, 0.5, 0.00001)
                optuna_dict['lr_init'] = trial.suggest_discrete_uniform('lr_init', 0.001, 0.01, 0.00001)
                optuna_dict['one_label_weight'] = trial.suggest_categorical('one_label_weight', [1,2,3,4,5,6,7,8,9,10])
                # optuna_dict['one_label_weight'] = trial.suggest_categorical('one_label_weight', [1])
                optuna_dict['discrim_weight'] = trial.suggest_categorical('discrim_weight', [1.0, 2.0, 3.0, 4.0])
                config['dropout'] = trial.suggest_float('dropout', 0.2, 0.5)

                config['fc_hidden'] = trial.suggest_int('fc_hidden', 256, 1024)
                config['pred_layer_num'] = trial.suggest_int('pred_layer_num', 1, 3)


                trial_id = trial.number

                config['mdl_dpath'] = os.path.join(optuna_savepath, f'trial_{trial_id}')
                config['log_fpath'] = os.path.join(optuna_savepath, f'logs/log_trial_{trial_id}.txt')
                os.makedirs(config['mdl_dpath'], exist_ok=True)
                os.makedirs(os.path.dirname(config['log_fpath']), exist_ok=True)  # 确保 logs 目录存在


                #config['mdl_dpath'] = os.path.join(optuna_savepath, 'trail_{}'.format(trial_id))
                #os.makedirs(config['mdl_dpath'], exist_ok=True)
                with open(os.path.join(optuna_savepath, 'param_{}.txt'.format(trial_id)), 'w') as w:
                    for param in optuna_dict:
                        config[param] = optuna_dict[param]
                        w.write('{}: {}'.format(param, optuna_dict[param]))

               

 
                learner = PPILearner(config)
                
                trial.set_user_attr('trial_id', trial_id)
                if config['exec_mode'] == 'train':
                    if config['k_fold']:
                        acc_list = []
                        for fold in range(config['k_fold']):
                            acc, results_log, metrics_best = learner.train(fold=fold)
                            acc_list.append(acc)
                            trial.set_user_attr('fold_{}_acc'.format(fold), acc)

                            with open(os.path.join(optuna_savepath, 'optuna_reuslts.log'), 'a') as w:
                                w.write('trial_id: {}, seed: {}_{}, ep: {}, val_acc {:.5}, '.format(trial_id, results_log['seed'], fold, results_log['best_epoch'], results_log['val_acc_opt']))
                                for key in metrics_best:
                                    w.write('{}: {:.5} '.format(key, metrics_best[key]))
                                w.write('\n')
                        acc = np.mean(acc_list)
                        std = np.std(acc_list)
                        trial.set_user_attr('std', std)
                    else:
                        acc, results_log, metrics_best = learner.train()
                        with open(os.path.join(optuna_savepath, 'optuna_reuslts.log'), 'a') as w:
                            w.write('trial_id: {}, seed: {}, ep: {}, val_acc {:.5}, '.format(trial_id, results_log['seed'], results_log['best_epoch'], results_log['val_acc_opt']))
                            for key in metrics_best:
                                w.write('{}: {:.5} '.format(key, metrics_best[key]))
                                if isinstance(metrics_best[key], torch.Tensor):
                                    metrics_best[key] = metrics_best[key].item()
                                trial.set_user_attr(key, metrics_best[key])
                            w.write('\n')
                            trial.set_user_attr('best_epoch', results_log['best_epoch'])
                    return acc
            study_name = args.optuna_name
            study = optuna.create_study(study_name=study_name, storage=args.optuna_storage, direction='maximize', load_if_exists=True)
            study.optimize(train, n_trials=args.optuna_trials)


    elif config['exec_mode'] == 'eval':
        acc_list = []
        config['tst_fpath'] = args.eval_csv
        config['patch_dpath'] = args.feat_h5_fpath
        learner = PPILearner(config)
        # eval_model_dpath = args.eval_dpath

        # model_fpath = os.path.join(eval_model_dpath, 'model_pred_opt_fold_{}.pth'.format(i))
        model_fpath = args.eval_model_fpath
        save_result_fpath = args.save_result_fpath
        save_feature_fpath = args.save_feature_fpath

        acc, _ = learner.eval(model_fpath, save_result_fpath, save_feature_fpath, fold=args.fold)

        print("acc:", acc)

if __name__ == '__main__':
    main()
