fold_number=$1

CUDA_VISIBLE_DEVICES=1 python main.py \
  --config_fname config/config_eval_OD.yaml \
  --data random \
  --eval \
  --overwrite \
  --eval_csv datasets/csvs/random_fold_$fold_number.csv \
  --eval_model_fpath tmp_results/ODrd01_5fold/exp0/model_pred_opt_fold_$fold_number.pth \
  --feat_h5_fpath datasets/embeddings/sdnn_corrected_ppi.h5 \
  --save_result_fpath eval_results/ODrd01_5fold_exp0_fold$fold_number.csv \
  --save_feature_fpath eval_results/feat_ODrd01_5fold_exp0_fold$fold_number.pt