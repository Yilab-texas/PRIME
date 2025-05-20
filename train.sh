export GLIBC_PATH=$HOME/glibc-2.27-bin/lib/x86_64-linux-gnu

#$GLIBC_PATH/ld-2.27.so --library-path $GLIBC_PATH:$CONDA_PREFIX/lib $CONDA_PREFIX/bin/python main.py --config_fname config_train.yaml --overwrite --save_model --save_results

$GLIBC_PATH/ld-2.27.so --library-path $GLIBC_PATH:$CONDA_PREFIX/lib $CONDA_PREFIX/bin/python main.py \
  --config_fname config_train.yaml \
  --use_optuna \
  --optuna_name sdnn \
  --optuna_trials 200 \
  --optuna_save
