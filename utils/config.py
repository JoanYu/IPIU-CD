# 

Config={
  "model": 'siamunet++',
  "metrics":[
    'cd_losses',
    'cd_corrects',
    'cd_precisions',
    'cd_recalls',
    'cd_f1scores',
    'learning_rate'
  ],
  "metrics_test":[
    'cd_corrects',
    'cd_precisions',
    'cd_recalls',
    'cd_f1scores',
  ],
  "patch_size": 256,
  "augmentation": True,
  "num_gpus": 2,
  "num_workers": 8,
  "num_channel": 3,
  "label_channel": 67,
  "EF": False,
  "epochs": 100,
  "batch_size": 8,
  "learning_rate": 1e-3,
  "loss_function": "hybrid",
  "dataset_dir": "data/",
  "weight_dir": "./outputs/",
  "log_dir": "./log/"
}
