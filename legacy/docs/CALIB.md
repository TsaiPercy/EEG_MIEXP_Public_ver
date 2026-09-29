# EEG_MIEXP
MIEXP MI data process, Lab work

## 資料架構

所需資料結構
```
MI_data_process/
├── data/                           # 存放原始數據
│   └── calibration/                # 校準用 data
│   
├── model/   
│   ├── label_encoder_6_7_8_9.pkl           # label_encoder
│   │    
│   └── EEGNet_models/              # 存放訓練好的模型權重
│       └── model_train.pth         # 訓練完的 model

│   
├── json/                           # 通用 JSON 配置文件
│   ├── background.json             # 背景參數不會修改
│   └── path.json                   # 存放所有 .csv 路徑
│   
├── main/
│   ├── src/                        # 功能模組
│   │   ├── classification_func.py
│   │   ├── classifier.py
│   │   ├── config.py
│   │   ├── data_norm_std.py
│   │   ├── EEGNet.py
│   │   ├── embedding.py
│   │   ├── norm.py
│   │   ├── preprocess.py
│   │   ├── process_outlier.py
│   │   └── window_slice.py
│   │   
│   ├── main_session.py             # 主要執行程序
│   │   
│   └── data_path_update.py         # 更新 path.json 用
│
│
├── result_record/                  # 實驗結果紀錄
│
└── setting_json/                   # 實驗參數設定檔
    │
    └── setting_calib.json          # calib 範例
```


## 使用方法

參考 "setting_calib.json"，修改參數

run
```
python main/main_session.py setting_calib.json
```

"setting_calib.json" 說明
---------------------------------------------------------------------------------------------
```
"mode_control": {

    "all_mode": [
        "training",
        "testing",
        "calibration"
    ],

    # 使用 calibration mode
    "current_using": "calibration",

    # 存放 label_encoder 位置
    "label_encoder_path": "./model/label_encoder_6_7_8_9.pkl", 
},

```
---------------------------------------------------------------------------------------------   
```

"DL_classifier_method": {} --->


"EEGNet": {

    # 小筆記
    "trails_target_data_DESCRIPTION": "choose_channels + add_C3_C4 + diff_C3_C4 + diff_C4_C3 + Cz*2",
    
    # calibration mode load 已經 train 好的 model
    "model_load_path": "./model/EEGNet_models/model_train.pth",

    # 新 model 路徑，若留空則不紀錄 model
    "model_save_path": "./model/EEGNet_models/model_calib.pth",

    # cross valid 比例，6 表示 train : valid = 5 : 1
    "n_splits": 6,

    # 開關
    "if_normailze": true,



    # mode 會自動填寫
    "mode": "calibration",

    # n_classes 會自動填寫
    "n_classes": 4,

    
    # 會自動紀錄 normailze 時所使用的所有 mean 跟 data
    "ALL_MARKER_MEAN": [
        [],
        [],
        [],
        []
    ],
    "ALL_MARKER_STD": [
        [],
        [],
        [],
        []
    ],


    # EEGNet 初始化參數
    "init": {
        "F1": 4,
        "D": 2,
        "F2": 8,

        # multi T kernel
        "T_kernSizes": [
            32,
            64,
            128,
            200,
            256,
            512
        ],

        # 目前自動採用 3, half_#channel, #channel
        "S_kernSizes": [],
        "dropout": 0.55
    },

    "augment": {
        "noise_std": 0.01,

        # channel_dropout 目前未使用
        "channel_dropout": 0.1
    },

    # train 用 hyper
    "using": {
        "n_epochs": 60,
        "batch_size": 16,
        "seed": 42,
        "lr": 0.005,
        "weight_decay": 0.0001,
        "max_grad_norm": 1.0
    }
}

```


---------------------------------------------------------------------------------------------  


## result_train.json 額外說明

其實 result.json 的上半部，就是如何產生這個 result.json 的 setting.json，所以要復現就直接把 result.json 當成 setting.json ，跑 main_session.py 就好

```
"all_accuracy": {

    # 模型名稱 (目前只實作了 EEGNet，可以嘗試 SCCNet or shallownet)
    "EEGNet": {
        "calib_result": {

            # "n_splits" 數量 == fold 數量 
            "FOLDS_HIGHEST_ACC": [
                0.8822916666666667,
                0.825,
                0.9010416666666666,
                0.91875,
                0.8901041666666667,
                0.8760416666666667
            ],
            "AVE_ACC": 0.8822048611111112,
            "HIGHEST_ACC": 0.91875,
            "LOWEST_ACC": 0.825,
            "confusion_matrix": [
                [
                    2544,
                    243,
                    44,
                    49
                ],
                [
                    195,
                    2600,
                    38,
                    47
                ],
                [
                    96,
                    103,
                    2478,
                    203
                ],
                [
                    106,
                    66,
                    167,
                    2541
                ]
            ]
        }
    }
}
