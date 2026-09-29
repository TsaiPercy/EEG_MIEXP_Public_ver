# EEG_MIEXP
MIEXP MI data process, Lab work

## 資料架構

所需資料結構
```
MI_data_process/
├── data/                           # 存放原始數據
│   └── testing/                    # 待測試 data
│  
├── model/
│   ├── label_encoder.pkl           # label_encoder
│   │  
│   └── EEGNet_models/              # 存放訓練好的模型權重
│       ├── model_train.pth         # 訓練完的 model
│       └── model_calib.pth         # 校準完的 model
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
    └── setting_test.json          # test 範例
```


## 使用方法

參考 "setting_test.json"，修改參數

run
```
python main/main_session.py setting_test.json
```

"setting_test.json" 說明
---------------------------------------------------------------------------------------------
```
"mode_control": {

    "all_mode": [
        "training",
        "testing",
        "calibration"
    ],

    # 使用 testing mode
    "current_using": "testing",

    # 存放 label_encoder 位置
    "label_encoder_path": "./model/label_encoder_6_7_8_9.pkl"
},

```
---------------------------------------------------------------------------------------------   
```

"DL_classifier_method": {} --->


"EEGNet": {

    # 小筆記
    "trails_target_data_DESCRIPTION": "choose_channels + add_C3_C4 + diff_C3_C4 + diff_C4_C3 + Cz*2",
    
    # testing mode load 已經 train or calib 好的 model
    "model_load_path": "./model/EEGNet_models/model_train.pth",

    # testing mode 不用 save
    "model_save_path": "",

    # testing mode 用不到 n_splits
    "n_splits": 6,

    # follow training 
    "if_normailze": true,

    # mode 會自動填寫
    "mode": "testing",

    # 下面設定都 follow training
    "n_classes": 4,
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
    "init": {
        "F1": 4,
        "D": 2,
        "F2": 8,
        "T_kernSizes": [
            32,
            64,
            128,
            200,
            256,
            512
        ],
        "S_kernSizes": [],
        "dropout": 0.55
    },
    "augment": {
        "noise_std": 0.01,
        "channel_dropout": 0.1
    },
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

    "test_result": {

        "EEGNet": {
            # testing 只有 準確率 跟 confusion_matrix 
            "ACC": ,
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
