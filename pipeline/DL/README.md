# DL：EEGNet 訓練與測試

讀取 [`data_process`](../data_process/) 產生的 `.pt` 檔，訓練與測試修改版 EEGNet。

## 快速開始

```bash
cd pipeline/DL

# 訓練（設定：json/setting_train.json）
python train.py --pt_path ../data_process/output/S01_processed.pt

# 用另一份資料測試（設定：json/setting_test.json，模型路徑寫在 model_load_path）
python test.py --pt_path path/to/another_session.pt
```

| 參數 | 說明 |
|---|---|
| `--pt_path` | 已處理的 `.pt` 檔 |
| `--setting_path` | 設定檔，預設 `json/setting_train.json`／`json/setting_test.json` |
| `--label_encoder_path` | 類別編碼檔，預設讀設定檔中的 `label_encoder_path`（`../label_encoder/label_encoder_6_7_8_9.pkl`） |
| `--result_json_path` | 結果 JSON 輸出位置，預設讀設定檔中的 `RESULT_JSON_PATH` |
| `--seed` | 隨機種子，預設 `60` |

## 訓練流程（`classification_func.py`）

1. **依 trial 切分**：Stratified K-fold（`n_splits`，預設 3），每一折的類別比例相同
2. **切分後才做滑動視窗**：訓練集與驗證集各自切成 1 秒、步長 0.1 秒的視窗。若先切視窗再切分，同一個 trial 的相鄰視窗會同時落在訓練與驗證集，造成資料洩漏
3. **可選逐類別正規化**（`if_normailze`）：只用訓練集計算每個類別的平均與標準差
4. **訓練 EEGNet**：每個 epoch 在驗證集上評估，保留驗證準確率最高的模型
5. **輸出**：各折準確率、平均準確率、混淆矩陣，連同當次設定一起寫入結果 JSON

## 模型（`model_architecture/EEGNet.py`）

以 EEGNet 為基礎修改：

| 層 | 做法 |
|---|---|
| 時間卷積 | **多個大小不同的卷積核並行**（預設 32、64、128、200、256、512 samples），同時擷取不同時間尺度的節律 |
| 空間卷積 | 多尺度 depthwise 空間卷積（全部通道、約一半通道、3 個通道） |
| 後段 | Separable convolution → pooling → dropout → 全連接分類 |
| 資料增強 | 訓練時加入高斯雜訊（`noise_std`）與隨機丟棄通道（`channel_dropout`） |

超參數（F1、D、F2、卷積核大小、dropout、epoch、batch size、學習率等）都在 `json/setting_train.json` 的 `DL_classifier_method.EEGNet` 中設定。

## 檔案

| 檔案 | 說明 |
|---|---|
| `train.py` | 訓練入口（`Trainer` 類別） |
| `test.py` | 測試入口（`Tester` 類別），載入已訓練模型評估另一份資料 |
| `classification_func.py` | K-fold 切分、滑動視窗、正規化、呼叫模型訓練與評估 |
| `model_architecture/EEGNet.py` | 模型定義、資料增強、訓練迴圈 |
| `utils/window_slice.py` | 滑動視窗切割 |
| `json/setting_train.json`、`json/setting_test.json` | 設定檔範例 |
