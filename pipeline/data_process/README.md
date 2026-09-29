# data_process：原始 CSV → `.pt`

把 Cygnus 匯出的原始 EEG（`.csv`）前處理後，存成訓練用的 `.pt` 檔。

## 快速開始

```bash
cd pipeline/data_process

# 依 json/setting.json 處理，輸出到 output/<SUBJECT_NAME>_processed.pt
python data_processor.py

# 產出後驗證格式
python data_processor.py --verify

# 一次處理多位受試者（逐一改寫 setting.json 後呼叫 data_processor.py）
python multi_data_process.py
```

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--setting_path` | `json/setting.json` | 前處理設定 |
| `--background_path` | `../../global_json/background.json` | 通道與 marker 定義 |
| `--output` | `output/{SUBJECT_NAME}_processed.pt` | 輸出路徑 |
| `--verify` | 關閉 | 產出後驗證 `.pt` 格式 |
| `--seed` | `60` | 隨機種子 |

原始 CSV 放在 `setting.json` 的 `data_find_folder_path`（預設 `ori_csv_data/S01/`），程式會遞迴搜尋底下所有 CSV，並把路徑寫進 `json/path.json`。

## 處理流程

```
CSV
 ↓ 1. 讀取 CSV，擷取事件 marker（Cue 後 +500 samples = 運動想像開始）
 ↓ 2. 轉為 MNE RawArray
 ↓ 3. 前處理：因果帶通濾波 1–40 Hz（phase='minimum'）、平均參考、可選 ICA
 ↓ 4. 離群值處理（可選，在濾波後的訊號上偵測較準確）
 ↓ 5. 切割 trial（tmin / tmax），合併所有 run，marker 6/7/8/9 → 類別 0/1/2/3
 ↓ 6. 通道選擇（依 background.json 的通道組合）
 ↓ 7. 通道增強（可選：C3+C4、C3−C4、C4−C3、Cz×2）
 ↓ 8. 時間裁剪出運動想像區段
 ↓ 9. 正規化：逐 trial、逐通道去均值（與即時系統一致）
 ↓ 10. 時間平移增強：每個 trial 隨機平移 ±0.2 秒，產生多個版本
 ↓
.pt
```

## 輸出格式

```python
{
    "x_data": torch.Tensor,   # (trial, channel, sample), float32
    "y_data": torch.Tensor,   # (trial,), int64
    "probs":  torch.Tensor,   # (trial, n_classes)，預設全 0，保留給即時系統使用
    "metadata": {
        "sfreq": 500,
        "channels": [...],
        "n_classes": 4,
        "subject": "S01",
        "normalization_method": "per_channel_demean",
        "time_jitter": True,
        "causal_filter": True,
        ...
    },
}
```

## `json/setting.json` 主要設定

| 區塊 | 欄位 | 說明 |
|---|---|---|
| `data_info` | `SUBJECT_NAME`、`sfreq`、`data_find_folder_path`、`label_encoder_path` | 受試者代號、取樣率、CSV 資料夾、類別編碼檔 |
| `choose_channels` | `MODE` | 通道組合：`9_mid`、`11_TC_mid`、`13_FC_mid_Pz` 等（定義在 `background.json`） |
| `choose_markers` | `NAMES` | 使用的 marker，預設為四類運動想像 |
| `preprocessing_trues` | `filter_true`、`reference_true`、`ICA_true`、`find_bad_channel_true` | 各前處理步驟開關 |
| `process_outlier` | `use`、`detect_method`（`zscore`／`iqr`）、`replace_strategy` | 離群值處理 |
| `data_normalization_standardization` | `use`、`method` | `per_channel_demean`（建議）、`ema`、`z_score`、`zero_mean` |
| `time_jitter_augmentation` | `use`、`max_shift_sec`、`augment_factor` | 時間平移增強 |
| `get_trail_data` | `tmin`、`tmax`、`base_time` | trial 切割區間與基線區間 |

## 檔案

| 檔案 | 說明 |
|---|---|
| `data_processor.py` | 主程式：`MIEXPDataProcessor` 類別與 CLI |
| `multi_data_process.py` | 批次處理多位受試者 |
| `utils/preprocess.py` | MNE 濾波、平均參考、ICA、壞通道偵測 |
| `utils/process_outlier.py` | 離群值偵測與取代 |
| `utils/data_norm_std.py`、`utils/norm.py` | 各種正規化方法 |
| `utils/data_path_update.py` | 掃描 CSV 並更新 `path.json` |
| `json/path.json` | CSV 路徑表（程式自動產生，目前內容為格式範例） |
