# legacy：第一版一體式分析流程（參考用）

這是專題初期（2025/11 – 2026/01）的第一版流程，資料讀取、前處理、特徵擷取、分類與結果紀錄都寫在同一支 `main_session.py` 中。之後拆分為 [`pipeline/data_process`](../pipeline/data_process/) 與 [`pipeline/DL`](../pipeline/DL/) 兩個獨立步驟，並改為與即時系統一致的處理方式。

> 這裡保留的是第一版的核心程式，供參考傳統特徵與消融實驗的做法。部分依賴的模組（如舊版 `src/config.py`、`src/EEGNet.py`）已移到新架構，因此**無法單獨執行**。

## 內容

| 檔案 | 說明 |
|---|---|
| [`main_session.py`](main_session.py) | 第一版主程式：依設定檔依序執行 training／testing／calibration 三種模式 |
| [`src/embedding.py`](src/embedding.py) | 傳統特徵擷取：**CSP**、**黎曼幾何**（共變異矩陣 → 切空間）、**Hjorth 參數**、**差分熵（DE）**、**PSD** |
| [`src/classifier.py`](src/classifier.py) | 分類器：**LDA**、**SVM**（one-vs-rest） |
| [`src/classification_func.py`](src/classification_func.py) | 傳統 ML 與 DL 的 K-fold 訓練與評估流程 |
| [`ablation/ablation.py`](ablation/ablation.py) | 消融實驗：對每位受試者，逐一開關六個前處理步驟並記錄結果 |
| [`ablation/ablation_plot.py`](ablation/ablation_plot.py) | 消融實驗結果繪圖 |
| [`ablation/setting_ablation_use.json`](ablation/setting_ablation_use.json) | 消融實驗使用的設定檔 |
| [`docs/`](docs/) | 第一版設定檔的說明（training／testing／calibration 三種模式） |

## 消融實驗設計

`ablation.py` 對每位受試者執行兩種方式：

- **移除法**：其他步驟全開，只關掉其中一個步驟，看準確率下降多少
- **加入法**：其他步驟全關，只開啟其中一個步驟，看準確率上升多少

結果彙整見 [`results/`](../results/README.md)。
