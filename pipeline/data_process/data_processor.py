"""
data_processor.py

MIEXP 離線實驗資料前處理 Pipeline

流程: .csv → MIEXPDataProcessor → .pt
之後: .pt → training

輸出 .pt 格式:
    {
        'x_data': torch.Tensor,   # (Trial, Channel, Sample)
        'y_data': torch.Tensor,   # (Trial,)
        'probs':  torch.Tensor,   # (Trial, n_classes) → 預設全 0
        'metadata': {
            'sfreq': int,
            'channels': list[str],
            'n_classes': int,
            'source_csvs': list[str],
            'subject': str,
        }
    }

使用方式:
    cd MI_data_process/main/data_process
    python data_processor.py
    python data_processor.py --setting_path ./json/setting.json --output ./output/result.pt
"""

import os
import sys
import argparse
import glob
import json
import copy
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import mne

from sklearn.preprocessing import LabelEncoder
import pickle

# ─── 基準目錄：data_processor.py 所在的 data_process/ 資料夾 ───
# 所有相對路徑都以此為基準，不依賴 CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 本地模組
sys.path.insert(0, BASE_DIR)
from utils import data_path_update as dpu
from utils import config
from utils import data_norm_std as dns
from utils import preprocess as pps
from utils import process_outlier as pcso



# ═══════════════════════════════════════════
# log setting
# ═══════════════════════════════════════════

class TeeLogger:
    """雙向輸出 Logger，同時寫入終端機與檔案"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_dual_logging():
    """設定 Log 雙向輸出 (終端機與檔案)"""
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True) 
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"process_log_{current_time}.txt")

    original_stdout = sys.stdout
    log_file = open(log_filename, "w", encoding="utf-8")
    sys.stdout = TeeLogger(original_stdout, log_file)


    print(f"✅ Log 將同時儲存至: {log_filename}")
    
    # 將 file object 和 original_stdout 回傳，以便主程式結束時能妥善關閉與還原
    return log_file, original_stdout




# ═══════════════════════════════════════════
# main class
# ═══════════════════════════════════════════

class MIEXPDataProcessor:
    """
    MIEXP 離線實驗資料前處理 Pipeline

    將 main_session.py 的 Step 1-14（train 之前的所有資料處理）
    封裝為可重用的 class。

    Attributes:
        settingData: 從 setting.json 讀取的設定
        backgroundData: 從 background.json 讀取的背景資訊
        processedData: 處理完成後的資料 dict
    """

    # ─── 事件偏移常數（cue → MI 開始） ───
    CUE_TO_MI_OFFSET_SAMPLES = 500  # 1s × 500Hz

    def __init__(self, settingPath=None, backgroundPath=None, seed=60):
        """
        初始化 MIEXPDataProcessor

        Args:
            settingPath: setting.json 路徑
            backgroundPath: background.json 路徑
            seed: 隨機種子
        """
      
        self.settingPath = settingPath
        self.backgroundPath = backgroundPath
        self.seed = seed

        # 載入設定
        self.settingData = self._loadJson(self.settingPath)
        self.backgroundData = self._loadJson(self.backgroundPath)

        # 套用 channel mode
        self.settingData = self._applyChannelMode(self.settingData, self.backgroundData)

        # 解析核心參數
        self.sfreq = self.settingData["data_info"]["sfreq"]
        self.allChannelNames = self.backgroundData["channel"]["all_channels"]["NAMES"]
        self.allChannelNum = len(self.allChannelNames)

        # 解析目標 marker
        self.chooseMarkerNames = self.settingData["choose_markers"]["NAMES"]
        self.chooseMarkers, self.chooseMarkerDict = self._parseMarkers()
        self.nClasses = len(self.chooseMarkers)

        # 解析目標通道
        self.chooseChannelNames = self.settingData["choose_channels"]["NAMES"]

        # Epoch 參數
        epochCfg = self.settingData["get_trail_data"]
        self.tmin = epochCfg["tmin"]
        self.tmax = epochCfg["tmax"]
        self.baseTime = epochCfg["base_time"]

        # 處理結果
        self.processedData = None

        # 設定隨機種子
        self._setSeed(self.seed)

        print(f"[MIEXPDataProcessor] 初始化完成")
        print(f"  settingPath={settingPath}")
        print(f"  sfreq={self.sfreq}")
        print(f"  allChannels={self.allChannelNum}, chooseChannels={len(self.chooseChannelNames)}")
        print(f"  markers={self.chooseMarkers} ({self.nClasses} classes)")
        print(f"  tmin={self.tmin}, tmax={self.tmax}, baseline={self.baseTime}")

    # ═══════════════════════════════════════════
    # 私有工具
    # ═══════════════════════════════════════════

    @staticmethod
    def _setSeed(seed):
        """設定全域隨機種子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _loadJson(path):
        """載入 JSON 檔案"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _applyChannelMode(settingData, backgroundData):
        """根據 MODE 填入 choose_channels"""
        mode = settingData["choose_channels"]["MODE"]
        settingData["choose_channels"].update(backgroundData["channel"][mode])
        return settingData


    def _parseMarkers(self):
        """解析目標 marker 號碼與名稱映射"""
        allEventMarkerDict = self.backgroundData['event_markers']
        chooseMarkers = []
        chooseMarkerDict = {}

        for name in self.chooseMarkerNames:
            val = allEventMarkerDict[name]
            chooseMarkerDict[name] = val
            chooseMarkers.append(val)

        return chooseMarkers, chooseMarkerDict

    

    # ═══════════════════════════════════════════
    # Step 1-2: 資料載入
    # ═══════════════════════════════════════════

    def _loadCsvPaths(self):
        """取得 CSV 路徑列表（路徑相對於 BASE_DIR 解析）"""
        pathJsonPath = os.path.join(BASE_DIR, "json", "path.json")

        # 如果需要重新掃描
        if self.settingData["data_info"]["IF_DATA_RELOAD"]:
            with open(pathJsonPath, "r", encoding="utf-8") as f:
                pathData = json.load(f)
            pathData["data_find_folder_path"] = self.settingData["data_info"]["data_find_folder_path"]
            with open(pathJsonPath, "w", encoding="utf-8") as f:
                json.dump(pathData, f, ensure_ascii=False, indent=4)
            dpu.save_csv_paths_to_json(pathJsonPath)


        with open(pathJsonPath, "r", encoding="utf-8") as f:
            pathData = json.load(f)

        csvPaths = pathData["data_paths"]["PATHS"]
        print(f"[DEBUG] _loadCsvPaths: 找到 {len(csvPaths)} 個 CSV")
        return csvPaths

    def _loadCsvs(self, csvPaths):
        """
        CSV → Pandas DataFrame 列表

        Args:
            csvPaths: CSV 路徑列表

        Returns:
            dataframes: list[pd.DataFrame]，每個元素為一個 run
        """
        dataframes = []
        for csvPath in csvPaths:
            df = pd.read_csv(csvPath, skiprows=10, encoding="utf-8")
            dataframes.append(df)
            print(f"[DEBUG] _loadCsvs: {os.path.basename(csvPath)}, shape={df.shape}")

        return dataframes

    # ═══════════════════════════════════════════
    # Step 3: 事件提取
    # ═══════════════════════════════════════════

    def _extractEvents(self, dataframes):
        """
        從 marker 欄位提取事件，並加入 cue → MI 偏移

        MIEXP 實驗時間線：
            marker (cue) 出現 → 1s cue 顯示 → 3s MI 執行
            所以 MI 座標 = cue 座標 + 500 samples

        Args:
            dataframes: list[pd.DataFrame]

        Returns:
            eventsMulti: list[np.ndarray]，每個 (n_events, 3)
                         格式: [MI_start_idx, prev_val, marker_val]
        """
        eventsMulti = []

        for df in dataframes:
            markerCol = df['Software Marker'].fillna(0).to_numpy()
            eventsList = []

            for idx, val in enumerate(markerCol):
                if val in self.chooseMarkers:
                    # cue 座標 + 偏移 = MI 開始座標
                    eventsList.append([
                        idx + self.CUE_TO_MI_OFFSET_SAMPLES,
                        0,
                        int(val)
                    ])

            events = np.array(eventsList).astype(int)
            eventsMulti.append(events)
            print(f"[DEBUG] _extractEvents: run events={events.shape}")

        return eventsMulti

    # ═══════════════════════════════════════════
    # Step 4: Pandas → NumPy
    # ═══════════════════════════════════════════

    def _toNumpy(self, dataframes, channelNames):
        """
        從 DataFrame 中提取指定通道，轉為 NumPy

        Args:
            dataframes: list[pd.DataFrame]
            channelNames: 要提取的通道名稱列表

        Returns:
            dataList: list[np.ndarray]，每個 (channel, time)
        """
        dataList = []
        for df in dataframes:
            dfNoNan = df.fillna(0)
            dfChannels = dfNoNan[channelNames]
            npData = dfChannels.to_numpy().T  # (time, ch) → (ch, time)
            dataList.append(npData)

        print(f"[DEBUG] _toNumpy: {len(dataList)} runs, shape={dataList[0].shape}")
        return dataList


    # ═══════════════════════════════════════════
    # Step 5-6: MNE 建立 + 前處理
    # ═══════════════════════════════════════════

    def _createMneRaw(self, dataList):
        """
        NumPy → MNE RawArray

        Args:
            dataList: list[np.ndarray]，每個 (channel, time)

        Returns:
            rawList: list[mne.io.RawArray]
        """
        channelTypes = ['eeg'] * self.allChannelNum
        info = mne.create_info(
            ch_names=self.allChannelNames,
            sfreq=self.sfreq,
            ch_types=channelTypes
        )

        rawList = []
        for data in dataList:
            raw = mne.io.RawArray(data, info)
            raw.set_montage('standard_1020')
            rawList.append(raw)

        print(f"[DEBUG] _createMneRaw: {len(rawList)} RawArray")
        return rawList

    def _mnePreprocess(self, rawList):
        """
        MNE 前處理（濾波、ICA、re-reference 等）

        Args:
            rawList: list[mne.io.RawArray]

        Returns:
            processedList: list[mne.io.RawArray]
        """
        ppsCfg = self.settingData["preprocessing_trues"]
        processedList = pps.preprocessing_multi(
            rawList,
            find_bad_channel_true=ppsCfg["find_bad_channel_true"],
            filter_true=ppsCfg["filter_true"],
            reference_true=ppsCfg["reference_true"],
            ICA_true=ppsCfg["ICA_true"]
        )
        print(f"[DEBUG] _mnePreprocess: 完成")
        return processedList




    # ═══════════════════════════════════════════
    # Step 7: 離群值處理
    # ═══════════════════════════════════════════

    def _processOutliers(self, dataList):
        """
        離群值偵測與替換（可選）

        Args:
            dataList: list[np.ndarray]，每個 (channel, time)

        Returns:
            processedList: list[np.ndarray]
        """
        outlierCfg = self.settingData["process_outlier"]
        if not outlierCfg["use"]:
            print(f"[DEBUG] _processOutliers: 跳過（未啟用）")
            return dataList

        processedList = pcso.process_outlier_multi(
            dataList,
            self.allChannelNames,
            detect_method=outlierCfg["detect_method"],
            replace_strategy=outlierCfg["replace_strategy"],
            prt_info=outlierCfg.get("prt_info", False)
        )
        print(f"[DEBUG] _processOutliers: 完成")
        return processedList




    # ═══════════════════════════════════════════
    # Step 8: 正規化 / 標準化
    # ═══════════════════════════════════════════

    def _normalize(self, data):
        """
        正規化（可選）

        支援兩種輸入格式:
          - list[np.ndarray]: 每個 (channel, time)，對每個 run 獨立處理
          - np.ndarray: (Trial, Channel, Time)，對每個 trial 獨立處理
            → 這是對齊即時端的用法（per-trial demean）

        Args:
            data: list[np.ndarray] 或 np.ndarray

        Returns:
            正規化後的資料（格式與輸入一致）
        """
        normCfg = self.settingData["data_normalization_standardization"]
        if not normCfg["use"]:
            print(f"[DEBUG] _normalize: 跳過（未啟用）")
            return data

        method = normCfg["method"]

        if isinstance(data, np.ndarray) and data.ndim == 3:
            # 3D input: (Trial, Channel, Time) — per-trial 正規化
            normalized = dns.norm_std_func(data, method=method)
            print(f"[DEBUG] _normalize: 完成 (3D per-trial), method={method}, shape={data.shape}")
            return normalized
        elif isinstance(data, list):
            # list input: 每個 (channel, time) — per-run 正規化
            normalizedList = []
            for runData in data:
                normalized = dns.norm_std_func(runData, method=method)
                normalizedList.append(normalized)
            print(f"[DEBUG] _normalize: 完成 (list per-run), method={method}")
            return normalizedList
        else:
            raise ValueError(f"[ERROR] _normalize: 不支援的輸入格式 type={type(data)}")






    # ═══════════════════════════════════════════
    # Step 9: 切 Epochs
    # ═══════════════════════════════════════════

    def _cutEpochs(self, rawList, eventsMulti):
        """
        使用 MNE Epochs 切割 trials

        已對事件座標做 +500 偏移（cue → MI），
        所以 tmin=0 即為 MI 開始。

        Args:
            rawList: list[mne.io.RawArray]
            eventsMulti: list[np.ndarray]

        Returns:
            epochsList: list[mne.Epochs]
        """
        epochsList = []
        for raw, events in zip(rawList, eventsMulti):
            epochs = mne.Epochs(
                raw,
                events,
                event_id=self.chooseMarkerDict,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=tuple(self.baseTime) if self.baseTime else None,
                preload=True
            )
            epochsList.append(epochs)
            print(f"[DEBUG] _cutEpochs: run epochs={epochs.get_data().shape}")

        return epochsList

    # ═══════════════════════════════════════════
    # Step 10-11: 合併 + Marker 映射
    # ═══════════════════════════════════════════

    def _combineAndRemap(self, epochsList):
        """
        合併所有 run 的 trials，並將 marker 映射為 0-based

        6→0, 7→1, 8→2, 9→3（透過 LabelEncoder）

        Args:
            epochsList: list[mne.Epochs]

        Returns:
            combineData: np.ndarray (Trial, Channel, Time)
            combineEvent: np.ndarray (Trial,)
            labelEncoder: LabelEncoder
        """
        allData = []
        allEvents = []

        for epochs in epochsList:
            allData.append(epochs.get_data())
            allEvents.append(epochs.events[:, -1])

        combineData = np.concatenate(allData, axis=0)
        combineEvent = np.concatenate(allEvents, axis=0)

        # LabelEncoder: 6,7,8,9 → 0,1,2,3
        labelEncoder = LabelEncoder()
        combineEvent = labelEncoder.fit_transform(combineEvent)

        # 儲存 LabelEncoder
        lePath = self.settingData["data_info"].get("label_encoder_path", "")
        if lePath:
            os.makedirs(os.path.dirname(lePath), exist_ok=True)
            pickle.dump(labelEncoder, open(lePath, "wb"))
            print(f"[DEBUG] LabelEncoder 已儲存: {lePath}")

        print(f"[DEBUG] _combineAndRemap: data={combineData.shape}, "
              f"events={combineEvent.shape}, "
              f"classes={np.unique(combineEvent)}")

        return combineData, combineEvent, labelEncoder

    # ═══════════════════════════════════════════
    # Step 12: 通道選擇
    # ═══════════════════════════════════════════

    def _selectChannels(self, combineData):
        """
        從全通道中選取目標通道子集

        使用名稱匹配（case-insensitive），不硬編碼 index

        Args:
            combineData: np.ndarray (Trial, AllChannel, Time)

        Returns:
            targetData: np.ndarray (Trial, ChooseChannel, Time)
        """
        allNamesLower = [ch.lower() for ch in self.allChannelNames]
        chooseNamesLower = [ch.lower() for ch in self.chooseChannelNames]

        targetIdx = [allNamesLower.index(ch) for ch in chooseNamesLower]
        targetData = combineData[:, targetIdx, :]

        print(f"[DEBUG] _selectChannels: {combineData.shape[1]} → {targetData.shape[1]} channels")
        return targetData

    # ═══════════════════════════════════════════
    # Step 13: 通道增強（可選）
    # ═══════════════════════════════════════════

    def _augmentChannels(self, targetData, combineData):
        """
        合成通道增強: C3+C4, C3-C4, C4-C3, Cz*2

        使用名稱查找，不硬編碼 index

        Args:
            targetData: np.ndarray (Trial, ChooseChannel, Time)
            combineData: np.ndarray (Trial, AllChannel, Time) — 從全通道取 C3/Cz/C4

        Returns:
            augmentedData: np.ndarray (Trial, ChooseChannel+4, Time)
        """
        useDlAug = self.settingData.get("channel_aug", {}).get("DL", False)
        if not useDlAug:
            print(f"[DEBUG] _augmentChannels: 跳過（未啟用）")
            return targetData

        # 按名稱查找 C3, Cz, C4 在全通道中的 index
        allNamesLower = [ch.lower() for ch in self.allChannelNames]
        try:
            idxC3 = allNamesLower.index("c3")
            idxCz = allNamesLower.index("cz")
            idxC4 = allNamesLower.index("c4")
        except ValueError as findErr:
            print(f"[WARNING] _augmentChannels: 找不到 C3/Cz/C4: {findErr}，跳過增強")
            return targetData

        c3 = np.expand_dims(combineData[:, idxC3, :], axis=1)
        cz = np.expand_dims(combineData[:, idxCz, :], axis=1)
        c4 = np.expand_dims(combineData[:, idxC4, :], axis=1)

        augmentedData = np.concatenate([
            targetData,
            c3 + c4,      # addition_C3_C4
            c3 - c4,      # diff_C3_C4
            c4 - c3,      # diff_C4_C3
            cz * 2,       # Cz*2
        ], axis=1)

        print(f"[DEBUG] _augmentChannels: {targetData.shape[1]} → {augmentedData.shape[1]} channels")
        return augmentedData

    # ═══════════════════════════════════════════
    # Step 14: 時間裁剪（MI 有效區段）
    # ═══════════════════════════════════════════

    def _cropTime(self, data):
        """
        裁剪出 MI 有效時間區段

        原始 Epoch 時間軸: tmin ~ tmax（例如 -3.0 ~ 3.25 秒）
        MI 開始點 = 0 秒（因為事件已做 +500 偏移）
        MI 結束點 = tmax 秒

        所以要取: [-tmin * sfreq : -tmin * sfreq + MI_duration * sfreq]
        即從 「0 秒」到「tmax 秒」的區段

        Args:
            data: np.ndarray (Trial, Channel, Time)

        Returns:
            croppedData: np.ndarray (Trial, Channel, CroppedTime)
        """
        # t=0 在 epoch 中的 sample index
        miStartSample = int(abs(self.tmin) * self.sfreq)
        # MI 區段結束（取到 tmax）
        miEndSample = int((abs(self.tmin) + self.tmax) * self.sfreq)

        # 邊界保護
        miEndSample = min(miEndSample, data.shape[2])

        croppedData = data[:, :, miStartSample:miEndSample]

        duration = (miEndSample - miStartSample) / self.sfreq
        print(f"[DEBUG] _cropTime: {data.shape[2]} → {croppedData.shape[2]} samples "
              f"(t=0 ~ t={duration:.2f}s)")

        return croppedData

    # ═══════════════════════════════════════════
    # Step 15: 時間平移資料增強
    # ═══════════════════════════════════════════

    def _augmentTimeJitter(self, data, labels):
        """
        時間平移資料增強 (Time Jitter Augmentation)
        
        對每個 trial 產生隨機時間偏移版本，當超出邊界時用 0 補充。
        這樣保持原始長度但引入時間隨機性。
        
        Args:
            data:   np.ndarray (Trial, Channel, Time) — 已裁剪至 MI 有效區段
            labels: np.ndarray (Trial,)
        
        Returns:
            augData:   np.ndarray (Trial*(1+augmentFactor), Channel, OriginalTime)
            augLabels: np.ndarray (Trial*(1+augmentFactor),)
        """
        jitterCfg = self.settingData.get("time_jitter_augmentation", {})
        if not jitterCfg.get("use", False):
            print(f"[DEBUG] _augmentTimeJitter: 跳過（未啟用）")
            return data, labels

        maxShiftSec = jitterCfg.get("max_shift_sec", 0.2)
        augmentFactor = jitterCfg.get("augment_factor", 3)
        maxShiftSamples = int(maxShiftSec * self.sfreq)

        nTrials, nChannels, nTime = data.shape
        targetLen = nTime  # ★ 改: 保持原始長度，不扣除 margin

        augDataList = []
        augLabelList = []

        for trialIdx in range(nTrials):
            trial = data[trialIdx]  # (Channel, Time)

            # 原始版本（無偏移）
            augDataList.append(trial.copy())
            augLabelList.append(labels[trialIdx])

            # 增強版本（隨機偏移 + 邊界補 0）
            for augIdx in range(augmentFactor):
                shiftSamples = random.randint(-maxShiftSamples, maxShiftSamples)
                
                # 建立等長的新陣列，初始化為 0
                augTrial = np.zeros_like(trial)
                
                # 計算有效貼上範圍
                srcStart = max(0, -shiftSamples)
                srcEnd = min(nTime, nTime - shiftSamples)
                dstStart = max(0, shiftSamples)
                dstEnd = dstStart + (srcEnd - srcStart)
                
                # 將原始資料貼到偏移位置，超出部分自動變 0
                augTrial[:, dstStart:dstEnd] = trial[:, srcStart:srcEnd]
                
                augDataList.append(augTrial)
                augLabelList.append(labels[trialIdx])

        augData = np.stack(augDataList, axis=0)
        augLabels = np.array(augLabelList)

        print(f"[DEBUG] _augmentTimeJitter: {nTrials} → {augData.shape[0]} trials "
            f"(factor={augmentFactor}, shift=±{maxShiftSec}s={maxShiftSamples}samples, "
            f"targetLen={targetLen}, padding_method='zero')")

        return augData, augLabels

    # ═══════════════════════════════════════════
    # 主流程
    # ═══════════════════════════════════════════

    def process(self):
        """
        執行完整前處理 Pipeline

        前處理順序（v2 — 對齊即時端）:
            1. 載入 CSV
            2. 提取事件（+500 偏移）
            3. 轉 NumPy
            4. MNE RawArray + 因果濾波 (phase='minimum')
            5. 離群值處理（可選）
            6. 重建 MNE RawArray → 切 Epochs
            7. 合併 + Marker 映射
            8. 通道選擇
            9. 通道增強（可選）
            10. 時間裁剪（MI 有效區段）
            11. 正規化（per-trial demean，對齊即時端）  ← 移到 Epoch 後
            12. 時間平移增強（可選）                    ← 新增

        為什麼正規化移到 Epoch 後:
            - 即時端的 each_channel_demean 是在每個 sliding window 上獨立計算
            - 離線端也應該在每個 trial 上獨立做 demean，確保分佈一致
            - 避免全域統計量洩漏到即時環境無法取得的資訊

        Returns:
            processedData: dict
                'x_data': np.ndarray (Trial, Channel, Sample)
                'y_data': np.ndarray (Trial,)
                'probs':  np.ndarray (Trial, n_classes) — 全 0 預設
        """
        print(f"\n{'#'*60}")
        print(f"# MIEXPDataProcessor.process() 開始 (v2 — 對齊即時端)")
        print(f"{'#'*60}\n")

        # Step 1-2: 載入 CSV
        csvPaths = self._loadCsvPaths()
        dataframes = self._loadCsvs(csvPaths)

        # Step 3: 提取事件
        eventsMulti = self._extractEvents(dataframes)

        # Step 4: 轉 NumPy（全通道）
        dataList = self._toNumpy(dataframes, self.allChannelNames)

        # Step 5: MNE RawArray + 因果濾波 (phase='minimum')
        # ★ 使用因果 FIR 濾波器，只看過去資料，與即時端行為一致
        rawList = self._createMneRaw(dataList)
        rawList = self._mnePreprocess(rawList)

        # 從 MNE 取回濾波後的 NumPy 資料
        filteredDataList = [raw.get_data() for raw in rawList]
        print(f"[DEBUG] 因果濾波後取回 NumPy: {len(filteredDataList)} runs")

        # Step 6: 離群值處理（在濾波後的乾淨訊號上偵測）
        filteredDataList = self._processOutliers(filteredDataList)

        # ★ 注意：正規化已移到切 Epoch 之後（per-trial 級別）

        # Step 7: 重建 MNE RawArray（使用處理後資料），用於切 Epochs
        rawListClean = self._createMneRaw(filteredDataList)

        # Step 8: 切 Epochs
        epochsList = self._cutEpochs(rawListClean, eventsMulti)

        # Step 9: 合併 + Marker 映射
        combineData, combineEvent, labelEncoder = self._combineAndRemap(epochsList)

        # Step 10: 通道選擇
        targetData = self._selectChannels(combineData)

        # Step 11: 通道增強
        targetData = self._augmentChannels(targetData, combineData)

        # Step 12: 時間裁剪
        targetData = self._cropTime(targetData)

        # Step 13: 正規化（per-trial demean，在切完 Epoch 後做）
        # ★ 對齊即時端: 即時端在每個 sliding window 上做 each_channel_demean
        #   離線端在每個 trial 上做 per_channel_demean
        targetData = self._normalize(targetData)

        # Step 14: 時間平移增強（可選）
        targetData, combineEvent = self._augmentTimeJitter(targetData, combineEvent)

        # 組裝輸出
        nTrials = targetData.shape[0]
        self.processedData = {
            'x_data': targetData,                                    # (Trial, Channel, Sample)
            'y_data': combineEvent,                                  # (Trial,)
            'probs': np.zeros((nTrials, self.nClasses)),             # (Trial, n_classes) 預設全 0
            'metadata': {
                'sfreq': self.sfreq,
                'channels': self.chooseChannelNames,
                'n_classes': self.nClasses,
                'source_csvs': csvPaths,
                'subject': self.settingData["data_info"].get("SUBJECT_NAME", "unknown"),
                'normalization_method': self.settingData["data_normalization_standardization"].get("method", "none"),
                'time_jitter': self.settingData.get("time_jitter_augmentation", {}).get("use", False),
                'causal_filter': True,
            }
        }

        print(f"\n{'='*60}")
        print(f"[INFO] 處理完成 (v2 — 對齊即時端):")
        print(f"  x_data: {targetData.shape}")
        print(f"  y_data: {combineEvent.shape}, classes={np.unique(combineEvent)}")
        print(f"  probs:  {self.processedData['probs'].shape}")
        print(f"  normalization: {self.processedData['metadata']['normalization_method']}")
        print(f"  time_jitter: {self.processedData['metadata']['time_jitter']}")
        print(f"  causal_filter: {self.processedData['metadata']['causal_filter']}")
        print(f"{'='*60}\n")

        return self.processedData

    # ═══════════════════════════════════════════
    # 儲存
    # ═══════════════════════════════════════════

    def savePt(self, outputPath):
        """
        將處理結果儲存為 .pt 檔案

        格式與 2026_BCI_- 的 data_process_np._save_as_pt 相容

        Args:
            outputPath: 輸出 .pt 檔案路徑
        """
        if self.processedData is None:
            raise RuntimeError("請先執行 process() 再儲存")

        # 轉為 Tensor
        saveDict = {
            'x_data': torch.from_numpy(self.processedData['x_data']).float(),
            'y_data': torch.from_numpy(self.processedData['y_data']).long(),
            'probs': torch.from_numpy(self.processedData['probs']).float(),
            'metadata': self.processedData['metadata'],
        }

        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        torch.save(saveDict, outputPath)

        print(f"[INFO] 已儲存: {outputPath}")
        print(f"  x_data: {saveDict['x_data'].shape}")
        print(f"  y_data: {saveDict['y_data'].shape}")
        print(f"  probs:  {saveDict['probs'].shape}")

    # ═══════════════════════════════════════════
    # 驗證
    # ═══════════════════════════════════════════

    @staticmethod
    def verifyPt(ptPath):
        """
        驗證 .pt 檔案格式是否正確

        Args:
            ptPath: .pt 檔案路徑
        """
        data = torch.load(ptPath, weights_only=False)

        print(f"\n[Verify] {ptPath}")
        print(f"  keys: {list(data.keys())}")
        print(f"  x_data: {data['x_data'].shape}, dtype={data['x_data'].dtype}")
        print(f"  y_data: {data['y_data'].shape}, dtype={data['y_data'].dtype}")
        print(f"  probs:  {data['probs'].shape}, dtype={data['probs'].dtype}")
        print(f"  metadata: {data.get('metadata', 'N/A')}")

        # 基本檢查
        assert data['x_data'].dim() == 3, f"x_data 應為 3D, 但得到 {data['x_data'].dim()}D"
        assert data['y_data'].dim() == 1, f"y_data 應為 1D, 但得到 {data['y_data'].dim()}D"
        assert data['probs'].dim() == 2, f"probs 應為 2D, 但得到 {data['probs'].dim()}D"
        assert data['x_data'].shape[0] == data['y_data'].shape[0], "x_data 和 y_data trial 數不一致"
        assert data['x_data'].shape[0] == data['probs'].shape[0], "x_data 和 probs trial 數不一致"

        print(f"  ✅ 格式驗證通過")
        return data



# ═══════════════════════════════════════════
# parse_args
# ═══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="MIEXP 資料前處理 Pipeline: .csv → .pt")
    parser.add_argument(
        "--setting_path", 
        type=str,
        default=os.path.join(BASE_DIR, "json", "setting.json"),
        help="setting.json 路徑（預設: data_process/json/setting.json）")
    parser.add_argument(
        "--background_path", 
        type=str,
        default="../../global_json/background.json",
        help="background.json 路徑（預設: ../../global_json/background.json)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default=None,
        help="輸出 .pt 路徑（預設根據 subject_name 自動產生）"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="產出後驗證 .pt 格式"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=60,
        help="Random seed"
    )
    
    return parser.parse_args()


# ═══════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    log_file, original_stdout = setup_dual_logging()

    try:
        # 建立 Processor（不指定路徑時使用 BASE_DIR 相對預設值）
        processor = MIEXPDataProcessor(
            settingPath=args.setting_path,
            backgroundPath=args.background_path,
            seed=args.seed,
        )

        # 執行前處理
        result = processor.process()

        # 決定輸出路徑
        outputPath = args.output
        if outputPath is None:
            subject = result['metadata']['subject']
            outputPath = os.path.join(BASE_DIR, "output", f"{subject}_processed.pt")

        # 儲存
        processor.savePt(outputPath)

        # 驗證
        if args.verify:
            MIEXPDataProcessor.verifyPt(outputPath)

    finally:
        # 確保程式結束或報錯時，能正確關閉檔案並恢復原本的 stdout
        sys.stdout = original_stdout
        log_file.close()
