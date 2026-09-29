import numpy as np


"""
對 each channel 做 normalize 跟 standardization

支援方法:
  - Min_Max: 通道級 min-max scaling
  - z_score: 通道級 z-score standardization
  - zero_mean: 通道級去均值 (legacy，不建議用於離線-線上對齊)
  - per_channel_demean: 每個 trial/window 獨立去均值 (對齊即時端)
  - ema: 指數移動平均正規化 (備用進階方案)
"""


def min_max_std(X):
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)

    # 避免除以 0
    X_range = X_max - X_min
    X_range[X_range == 0] = 1e-8

    X_minmax = (X - X_min) / X_range

    return X_minmax

def z_score_std(X):
    X_mean = X.mean(axis=1, keepdims=True)
    X_std  = X.std(axis=1, keepdims=True)

    # 避免除以 0
    X_std[X_std == 0] = 1e-8

    X_zscore = (X - X_mean) / X_std

    return X_zscore


def zero_mean_std(X):
    
    MEAN = X.mean(axis=1, keepdims=True)
    X_zero_mean = X - MEAN

    return X_zero_mean


def per_channel_demean(X):
    """
    Per-channel demean: 對時間軸 (最後一個維度) 去均值

    與即時端 eeg_predictor.py / preprocess.py 的 each_channel_demean 行為完全一致。
    即時端做法: data - np.mean(data, axis=1, keepdims=True)
    
    支援:
      - 2D input (channel, time): 對每個 channel 獨立去均值
      - 3D input (trial, channel, time): 對每個 trial 的每個 channel 獨立去均值

    Args:
        X: np.ndarray, shape (channel, time) 或 (trial, channel, time)

    Returns:
        X_demeaned: 去均值後的資料，shape 不變
    """
    X_demeaned = X - X.mean(axis=-1, keepdims=True)
    print(f"[DEBUG] per_channel_demean: input shape={X.shape}, "
          f"mean before={np.abs(X.mean()):.6f}, mean after={np.abs(X_demeaned.mean()):.6f}")
    return X_demeaned


def exponential_moving_average(X, alpha=0.01):
    """
    指數移動平均 (EMA) 正規化 — 備用進階方案

    使用 EMA 追蹤每個 channel 的 running mean，
    模擬即時端「前 N 秒」的 baseline 校準行為。

    Args:
        X: np.ndarray, shape (channel, time) 或 (trial, channel, time)
        alpha: EMA 衰減係數，越小越平滑 (預設 0.01)

    Returns:
        X_ema_norm: EMA 正規化後的資料
    """
    if X.ndim == 2:
        # (channel, time)
        channelCount, timeLen = X.shape
        emaMean = np.zeros(channelCount)
        emaVar = np.ones(channelCount)
        result = np.zeros_like(X)

        for timeIdx in range(timeLen):
            sample = X[:, timeIdx]
            emaMean = alpha * sample + (1 - alpha) * emaMean
            emaVar = alpha * (sample - emaMean) ** 2 + (1 - alpha) * emaVar
            emaStd = np.sqrt(emaVar)
            emaStd[emaStd < 1e-8] = 1e-8
            result[:, timeIdx] = (sample - emaMean) / emaStd

        print(f"[DEBUG] exponential_moving_average: shape={X.shape}, alpha={alpha}")
        return result

    elif X.ndim == 3:
        # (trial, channel, time) — 對每個 trial 獨立做
        result = np.zeros_like(X)
        for trialIdx in range(X.shape[0]):
            result[trialIdx] = exponential_moving_average(X[trialIdx], alpha=alpha)
        print(f"[DEBUG] exponential_moving_average: {X.shape[0]} trials processed")
        return result
    else:
        raise ValueError(f"EMA 正規化不支援 {X.ndim}D 輸入")


def norm_std_func(X, method="per_channel_demean", **kwargs):
    """
    正規化/標準化路由函數

    Args:
        X: np.ndarray
        method: 正規化方法名稱
        **kwargs: 傳遞給特定方法的額外參數 (如 EMA 的 alpha)
    """
    if method == "Min_Max":
        X_norm = min_max_std(X)
    elif method == "z_score":
        X_norm = z_score_std(X)
    elif method == "zero_mean":
        X_norm = zero_mean_std(X)
    elif method == "per_channel_demean":
        X_norm = per_channel_demean(X)
    elif method == "ema":
        alpha = kwargs.get("alpha", 0.01)
        X_norm = exponential_moving_average(X, alpha=alpha)
    else:
        raise ValueError(f"Unknown normalization/standardization method: {method}")

    return X_norm