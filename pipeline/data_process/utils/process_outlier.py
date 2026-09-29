import numpy as np



def detect_outliers(data, method):#, threshold, n_neighbors):
    """
    判斷 list 或 1D numpy array 的每個值是否為離群值

    Parameters
    ----------
    data : list or np.ndarray
        一維資料

    method : str
        離群值檢測方法: 'iqr', 'zscore', 'lof'

    threshold : float
       IQR (1.5) 或 z-score (3) 的閾值

    n_neighbors : int
        LOF 方法使用的鄰居數量

    Returns
    -------
    outlier_flags : np.ndarray
        與 data 長度相同，True 表示離群值，False 表示正常
    """

    data = np.array(data).flatten()

    if method.lower() == 'iqr':
        threshold = 1.5                     # 推薦用
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_flags = (data < lower_bound) | (data > upper_bound)

    elif method.lower() == 'zscore':
        threshold = 3                       # 推薦用
        mean = np.mean(data)
        std = np.std(data)
        z = (data - mean) / std
        outlier_flags = np.abs(z) > threshold

        '''
        # lof 適用多維
        elif method.lower() == 'lof':
        data_reshaped = data.reshape(-1, 1)  # 轉成 n_samples x n_features
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        outlier_flags = lof.fit_predict(data_reshaped) == -1
        '''

    else:
        raise ValueError("method must be 'iqr', 'zscore', or 'lof'")

    return outlier_flags

def replace_outliers(data, outlier_mask, replace_strategy):
    """
    替換離群值（只改 outlier 部分）

    參數:
    ----------
    data : list or np.ndarray
        原始資料

    replace_strategy : str
        'median' (全域中位數), 'mean' (全域平均), 'neighbor' (用鄰近值平滑)

    outlier_flags : np.ndarray
        與 data 長度相同，True 表示離群值，False 表示正常
    """

    data = np.array(data, dtype=float)
    clean_data = data.copy()

    if replace_strategy == 'median':
        replacement = np.nanmedian(data[~outlier_mask])
        clean_data[outlier_mask] = replacement

    elif replace_strategy == 'mean':
        replacement = np.nanmean(data[~outlier_mask])
        clean_data[outlier_mask] = replacement

    elif replace_strategy == 'neighbor':

        for i in np.where(outlier_mask)[0]:
            
            left = data[i - 1] if i > 0 else data[i]
            right = data[i + 1] if i < len(data) - 1 else data[i]
            clean_data[i] = np.nanmean([left, right])

    else:
        raise ValueError(f"未知替換策略: {replace_strategy}")

    return clean_data


def process_outlier_multi(DATAs_np, all_channels_names, detect_method, replace_strategy, prt_info):

    """
    針對 each channel 做
    """
    print()
    print("#########################")
    print("now process_outlier_multi")
    print()
    print(f"detect_method: {detect_method}")
    print(f"replace_strategy: {replace_strategy}")

    DATAs_np_copy = DATAs_np.copy()

    for run in range(len(DATAs_np_copy)):

        DATA_np = DATAs_np_copy[run]
        print()
        print(f"run: {run}")
        print()

        for chl in range(DATA_np.shape[0]):
        
            dt = DATA_np[chl]

            # dt = (, time)

            outlier_flags = detect_outliers(dt, method=detect_method)#, threshold=3, n_neighbors=20)

            if prt_info:
                '''
                該 .csv 資料在 each channel 的 離群值數量
                '''
                print(f"chl: {all_channels_names[chl]}, num_of_outer_{detect_method}: {sum(outlier_flags)}")

            DATA_np[chl] = replace_outliers(dt, outlier_flags, replace_strategy=replace_strategy)

        DATAs_np_copy[run] = DATA_np


        
    print("#########################")
    print()


    return DATAs_np_copy