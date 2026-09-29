import numpy as np

# 2026/01/23 改成維持 run 順序排序
def window_slice(X, y, sfreq, window_time=1, step_time=0.1):

    X_record = []
    y_record = []

    print(f"X.shape = {X.shape}")


    # 區間取 [:,:,500] 一秒
    window = int(sfreq * window_time)

    # 每次移動 0.1 秒
    step = int(sfreq * step_time)

    # 移動次數
    num_step = int((X.shape[2] - window) // step) 

    for t in range(X.shape[0]):

        X_trail = X[t]
        y_trail = y[t]

        for i in range(num_step):
            X_window = X_trail[:,  i * step : i * step + window]
            y_window = y_trail

            X_record.append(X_window)
            y_record.append(y_window)

    X_ws = np.array(X_record)
    y_ws = np.array(y_record)

    print(f"X_ws: {X_ws.shape}")
    print(f"y_ws: {y_ws.shape}")


    return X_ws, y_ws