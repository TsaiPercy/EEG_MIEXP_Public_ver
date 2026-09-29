import numpy as np

"""
concat 4 種 normalize 後資料
"""

def norm_per_marker(X, y, ALL_MARKER_MEAN, ALL_MARKER_STD):
    '''
    ALL_MARKER_MEAN: list, 每個 class 的 mean
    ALL_MARKER_STD: list, 每個 class 的 std
    '''

    X_record = []
    y_record = []

    n_classes = len(ALL_MARKER_MEAN)

    for i in range(n_classes):
    
        MARKER_MEAN = np.array(ALL_MARKER_MEAN[i])
        MARKER_STD = np.array(ALL_MARKER_STD[i])
    
        X_record.append( (X[:, :, :] - MARKER_MEAN.reshape(1, X.shape[1], 1)) / (MARKER_STD.reshape(1, X.shape[1], 1) + 1e-10) )
        y_record.append( y )

    '''
    # X_record = (nv=4, trail, choose_channel, time)
    # X_norm = (trail*nv, choose_channel, time)
    '''

    X_norm = np.concatenate(X_record)
    y_norm = np.concatenate(y_record)
    

    print(f"X_norm.shape: {X_norm.shape}")
    print(f"y_norm.shape: {y_norm.shape}")

    return X_norm, y_norm


def norm_once(X):

    #################################################################################################
            
    '''
    for marker in np.unique(y_train):
    
        marker_idx_train = (y_train == marker)

        TRAIL_MEAN = X_train[marker_idx_train, :, :].mean(axis=2, keepdims=True)
        TRAIL_STD = X_train[marker_idx_train, :, :].std(axis=2, keepdims=True)

        MARKER_MEAN = TRAIL_MEAN.mean(axis=0).squeeze()
        MARKER_STD = TRAIL_STD.mean(axis=0).squeeze()

        X_train[marker_idx_train, :, :] -= MARKER_MEAN.reshape(1, X_train.shape[1], 1)
        X_train[marker_idx_train, :, :] /= (MARKER_STD.reshape(1, X_train.shape[1], 1) + 1e-10)

        marker_idx_valid = (y_valid == marker)
        X_valid[marker_idx_valid, :, :] -= MARKER_MEAN.reshape(1, X_valid.shape[1], 1)
        X_valid[marker_idx_valid, :, :] /= (MARKER_STD.reshape(1, X_valid.shape[1], 1) + 1e-10)

        ALL_MARKER_MEAN.append(MARKER_MEAN.tolist())
        ALL_MARKER_STD.append(MARKER_STD.tolist())

    

    TRAIL_MEAN = X_train.mean(axis=2, keepdims=True)
    TRAIL_STD = X_train.std(axis=2, keepdims=True)

    MARKER_MEAN = TRAIL_MEAN.mean(axis=0).squeeze()
    MARKER_STD = TRAIL_STD.mean(axis=0).squeeze()

    X_train -= MARKER_MEAN.reshape(1, X_train.shape[1], 1)
    X_train /= (MARKER_STD.reshape(1, X_train.shape[1], 1) + 1e-10)

    X_valid -= MARKER_MEAN.reshape(1, X_valid.shape[1], 1)
    X_valid /= (MARKER_STD.reshape(1, X_valid.shape[1], 1) + 1e-10)

    ALL_MARKER_MEAN = MARKER_MEAN.tolist()
    ALL_MARKER_STD = MARKER_STD.tolist()
    '''

    print("norm_once current no use")
    return