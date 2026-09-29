
import numpy as np


from mne.decoding import CSP

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from scipy.signal import butter, filtfilt, welch 
from sklearn.decomposition import PCA


def do_PCA(X_ori, n_components=100):  # 只保留前100維主要成分

    X = X_ori
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=n_components)  
    X_pca = pca.fit_transform(X_flat)

    return X_pca

def embedding(X_train, 
                y_train, 
                X_test,
                sfreq,
                embedding_method_dict,
                prt_info=False
            ):

    """
    X_train = (trail, choose_channel, time)
    y_train = (trail, 3)
    X_test = (trail, choose_channel, time)

    embedding_method_dict
    
    spatial_emb_method ====> "CSP", "Riemannian"
    temporal_freq_emb_method ====> "Hjorth", "DE", "PSD"


    return: (trails, total_feature_num)

    """

    embedding_method_list = embedding_method_dict["current_using"]

    if len(embedding_method_list) == 0:

        X_train_emb_combine = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
        X_test_emb_combine = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

        if prt_info:
            print(f"X_train_emb_combine.shape: {X_train_emb_combine.shape}")
            print(f"X_test_emb_combine.shape: {X_test_emb_combine.shape}")

        return X_train_emb_combine, X_test_emb_combine


    spatial_emb_method = []
    temporal_freq_emb_method = []

    for method in embedding_method_list:
        if method == "CSP" or method == "Riemannian":
            spatial_emb_method.append(method)

        if method == "PSD" or method == "DE" or method == "Hjorth":
            temporal_freq_emb_method.append(method)

    
    X_train_emb_list = []
    X_test_emb_list = []


    for method in spatial_emb_method:

        X_train_emb, X_test_emb = spatial_features_embedding(X_train, 
                                                                y_train, 
                                                                X_test,
                                                                embedding_method=method,
                                                                aug_dict=embedding_method_dict[method]
                                                                )
        
        X_train_emb_list.append(X_train_emb)
        X_test_emb_list.append(X_test_emb)



    for method in temporal_freq_emb_method:

        X_train_emb = temporal_freq_features_embedding(X_train,
                                                        sfreq,
                                                        embedding_method=method,
                                                        aug_dict=embedding_method_dict[method]
                                                        )

        X_test_emb = temporal_freq_features_embedding(X_test,
                                                        sfreq,
                                                        embedding_method=method,
                                                        aug_dict=embedding_method_dict[method]
                                                        )
        X_train_emb_list.append(X_train_emb)
        X_test_emb_list.append(X_test_emb)


    # X_train_emb_combine = (trails, total_feature_num)
    # X_test_emb_combine = (trails, total_feature_num)

    X_train_emb_combine = np.concatenate(X_train_emb_list, axis=1)
    X_test_emb_combine = np.concatenate(X_test_emb_list, axis=1)

    if prt_info:
        
        print(f"X_train_emb_combine.shape: {X_train_emb_combine.shape}")
        print(f"X_test_emb_combine.shape: {X_test_emb_combine.shape}")

    return X_train_emb_combine, X_test_emb_combine



def spatial_features_embedding(X_train, 
                                y_train, 
                                X_test,
                                embedding_method,
                                aug_dict
                                ):   
    """
    X_train = (trail, choose_channel, time)
    y_train = (trail, 3)
    X_test = (trail, choose_channel, time)

    embedding_method ====> "CSP" or "Riemannian"

    return: (n_trials, num_features)

    """

    """
    # 空間域特徵（spatial features)

    ###############################################################################
    CSP(Common Spatial Pattern):從多通道 EEG 中找出能最大區分類別的空間濾波器。

    n_components=2 → 取出 2 個最具判別力的空間特徵，通常會取「對稱數量」的濾波器2、4、6

    reg='ledoit_wolf' → 正則化共變異矩陣估計（避免奇異）

    log=True → 對特徵取對數（讓分佈更穩定）

    norm_trace=True → 正規化共變異矩陣的 trace
    
    -------------------------------------------------------------------------------

    Riemannian Tangent Space:

    oas 或 lwf 都會自動對共變異矩陣進行 regularization。
    避免 輸入的 共變異矩陣 (covariance matrix) 不是正定 (positive definite)。

    covs = Covariances(estimator='oas').fit_transform(X)
    covs = Covariances(estimator='lwf').fit_transform(X)

    lwf 效果好一點點
    ###################################################################################
    # """

    
    if embedding_method == "CSP":

        print("使用 CSP 特徵抽取\n")

        emb = CSP(n_components=aug_dict["n_components"], 
                    reg=aug_dict["reg"], 
                    log=aug_dict["log"], 
                    norm_trace=aug_dict["norm_trace"])
        
        X_train_emb = emb.fit_transform(X_train, y_train)
        X_test_emb = emb.transform(X_test)

        # X_train_emb = (n_trials, n_components)
        # X_test_emb = (n_trials, n_components)

        

    elif embedding_method == "Riemannian":

        print("使用 Riemannian Tangent Space 特徵抽取\n")

        estr = aug_dict["estimator"]

        # cov_train = (n_trials, n_channels, n_channels)
        # cov_test = (n_trials, n_channels, n_channels)

        cov_train = Covariances(estimator=estr).fit_transform(X_train)
        cov_test = Covariances(estimator=estr).fit_transform(X_test)

        emb = TangentSpace()
        X_train_emb = emb.fit_transform(cov_train, y_train)
        X_test_emb = emb.transform(cov_test)

        # X_train_emb = (n_trials, n_channels * (n_channels + 1) / 2)
        # X_test_emb  = (n_trials, n_channels * (n_channels + 1) / 2)

    else:
        raise ValueError("embedding_method 必須是 'CSP' 或 'Riemannian'")
    

    return X_train_emb, X_test_emb



def temporal_freq_features_embedding(X,
                                        sfreq,
                                        embedding_method,
                                        aug_dict
                                    ):   
    """
    X = (trail, choose_channel, time)

    embedding_method ====> "Hjorth", "DE", "PSD"
    return: shape = (n_trials, n_channels * features_num)
    
    """



    """
    假設 X.shape = (100 trials, 32 channels, 500 timepoints)


    features = extract_features_all(data, sfreq=250)

    features.shape = (n_trials, n_channels * 9)
    (100, 32 * 9) = (100, 288)
    """


    """
    # 時頻域特徵（temporal_freq features)

    | 特徵類型    |feature_num |
    | ------    | ------ |
    | Hjorth    | 3      |
    | DE        | 1      |
    | PSD       | 5      |
    | **total** | **9**  |

    ###############################################################################
    Hjorth:

    計算 Hjorth Parameters: activity, mobility, complexity
    signal: shape (n_times,)
    return: np.array([activity, mobility, complexity])
    
    -------------------------------------------------------------------------------
    DE:
    
    計算 Differential Entropy (DE)
    假設信號近似高斯分佈: DE = 0.5 * log(2πeσ²)
    
    --------------------------------------------------------------------------------

    PSD:

    使用 Welch 方法計算 PSD 並取五個頻帶平均功率
    頻帶: delta(1–4), theta(4–8), alpha(8–13), beta(13–30), gamma(30–40)
    
    ###################################################################################
    # """

    print(f"使用 {embedding_method} 特徵抽取\n")
    n_trials, n_channels, _ = X.shape

    ## all_features = (trail, n_channels * feature_num)
    all_features = []
    
    ## all_trail_features ====> list, len = n_trials
    ## element in all_trail_features ====> np, dim = (channel, feature_num)
    all_trail_features = []

    for trial in range(n_trials):

        ## all_channel_features ====> list, len = n_channels
        ## element in all_channel_features ====> np, dim = (feature_num,)
        
        all_channel_features = []

        for ch in range(n_channels):
            sig = X[trial, ch, :]

            if embedding_method == "Hjorth":
                # feature_num = 3
                all_channel_features.append(Hjorth(sig))


            elif embedding_method == "DE":
                # feature_num = 5
                all_channel_features.append(DE(sig, sfreq))


            elif embedding_method == "PSD":
                # feature_num = 5
                all_channel_features.append(PSD(sig, sfreq))


            else:
                raise ValueError('embedding_method 必須是 "Hjorth", "DE", "PSD"')


        all_trail_features.append(np.stack(all_channel_features, axis=0))


    # all_features = (n_trials, n_channels, feature_per_channel)
    all_features = np.stack(all_trail_features, axis=0)


    # 攤平成 (n_trials, n_channels * feature_per_channel)
    n_features = all_features.shape[1] * all_features.shape[2]


    # 以 Hjorth 舉例 
    # AMCAMC
    all_features = all_features.reshape(n_trials, n_features) 
    # AAMMCC
    # all_features = all_features.transpose(0, 2, 1).reshape(n_trials, -1)


    return all_features
    """
    ## all_features = (trail, n_channels * feature_num)


    ## all_trail_features ====> list, len = n_trials
    ## element in all_trail_features ====> np, dim = (channel, feature_num)
       

    ## all_channel_features ====> list, len = n_channels
    ## element in all_channel_features ====> np, dim = (feature_num,)
            
    
    # all_features = (n_trials, n_channels, feature_per_channel)
    # ---->
    # 攤平成 (n_trials, n_channels * feature_per_channel)

    """

    

# -----------------------------------------------
# Hjorth
# -----------------------------------------------

def Hjorth(signal):
    """
    計算 Hjorth Parameters: activity, mobility, complexity
    signal: shape (n_times,)
    return: np.array([activity, mobility, complexity])
    """
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var0 = np.var(signal)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)

    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 != 0 else 0
    complexity = np.sqrt(var2 / var1) / mobility if (var1 != 0 and mobility != 0) else 0
    return np.array([activity, mobility, complexity])


# -----------------------------------------------
# Differential Entropy (DE)
# -----------------------------------------------
def DE(signal, sfreq):
    """
    計算 Differential Entropy (DE)
    假設信號近似高斯分佈: DE = 0.5 * log(2πeσ²)
    signal: shape (n_times,)
    return: np.array([DE])
    """
    def bandpass_filter(signal, sfreq, low, high):
        """Butterworth 濾波器"""
        b, a = butter(4, [low / (sfreq/2), high / (sfreq/2)], btype='band')
        return filtfilt(b, a, signal)

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 50)
    }

    de_features = []

    for band_name, (low, high) in bands.items():

        filtered = bandpass_filter(signal, sfreq, low, high)
        var = np.var(filtered)
        de_val = 0.5 * np.log(2 * np.pi * np.e * var)
        de_features.append(de_val)

    
    de_features = np.array(de_features)
    return de_features
    
    
    var = np.var(signal)
    if var <= 0:
        return 0.0
    return np.array([0.5 * np.log(2 * np.pi * np.e * var)])



# -----------------------------------------------
# Power Spectral Density (PSD)
# -----------------------------------------------
def PSD(signal, sfreq):

    """
    使用 Welch 方法計算 PSD 並取五個頻帶平均功率
    頻帶: delta(1–4), theta(4–8), alpha(8–13), beta(13–30), gamma(30–40)
    signal: shape (n_times,)
    return: np.array([delta, theta, alpha, beta, gamma])
    """

    freqs, psd = welch(signal, sfreq, nperseg=sfreq)
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 40)}
    band_powers = []
    for (fmin, fmax) in bands.values():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_powers.append(np.mean(psd[idx]))
    return np.array(band_powers)
