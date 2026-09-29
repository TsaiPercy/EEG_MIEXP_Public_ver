

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC                         # 只用這個是二分類
from sklearn.multiclass import OneVsRestClassifier  # 加這個才多分類
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import classification_report



from src import classifier as clsier
from src import EEGNet as ent
from src import embedding as ebd
from src import norm as nm
from src import window_slice as ws


def process_before_fold(X, y):


    X_copy = X.copy()
    y_copy = y.copy()

    print("X_copy NaN:", np.isnan(X_copy).any())
    print("X_copy Inf:", np.isinf(X_copy).any())

    print("y_copy NaN:", np.isnan(y_copy).any())
    print("y_copy Inf:", np.isinf(y_copy).any())

    X_copy = np.nan_to_num(X_copy, nan=1e-10, posinf=1e-10, neginf=1e-10)
    y_copy = np.nan_to_num(y_copy, nan=0.0, posinf=0.0, neginf=0.0)

    print(X_copy.shape)
    print(y_copy.shape)

    '''
    計算每一類別有多少筆 trial

    再檢查該類別 trial 的標準差（若太小表示該 trial 幾乎是常數，沒訊號變化）

    若小於 1e-12 → 印出提醒要移除
    '''
    for cls in np.unique(y_copy):
        print(f"class {cls}:", np.sum(y_copy==cls), "trials")

        cls_std = np.std(X_copy[y_copy==cls], axis=(1,2))
        if np.any(cls_std < 1e-12):
            print(f"Class {cls} has near-zero trials, remove them")


    '''
    for cls in np.unique(y_copy):
        cov = np.cov(X_copy[y_copy==cls].reshape(len(X_copy[y_copy==cls]), -1).T)
        print(f"class {cls} cov rank: {np.linalg.matrix_rank(cov)}")
    '''

    print("X_final NaN:", np.isnan(X_copy).any())
    print("X_final Inf:", np.isinf(X_copy).any())

    print("y_final NaN:", np.isnan(y_copy).any())
    print("y_final Inf:", np.isinf(y_copy).any())

    return X_copy, y_copy


# 需維修
def classification_ML(X,
                        y,
                        sfreq,
                        classifier,
                        arg_dict
                    ):
    """
    X = (trail, choose_channel, time)
    y = (trail, 3)
    """

    """
    classifier ====> "SVM" or "LDA"
    n_splits 是 K fold 的 K
    if_normailze, true 會做
    
    embedding_method
    "all_method": [
        "CSP",
        "Riemannian",
        "Hjorth",
        "DE",
        "PSD"
    ]
            
    CSP_n_components 是 CSP 成分數        
    """

    n_splits = arg_dict["n_splits"]
    if_normailze = arg_dict["if_normailze"]
    embedding_method_dict = arg_dict["embedding_method"]
    embedding_method_list = embedding_method_dict["current_using"]
    
    ###########################################################
    # K fold
    X, y = process_before_fold(X, y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    #############################################################
    # 可比較的 kernel 種類, SVM use
    if classifier == "SVM":
        kernels = arg_dict["kernels"]
        acc_results = {k: [] for k in kernels}


    # LDA use
    all_acc = []
    
    #############################################################
    # K fold
    
    print(f"\n使用 {embedding_method_dict["current_using"]} 特徵抽取")
    print(f"使用 {classifier} classifier")


    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== Fold {fold+1} =====")

        X_train, X_test = X[train_idx], X[test_idx] # (trial, channel, time)
        y_train, y_test = y[train_idx], y[test_idx] # (trial,)


        ##################################################################
        # X zscore 調整
        if if_normailze:

            ALL_MARKER_MEAN = []
            ALL_MARKER_STD = []

            '''
            for marker in np.unique(y_train):
            
                marker_idx_train = (y_train == marker)

                TRAIL_MEAN = X_train[marker_idx_train, :, :].mean(axis=2, keepdims=True)
                TRAIL_STD = X_train[marker_idx_train, :, :].std(axis=2, keepdims=True)

                MARKER_MEAN = TRAIL_MEAN.mean(axis=0).squeeze()
                MARKER_STD = TRAIL_STD.mean(axis=0).squeeze()

                X_train[marker_idx_train, :, :] -= MARKER_MEAN.reshape(1, X_train.shape[1], 1)
                X_train[marker_idx_train, :, :] /= (MARKER_STD.reshape(1, X_train.shape[1], 1) + 1e-10)

                marker_idx_test = (y_test == marker)
                X_test[marker_idx_test, :, :] -= MARKER_MEAN.reshape(1, X_test.shape[1], 1)
                X_test[marker_idx_test, :, :] /= (MARKER_STD.reshape(1, X_test.shape[1], 1) + 1e-10)

                ALL_MARKER_MEAN.append(MARKER_MEAN.tolist())
                ALL_MARKER_STD.append(MARKER_STD.tolist())
            '''
            TRAIL_MEAN = X_train.mean(axis=2, keepdims=True)
            TRAIL_STD = X_train.std(axis=2, keepdims=True)

            MARKER_MEAN = TRAIL_MEAN.mean(axis=0).squeeze()
            MARKER_STD = TRAIL_STD.mean(axis=0).squeeze()

            X_train -= MARKER_MEAN.reshape(1, X_train.shape[1], 1)
            X_train /= (MARKER_STD.reshape(1, X_train.shape[1], 1) + 1e-10)

            X_test -= MARKER_MEAN.reshape(1, X_test.shape[1], 1)
            X_test /= (MARKER_STD.reshape(1, X_test.shape[1], 1) + 1e-10)

            ALL_MARKER_MEAN = MARKER_MEAN.tolist()
            ALL_MARKER_STD = MARKER_STD.tolist()
        
        ##################################################################

        print(f"X_train.shape: {X_train.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"X_test.shape: {X_test.shape}")
        print(f"y_test.shape: {y_test.shape}")

        ##################################################################
        # embedding

        ## X_train_emb_combine = (trails, total_feature_num)
        ## X_test_emb_combine = (trails, total_feature_num)
        
        X_train_emb_combine, X_test_emb_combine= ebd.embedding(X_train, 
                                                                y_train, 
                                                                X_test,
                                                                sfreq,
                                                                embedding_method_dict,
                                                            )

        print(f"X_train_emb_combine.shape: {X_train_emb_combine.shape}")
        print(f"X_test_emb_combine.shape: {X_test_emb_combine.shape}")


        ######################################################################################
        # classifier


        if classifier == "LDA":
            all_acc.append(clsier.LDA(X_train_emb_combine, y_train,
                                         X_test_emb_combine, y_test))
            

        elif classifier == "SVM":
            acc_results = clsier.SVM(X_train_emb_combine, y_train, 
                                     X_test_emb_combine, y_test,
                                     kernels,
                                     acc_results)
            
    ##############################################################################
    # result

    result_dict = {}

    if classifier == "LDA":

        AVE_ACC = np.mean(all_acc)
        HIGHEST_ACC = np.max(all_acc)
        LOWEST_ACC = np.min(all_acc)

        print("\n########################################################################")
        print(f"LDA result\n")
        print("平均準確率:", AVE_ACC)
        print("最高準確率:", HIGHEST_ACC)
        print("最低準確率:", LOWEST_ACC)
        print("##########################################################################\n")

        result_dict["FOLDS_ACC"] = all_acc
        result_dict["AVE_ACC"] = AVE_ACC
        result_dict["HIGHEST_ACC"] = HIGHEST_ACC
        result_dict["LOWEST_ACC"] = LOWEST_ACC

    elif classifier == "SVM":


        print("\n########################################################################")
        print("SVM result\n")
        print("===== 準確率 =====")

        all_mean_acc = []
        all_max_acc = []
        all_min_acc = []

        for kernel in kernels:

            mean_acc = np.mean(acc_results[kernel])
            max_acc = np.max(acc_results[kernel])
            min_acc = np.min(acc_results[kernel])

            all_mean_acc.append(mean_acc)
            all_max_acc.append(max_acc)
            all_min_acc.append(min_acc)

            # std_acc = np.std(acc_results[kernel]) # , 標準差: {std_acc:.4f}")
            print(f"SVM ({kernel}) -> 平均: {mean_acc:.4f}, 最高: {max_acc:.4f}, 最低: {min_acc:.4f}") 


        print("########################################################################\n")

        AVE_HIGHEST_ACC = np.max(all_mean_acc)
        HIGHEST_ACC = np.max(all_max_acc)
        LOWEST_ACC = np.min(all_min_acc)

        AVE_HIGHEST_kernel = kernels[np.argmax(all_mean_acc)]
        HI_kernel = kernels[np.argmax(all_max_acc)]
        LO_kernel = kernels[np.argmin(all_min_acc)]

        result_dict["FOLDS_ACC"] = acc_results
        result_dict["AVE_ACC"] = all_mean_acc
        result_dict["HIGHEST_ACC_KERNELS"] = all_max_acc
        result_dict["LOWEST_ACC_KERNELS"] = all_min_acc

        result_dict["AVE_HIGHEST_ACC"] = AVE_HIGHEST_ACC
        result_dict["AVE_HIGHEST_kernel"] = AVE_HIGHEST_kernel
        result_dict["HIGHEST_ACC"] = HIGHEST_ACC
        result_dict["HI_kernel"] = HI_kernel
        result_dict["LOWEST_ACC"] = LOWEST_ACC
        result_dict["LO_kernel"] = LO_kernel


    return result_dict


# train 跟 calib 用
def classification_DL(X, 
                        y,
                        sfreq,
                        classifier,
                        arg_dict,
                        ):
    """
    X = (trail, choose_channel, time)
    y = (trail, 3)

    classifier ====> 目前只有 "EEGNet"
    n_splits 是 K fold 的 K
    if_normailze, true 會做
    """

    n_splits = arg_dict["n_splits"]
    if_normailze = arg_dict["if_normailze"]
    

    arg_dict["ALL_MARKER_MEAN"] = []
    arg_dict["ALL_MARKER_STD"] = []

    all_acc = []
    all_pred = []
    all_label = []
    best_acc_all = 0.0
    best_LOSS_all = 100.0

    print(f"使用 {classifier} classifier")
    
    ###########################################################
    # K fold

    # 2026/01/23 先分段切，先不要用 fold
    """

    X, y = process_before_fold(X, y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    """
        
    TSP = int(X.shape[0] * 3 // 4)

    train_idx = np.arange(TSP)
    valid_idx = np.arange(TSP, X.shape[0])


    for fold in range(1):
        
        print(f"\n===== Fold {fold+1} =====")
        # print(f"train_idx: {train_idx}")
        # print(f"valid_idx: {valid_idx}")

        X_train, X_valid = X[train_idx], X[valid_idx] # (trial, channel, time)
        y_train, y_valid = y[train_idx], y[valid_idx] # (trial,)

        ##################################################################

        print(f"X_train.shape: {X_train.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"X_valid.shape: {X_valid.shape}")
        print(f"y_valid.shape: {y_valid.shape}")
        print()

        ## window slice --> 應該要在 train valid set 分離之後，避免同 trail 相似性問題

        # 單位 秒
        window_time = 1
        step_time = 0.1
        X_train, y_train = ws.window_slice(X_train,
                                                    y_train,
                                                    sfreq=sfreq,
                                                    window_time=window_time, 
                                                    step_time=step_time
                                                    )

        X_valid, y_valid = ws.window_slice(X_valid,
                                                    y_valid,
                                                    sfreq=sfreq,
                                                    window_time=window_time, 
                                                    step_time=step_time
                                                    )

        print(f"X_train_ws.shape: {X_train.shape}")
        print(f"y_train_ws.shape: {y_train.shape}")
        print(f"X_valid_ws.shape: {X_valid.shape}")
        print(f"y_valid_ws.shape: {y_valid.shape}")
        print()

        ###################################################################

        X_train_norm = np.array([0])
        y_train_norm = np.array([0])
        
        ##################################################################
        # X zscore 調整

        if if_normailze:
            best_LOSS_ori = best_LOSS_all
            best_acc_ori = best_acc_all

            ALL_MARKER_MEAN = []
            ALL_MARKER_STD = []

            X_train_record = []
            y_train_record = []

            
            for marker in np.unique(y_train):
            
                marker_idx_train = (y_train == marker)

                TRAIL_MEAN = X_train[marker_idx_train, :, :].mean(axis=2, keepdims=True)
                TRAIL_STD = X_train[marker_idx_train, :, :].std(axis=2, keepdims=True)

                MARKER_MEAN = TRAIL_MEAN.mean(axis=0).squeeze()
                MARKER_STD = TRAIL_STD.mean(axis=0).squeeze()

                ALL_MARKER_MEAN.append(MARKER_MEAN.tolist())
                ALL_MARKER_STD.append(MARKER_STD.tolist())

            
            arg_dict["ALL_MARKER_MEAN"] = ALL_MARKER_MEAN
            arg_dict["ALL_MARKER_STD"] = ALL_MARKER_STD


            
            # X_train_record = (nv=4, trail, choose_channel, time)
            # X_train_norm = (trail*nv, choose_channel, time)
            

            X_train_norm, y_train_norm = nm.norm_per_marker(X_train, y_train, ALL_MARKER_MEAN, ALL_MARKER_STD)
            
            print(f"X_train_norm.shape: {X_train_norm.shape}")
            print(f"y_train_norm.shape: {y_train_norm.shape}")
            
            

        ######################################################################################
        # classifier
        # return
        print("\n########################################################################")

        if classifier == "EEGNet":
            
            X_valid_EEGNet = X_valid.copy()
            y_valid_EEGNet = y_valid.copy()

            if if_normailze:
                X_train_EEGNet = X_train_norm.copy()
                y_train_EEGNet = y_train_norm.copy()
            else:
                X_train_EEGNet = X_train.copy()
                y_train_EEGNet = y_train.copy()

                

            print(f"X_train_EEGNet.shape: {X_train_EEGNet.shape}")
            print(f"y_train_EEGNet.shape: {y_train_EEGNet.shape}")
            print(f"X_valid_EEGNet.shape: {X_valid_EEGNet.shape}")
            print(f"y_valid_EEGNet.shape: {y_valid_EEGNet.shape}")
            print()

            ##################################################################

            fold_highest_acc, fold_preds, fold_labels, best_acc_all, best_LOSS_all = ent.EEGNet_setting_using(
                                                                                X_train_EEGNet, 
                                                                                y_train_EEGNet,
                                                                                X_valid_EEGNet, 
                                                                                y_valid_EEGNet,
                                                                                arg_dict,
                                                                                best_acc_all,
                                                                                best_LOSS_all
                                                                            ) # '''

            all_acc.append(fold_highest_acc)
            all_pred.extend(fold_preds)
            all_label.extend(fold_labels)

        if if_normailze:
            
            # if best_acc_ori != best_acc_all:
            if best_LOSS_ori != best_LOSS_all:
                arg_dict["ALL_MARKER_MEAN"] = ALL_MARKER_MEAN
                arg_dict["ALL_MARKER_STD"] = ALL_MARKER_STD

        else:
            arg_dict["ALL_MARKER_MEAN"] = []
            arg_dict["ALL_MARKER_STD"] = []
        
        
        print(f"EEGNet result merge each fold\n")


    ##############################################################################
    # result

    AVE_ACC = np.mean(all_acc)
    HIGHEST_ACC = np.max(all_acc)
    LOWEST_ACC = np.min(all_acc)
    c_m = confusion_matrix(all_label, all_pred)

    
    print("平均準確率:", AVE_ACC)
    print("最高準確率:", HIGHEST_ACC)
    print("最低準確率:", LOWEST_ACC)
    print(c_m)
    print(classification_report(all_label, all_pred, digits=4))
    print("########################################################################\n")

    result_dict = {}
    result_dict["FOLDS_HIGHEST_ACC"] = all_acc
    result_dict["AVE_ACC"] = AVE_ACC
    result_dict["HIGHEST_ACC"] = HIGHEST_ACC
    result_dict["LOWEST_ACC"] = LOWEST_ACC
    result_dict["confusion_matrix"] = c_m.tolist()



    return result_dict


# test 用
def test_DL(X_test, 
            y_test,
            classifier,
            arg_dict,
            ):
    """
    X_test = (trail*nwindow, choose_channel, time)
    y_test = (trail, 3)

    classifier 目前只有 "EEGNet"
    """

    print(f"使用 {classifier} classifier") 

    ######################################################################################
    # classifier

    print("\n########################################################################")

    if classifier == "EEGNet":

        X_test_EEGNet = X_test.copy()
        y_test_EEGNet = y_test.copy()

        print(f"X_test_EEGNet.shape: {X_test_EEGNet.shape}")
        print(f"y_test_EEGNet.shape: {y_test_EEGNet.shape}")
       

        ##################################################################
        # 純 test

        acc, all_pred, all_label = ent.testing(X_test_EEGNet,
                                                y_test_EEGNet,
                                                arg_dict=arg_dict,
                                            )

        c_m = confusion_matrix(all_label, all_pred)
        
        print(f"EEGNet result TEST\n")



    ##############################################################################
    # result

    print("準確率:", acc)
    print(c_m)
    print(classification_report(all_label, all_pred, digits=4))
    print("########################################################################\n")

    result_dict = {}
    result_dict["ACC"] = acc
    result_dict["confusion_matrix"] = c_m.tolist()

    return result_dict
            
