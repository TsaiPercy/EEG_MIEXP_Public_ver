import os
# (removed) local QT_QPA_PLATFORM_PLUGIN_PATH setting
import argparse

import numpy as np
import pandas as pd
import json
import random
import torch

import matplotlib
matplotlib.use('Qt5Agg')  # 或 'TkAgg' 都可
import matplotlib.pyplot as plt

import mne



from sklearn.neighbors import LocalOutlierFactor


# embedding
from mne.decoding import CSP

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from scipy.signal import welch



from sklearn.model_selection import StratifiedKFold


# Classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import LabelEncoder
import pickle




'''in file import'''
import data_path_update as dpu


from src import classification_func as cf
from src import classifier as clsier
from src import config
from src import data_norm_std as dns
from src import EEGNet as ent
from src import embedding as ebd
# from src import norm as nm
from src import preprocess as pps
from src import process_outlier as pcso
from src import window_slice as ws




def get_datas(data_paths, 
                prt_info=False):
    """
    return DATAs ====> pd, dim = (time, channel)
    each element in DATAs is each run
    """
    
    DATA_descriptions = []
    DATAs = []

    for dp in data_paths:
        DATA_descriptions.append(pd.read_csv(dp,  nrows=10, on_bad_lines='skip', encoding="utf-8"))
        DATAs.append(pd.read_csv(dp, skiprows=10, encoding="utf-8"))

    if prt_info:
        print("DATAs info")
        print(f"{type(DATAs)}, len = {len(DATAs)}")

        print("DATA_descriptions")
        print(DATA_descriptions)

    return DATAs


def get_data_info(DATA, 
                    prt_info=False):
    
    columns_name = DATA.columns

    n_sample = DATA.shape[0]
    n_column = DATA.shape[1]

    if prt_info:
        print(f' columns_name = {columns_name}')
        print(f' n_sample = {n_sample}')
        print(f' n_column = {n_column}')


def process_before_mne_multi(DATAs, 
                             channel_names, 
                             prt_info=False):
    """
    DATAs ====> list
    channel_names type ====> list
    
    # element in DATAs = (time, channel)


    return DATAs_np ====> list

    # element in DATAs_np = (channel, time)
    """

    DATAs_copy = DATAs
    
    DATAs_np = []


    for DATA_copy in DATAs_copy:

        # remove_nan
        DATA_no_nan = DATA_copy.fillna(0)

        # only select channels column
        DATA_ch = DATA_no_nan[channel_names]
        
        # pd_to_np
        DATA_np = DATA_ch.to_numpy()

        # change dimension to (channel, time)
        DATA_np = DATA_np.T

        # flatten, use for marker
        if len(channel_names) == 1:
            DATA_np.reshape(-1)

        DATAs_np.append(DATA_np)


    if prt_info:
        print("DATAs_np: ", type(DATAs_np), len(DATAs_np))
        print("DATA_np: ", type(DATAs_np[0]), DATAs_np[0].shape)


    return DATAs_np


def get_event_multi(DATAs_event, choose_markers, prt_info):
    
    ##############################
    # 不想用判斷的，這邊動點手腳在event idx上，為了把 cue 座標轉成 MI 座標
    # 1s 500 sample,idx +500

    ##############################
    # events dimension = (idx, pre_val, marker_name)
    # events_multi: type: list, run * (idx, pre_val, marker_name)
    ##############################

    events_multi = []

    for DATA_event in DATAs_event:
        prev = 0
        events_list = []

        for idx, val in enumerate(DATA_event):

            if val in choose_markers:
                events_list.append([idx+500, prev, val])


        events = np.array(events_list).astype(int)

        if prt_info:
            print(f' choose_markers = {choose_markers}' )
            print(f' events.shape = {events.shape}' )
            print(f' events.dtype = {events.dtype}' )

        events_multi.append(events)

    return events_multi  


def get_mne_data_multi(DATAs, 
                       sfreq, 
                       channel_names, 
                       channel_num, 
                       prt_info=False
                       ):
                       
    get_mne_data_multi_dict = {}
    get_mne_data_multi_dict["get_mne_data_multi"] = []

    # 定義通道名稱與類型
    channel_types = ['eeg'] * channel_num

    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    raw_mne_data_multi = []

    for DATA in DATAs:
        # 建立 RawArray
        raw_mne_data = mne.io.RawArray(DATA, info) # (n_channels, n_times)

        # set channel locations
        raw_mne_data.set_montage('standard_1020')


        if prt_info:
            # 查看資訊
            print(raw_mne_data.info)

            get_mne_data_multi_dict["get_mne_data_multi"].append(raw_mne_data.info)
            #raw_mne_data.plot(n_channels=channel_num, scalings='auto', title='EEG from CSV')
            #print(raw_mne_data)


        raw_mne_data_multi.append(raw_mne_data)
        
    if prt_info:
        print("raw_mne_data_multi: ", type(raw_mne_data_multi), len(raw_mne_data_multi))

    # return raw_mne_data_multi
    return raw_mne_data_multi ,get_mne_data_multi_dict


def get_trail_data_multi(raw_mne_data_multi, 
                         events_multi, 
                         choose_markers, 
                         tmin, 
                         tmax, 
                         base_time, 
                         prt_info):

    ######################################################################
    # 已動過手腳 idx+500
    # (sec)

    # tmin: start before event

    # tmax: end after event

    # base_time should between tmin_tmax
    ######################################################################

    get_trail_data_multi_dict = {}
    get_trail_data_multi_dict["get_trail_data_multi"] = []
    get_trail_data_multi_dict["trails"] = []
    get_trail_data_multi_dict["trails.events.shape"] = []
    get_trail_data_multi_dict["trails.get_data().shape"] = []

    trails_multi = []

    for raw_mne_data, events in zip(raw_mne_data_multi, events_multi):

        trails = mne.Epochs(raw_mne_data, 
                            events, 
                            event_id=choose_markers,
                            tmin=tmin, 
                            tmax=tmax,
                            baseline=base_time, 
                            preload=True)
                            
        if prt_info:
            print(trails)                  # Epochs summary
            print(trails.events.shape)     # (n_epochs, 3)
            print(trails.get_data().shape) # (n_epochs, n_channels, n_times)

            get_trail_data_multi_dict["trails"].append(trails)
            get_trail_data_multi_dict["trails"].append(trails.events.shape)
            get_trail_data_multi_dict["trails"].append(trails.get_data().shape)

        trails_multi.append(trails)

    return trails_multi
    return trails_multi, get_trail_data_multi_dict


'''
目前未使用

def smooth_series(data, window=5, center=True):
    """
    使用滑動平均平滑資料（自動忽略 NaN 並回傳 numpy array）

    參數:
    ----------
    data : list or np.ndarray
        原始數列資料。
    window : int
        視窗大小（滑動平均的點數）。
    center : bool
        是否以當前點為中心。

    回傳:
    ----------
    smooth_data : np.ndarray
        平滑化後的數列（已自動補 NaN）。
    """
    
    # 轉成 Series 進行平滑
    smooth_data = pd.Series(data).rolling(window=window, center=center).mean()
    
    # 自動補前後 NaN：用最接近的非NaN值填補（向前或向後填）
    smooth_data = smooth_data.fillna(method='bfill').fillna(method='ffill')
    
    # 轉回 numpy array
    return smooth_data.to_numpy()

def trails_evoked_plot(trails, 
                       choose_marker_dict, 
                       choose_marker_name, 
                       choose_channels, 
                       save_img_path):
    """
    待修正
    """
    trails_class = trails[ choose_marker_dict[choose_marker_name] ]
    evoked_class = trails_class.average()

    
    fig = evoked_class.plot(picks=choose_channels, spatial_colors=True, show=False)  # ERP 波形圖
    fig.savefig(save_img_path, dpi=300)

    return trails_class, evoked_class

def mne_plot(raw_mne_data, choose_channels, prt_all):

    if prt_all:
        raw_mne_data.plot()
    else:
        raw_mne_data.plot(picks=choose_channels)

    plt.show()

def result_record_json(path, ):
    return

def result_reproduce():
    """
    for some mne that don't see in the 
    """
    return
'''

def channel_mode_choose(setting_data, background_data):
    setting_data["choose_channels"].update(background_data["channel"][setting_data["choose_channels"]["MODE"]])
    return setting_data

def seed_setting(seed=42):
    # all random seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(setting_path="./setting_json/setting.json", use_input_setting_path=True):

    # all random seed setting
    seed = 60
    seed_setting(seed)

    ####################################################################
    # setting.json 可用指令輸入

    if use_input_setting_path:
        parser = argparse.ArgumentParser(description="讀取輸入檔案名稱")
        parser.add_argument("--setting_path", type=str, default="./setting_json/setting.json", help="setting JSON")
        args = parser.parse_args()

        setting_path = args.setting_path


    prt_info = False
    print(f"setting_path: {setting_path}")

    ####################################################################
    # 設定
    
    background_path = config.background_path


    with open(setting_path, "r", encoding="utf-8") as f:
        setting_data = json.load(f)

    with open(background_path, "r", encoding="utf-8") as f:
        background_data = json.load(f)


    setting_data = channel_mode_choose(setting_data, background_data)

    #####################################################################
    # path.json update

    path_json_path = config.path_path

    if setting_data["data_info"]["IF_DATA_RELOAD"]:

        with open(path_json_path, "r", encoding="utf-8") as f:
            path_data = json.load(f)

        path_data["data_find_folder_path"] =  setting_data["data_info"]["data_find_folder_path"]
        
        with open(path_json_path, "w", encoding="utf-8") as f:
            json.dump(path_data, f, ensure_ascii=False, indent=4)

        dpu.save_csv_paths_to_json(path_json_path)

    #####################################################
    # sample freq
    sfreq = setting_data["data_info"]["sfreq"]


    #####################################################
    # get cvs to pd
    
    with open(path_json_path, "r", encoding="utf-8") as f:
        path_data = json.load(f)

    data_paths = path_data["data_paths"]["PATHS"]

    ## DATAs ====> list
    ## element in DATAs = (time, title)
    ## each element each run
    DATAs = get_datas(data_paths, prt_info=setting_data["all_prt_info"]["get_datas"])
    
    
    ###############################################################
    # get all channel info

    all_channels = background_data["channel"]["all_channels"]
    all_channels_names = all_channels["NAMES"]
    all_channels_num = len(all_channels_names)

    background_data["channel"]["all_channels"]["NUM"] = all_channels_num

    ###############################################################
    # get target marker
    all_event_marker_dict = background_data['event_markers']

    choose_marker_names = setting_data["choose_markers"]["NAMES"]
    choose_marker_num = len(choose_marker_names)

    setting_data["choose_markers"]["NUM"] = choose_marker_num


    ## marker num, ex 6, 7, 8, 9
    choose_markers = []

    choose_marker_dict = {}

    for k in choose_marker_names:

        val = all_event_marker_dict[k]

        choose_marker_dict[k] = val
        choose_markers.append(val)
    
    ################################################################
    # get event
    ## each element each run

    DATAs_event = process_before_mne_multi(DATAs, 
                                            'Software Marker')

    events_multi = get_event_multi(DATAs_event, 
                                    choose_markers, 
                                    prt_info=prt_info)
    
    print("DATAs_event: ", type(DATAs_event), len(DATAs_event))
    print("events_multi: ", type(events_multi), len(events_multi))


    ################################################################
    # get all channel data, turn into np


    ## DATAs_np ====> list
    ## element in DATAs_np = (channel, time)
    ## each element each run

    DATAs_np = process_before_mne_multi(DATAs, 
                                        all_channels_names,
                                        prt_info=setting_data["all_prt_info"]["process_outlier"])
    
    
    ################################################################
    # process_outlier

    ## DATAs_np ====> list
    ## element in DATAs_np = (channel, time)
    ## each element each run

    if setting_data["process_outlier"]["use"]:
        DATAs_np = pcso.process_outlier_multi(DATAs_np, 
                                        all_channels_names, 
                                        detect_method=setting_data["process_outlier"]["detect_method"], 
                                        replace_strategy=setting_data["process_outlier"]["replace_strategy"],
                                        prt_info=setting_data["all_prt_info"]["process_outlier"])


    ## 這樣應該算是勉強過濾掉眼動跟肌電了吧
    ## 有幾個run的離群值出現通道怪怪的，要檢查


    ##################################################################
    # normalization, standardization

    ## DATAs_np ====> list
    ## element in DATAs_np = (channel, time)
    ## each element each run

    if setting_data["data_normalization_standardization"]["use"]:
        for i in range(len(DATAs_np)):
            dns.norm_std_func(DATAs_np[i], method=setting_data["data_normalization_standardization"]["method"])
        
    ##################################################################
    # get_mne_data
    ## each element each run

    raw_mne_data_multi, get_mne_data_multi_dict = get_mne_data_multi(DATAs_np, 
                                                                    sfreq, 
                                                                    all_channels_names, 
                                                                    all_channels_num, 
                                                                    prt_info=setting_data["all_prt_info"]["get_mne_data_multi"])
                            

    ###################################################################
    # preprocessing
    ## each element each run

    pro_raw_mne_data_multi = raw_mne_data_multi
    
    pro_raw_mne_data_multi = pps.preprocessing_multi(raw_mne_data_multi, 
                                                    find_bad_channel_true=setting_data["preprocessing_trues"]["find_bad_channel_true"], 
                                                    filter_true=setting_data["preprocessing_trues"]["filter_true"], 
                                                    reference_true=setting_data["preprocessing_trues"]["reference_true"], 
                                                    ICA_true=setting_data["preprocessing_trues"]["ICA_true"])

    print("pro_raw_mne_data_multi: ", type(pro_raw_mne_data_multi), len(pro_raw_mne_data_multi))
    
    #####################################################################
    # cut trails times

    ## tmin, start from fixation
    ## tmax, end at MI finish

    '''
    # cue 座標下，要改event idx
    tmin = -2.0                 
    tmax =  4.25                
    base_time = (-1.0, 0.0)
    '''

    '''
    #  MI 座標下
    tmin = -3.0
    tmax =  3.25
    base_time = [-2.0, -1.0]
    '''

    tmin = setting_data["get_trail_data"]["tmin"]
    tmax = setting_data["get_trail_data"]["tmax"]
    base_time = setting_data["get_trail_data"]["base_time"]


    ## trails_multi ====> list
    trails_multi = get_trail_data_multi(pro_raw_mne_data_multi, 
                                        events_multi, 
                                        choose_markers,
                                        tmin, 
                                        tmax,
                                        base_time,
                                        prt_info=True
                                        )
    
    print("trails_multi: ", type(trails_multi), len(trails_multi))

    """
    for trails in trails_multi:

        trails_data = trails.get_data()

        trial_max = trails_data.max(axis=(1,2))  # 每個 trial 的最大 EEG 值
        trial_min = trails_data.min(axis=(1,2))  # 每個 trial 的最小 EEG 值

        for i, (mx, mn) in enumerate(zip(trial_max, trial_min)):
            print(f"Trial {i}: min={mn:.1f} μV, max={mx:.1f} μV")
    # """

    ############################################################################
    # combine all cvs data into one 

    trails_multi_data = []
    trails_multi_event = []

    ## trails_multi ====> list
    for trails in trails_multi:
        trails_multi_data.append(trails.get_data())
        trails_multi_event.append(trails.events[:,-1])


    ## trails_combine_data = (trail, channel, time)
    ## trails_combine_event = (trail, )

    trails_combine_data = np.concatenate(trails_multi_data, axis=0)
    trails_combine_event = np.concatenate(trails_multi_event, axis=0)

    print(f'trails_combine_data shape: {trails_combine_data.shape}')
    print(f'trails_combine_event shape: {trails_combine_event.shape}')


    ####################################################################################
    # trails_combine_event marker 調整

    """
    6->0, 7->1, 8->2, 9->3
    不連續也會變連續 if only 6->0, 8->1
    
    if 用 store 就是 follow fit 時設定
    """

    setting_data["choose_markers"]["marker_ori"] = np.unique(trails_combine_event).tolist()

    if setting_data["mode_control"]["current_using"] == "training":
        le = LabelEncoder()
        trails_combine_event = le.fit_transform(trails_combine_event)
        pickle.dump(le, open(setting_data["mode_control"]["label_encoder_path"], "wb"))

        setting_data["choose_markers"]["marker_transform"] = np.unique(trails_combine_event).tolist()
        


    elif setting_data["mode_control"]["current_using"] in ["testing", "calibration"]:
        
        le = pickle.load(open(setting_data["mode_control"]["label_encoder_path"], "rb"))
        trails_combine_event = le.transform(trails_combine_event)
    
    
    
    print("trails_combine_event unique:", np.unique(trails_combine_event))


    


    #############################################################################
    # choose channel 

    choose_channels = setting_data["choose_channels"]
    choose_channels_names = choose_channels["NAMES"]
    choose_channels_num = len(choose_channels_names)

    setting_data["choose_channels"]["NUM"] = choose_channels_num


    all_channels_names_lower_all = [ch.lower() for ch in all_channels_names]
    choose_channels_names_lower_all = [ch.lower() for ch in choose_channels_names]


    target_channel_idxs = [all_channels_names_lower_all.index(ch) for ch in choose_channels_names_lower_all]


    ## trails_target_data = (trail, choose_channel, time)
    trails_target_data = trails_combine_data[:, target_channel_idxs, :]

    #############################################################################
    # channel argument

    ## ex. trails_combine_C3 = (trail, 1, time)
    ## seen as new channel
    trails_combine_C3 = np.expand_dims(trails_combine_data[:,15,:], axis=1)
    trails_combine_Cz = np.expand_dims(trails_combine_data[:,16,:], axis=1)
    trails_combine_C4 = np.expand_dims(trails_combine_data[:,17,:], axis=1)

    addition_C3_C4 = trails_combine_C3 + trails_combine_C4
    diff_C3_C4 = trails_combine_C3 - trails_combine_C4
    diff_C4_C3 = trails_combine_C4 - trails_combine_C3

    ## trails_target_data_argument = (trail, choose_channel + aug , time)
    trails_target_data_argument = np.concatenate([trails_target_data,
                                                            addition_C3_C4,
                                                            diff_C3_C4,
                                                            diff_C4_C3,
                                                            trails_combine_Cz*2], axis=1)

    '''
    # 沒設定 multi S kernel 可能沒差
    trails_target_data_argument = np.concatenate([np.expand_dims(trails_combine_data[:,11,:], axis=1),
                                        np.expand_dims(trails_combine_data[:,16,:], axis=1),
                                        diff_C3_C4,
                                        np.expand_dims(trails_combine_data[:,15,:], axis=1),
                                        trails_combine_Cz*1.5,
                                        np.expand_dims(trails_combine_data[:,17,:], axis=1),
                                        diff_C4_C3,
                                        np.expand_dims(trails_combine_data[:,16,:], axis=1),
                                        np.expand_dims(trails_combine_data[:,21,:], axis=1),
                                        addition_C3_C4
                                        ], axis=1) 
    '''
    ###############################################################################
    # prepare DL data

    trails_target_data_in_DL = trails_target_data.copy()

    ## combine data argument
    if setting_data["channel_aug"]["DL"]:
        trails_target_data_in_DL = trails_target_data_argument


    # X 時間 -3 ~ 3.25 取 0 ~ 3.25
    trails_target_data_in_DL = trails_target_data_in_DL[:, :, int(sfreq * 3): int(sfreq * 6)]  # (trial, channel, time)

    """
    ## window slice --> 應該要在 train valid set 分離之後，避免同 trail 相似性問題

    # 單位 秒
    window_time = 1
    step_time = 0.1
    trails_target_data_in_DL_ws, trails_combine_event_ws = ws.window_slice(trails_target_data_in_DL,
                                                                                trails_combine_event,
                                                                                sfreq=sfreq,
                                                                                window_time=window_time, 
                                                                                step_time=step_time
                                                                                )
    """

    ###############################################################################
    # record all acc
    all_accuracy_dict = {}

    ###############################################################################
    # mode control

    current_using_mode = setting_data["mode_control"]["current_using"]
    
    ###############################################################################
    # training part (ML)
    '''
    for clfr in setting_data["ML_classifier_method"]["current_using"]:
        if current_using_mode == "training":

            ###############################################################################
            # classification_ML
            """
            trails_target_data = (trail, choose_channel, time)
            trails_combine_event = (trail, 3)
            classifier ====> "SVM" or "LDA"

            n_splits 是 K fold 的 K
            if_normailze, true 會做
            """
            ## LDA, SVM data argument 好像會有問題
            """
            LDA
            n_splits=5, # LDA 調到6會overfit

            # LDA 不能normalize，效果不好
            """
            """
            SVM
            通常 kernel linear 最好
            """

            trails_target_data_in_ML = trails_target_data

            for clfr in setting_data["traditional_ML_classifier_method"]["current_using"]:
                # all_accuracy_dict[clfr] 
                setting_data["all_accuracy"][clfr] = cf.classification_ML(trails_target_data_in_ML, 
                                                                        trails_combine_event,
                                                                        sfreq=sfreq,
                                                                        classifier=clfr,
                                                                        arg_dict=setting_data["traditional_ML_classifier_method"][clfr],
                                                                    )
    '''

    ###############################################################################
    # training part (DL)

    for clfr in setting_data["DL_classifier_method"]["current_using"]:

        ###############################################################################################
        # 參考 classification_func.py ---> classification_DL
        if current_using_mode == "training": 
            
            setting_data["DL_classifier_method"][clfr]["mode"] = "training"
            setting_data["DL_classifier_method"][clfr]["n_classes"] = choose_marker_num

            all_accuracy_dict["train_result"] = {}
            all_accuracy_dict["train_result"][clfr] = cf.classification_DL(trails_target_data_in_DL, 
                                                                                trails_combine_event,
                                                                                sfreq=sfreq,
                                                                                classifier=clfr,
                                                                                arg_dict=setting_data["DL_classifier_method"][clfr]
                                                                                )

        ###############################################################################################
        # 參考 classification_func.py ---> test_DL
        elif current_using_mode == "testing":

            setting_data["DL_classifier_method"][clfr]["mode"] = "testing"

            all_accuracy_dict["test_result"] = {}

            all_accuracy_dict["test_result"][clfr] = cf.test_DL(trails_target_data_in_DL_ws, 
                                                                    trails_combine_event_ws,
                                                                    classifier=clfr,
                                                                    arg_dict=setting_data["DL_classifier_method"][clfr]
                                                                    )


        #################################################################################
        # 參考 classification_func.py ---> classification_DL

        elif current_using_mode == "calibration":
            
            setting_data["DL_classifier_method"][clfr]["mode"] = "calibration" 
            setting_data["DL_classifier_method"][clfr]["n_classes"] = choose_marker_num

            all_accuracy_dict["calib_result"] = {}
            all_accuracy_dict["calib_result"][clfr] = classification_DL(trails_target_data_in_calib, 
                                                                    trails_combine_event,
                                                                    classifier=clfr,
                                                                    arg_dict=setting_data["DL_classifier_method"][clfr]
                                                                    )

        else:
            raise ValueError(f"Unknown normalization/standardization method: {method}")
    ######################################################################################
    # acc record
    setting_data["all_accuracy"] = all_accuracy_dict
    

    if setting_data["data_info"]["IF_RECORD_RESULT"]:
        with open(setting_data["data_info"]["RESULT_JSON_PATH"], "w", encoding="utf-8") as f:
            json.dump(setting_data, f, ensure_ascii=False, indent=4)

        print(f"{setting_data["data_info"]["RESULT_JSON_PATH"]} 已完成")


if __name__ == "__main__":
    main()