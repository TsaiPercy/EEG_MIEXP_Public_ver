import numpy as np

import mne
from mne.preprocessing import ICA, find_bad_channels_lof



# notch 還沒做 <-- band 已經 1~40, notch 60沒意義
'''
Notch Filter（陷波濾波器） 是一種專門用來「消除特定頻率」的濾波器。
與低通、高通、帶通不同的是，它只「抑制（notch out）某個窄頻帶」，而讓其他頻率通過。

🎯 主要用途：

在 EEG、ECG、EMG 等生理訊號中，常用來去除交流電源干擾（Power Line Noise）：

國家	電力頻率	常用陷波頻率
台灣 / 日本	60 Hz	60 Hz
歐洲 / 中國	50 Hz	50 Hz

有時還會順便去掉其倍頻（如 100 Hz、120 Hz）。'''


'''
def preprocessing(raw_mne_data, 
                  find_bad_channel_true, 
                  filter_true, 
                  reference_true, 
                  ICA_true):

    raw_mne_data_copy = raw_mne_data


    if find_bad_channel_true:
        raw_mne_data_copy = preprocessing_find_bad_channel(raw_mne_data_copy)
        

    # raw_mne_data_copy.set_channel_types({'EOG':'eog'}) # 試試把 F 設為眼動參考通道?

    if filter_true:
        raw_mne_data_copy = preprocess_filter(raw_mne_data_copy, 
                                            l_freq=1., 
                                            h_freq=40., 
                                            fir_design='firwin')

    if reference_true:
        raw_mne_data_copy = raw_mne_data_copy.set_eeg_reference('average')


    if ICA_true:
        raw_mne_data_copy = preprocess_ICA(raw_mne_data_copy, 
                                        n_components=20, 
                                        method='fastica', 
                                        random_state=42, 
                                        max_iter='auto',
                                        auto_adj_true=False,
                                        artifact_std_threshold=100e-6,
                                        prt_info=False)

    print("pre")

    return raw_mne_data_copy
    '''


def preprocessing_multi(raw_mne_data_multi, 
                        find_bad_channel_true, 
                        filter_true, 
                        reference_true, 
                        ICA_true):

    ## raw_mne_data_multi is list
    ## each element each run

    print("preprocessing_multi start")
    raw_mne_data_multi_copy = raw_mne_data_multi.copy()

    raw_mne_data_multi_copy_multi = []

    for raw_mne_data_copy in raw_mne_data_multi_copy:

        if find_bad_channel_true: # 目前沒用
            raw_mne_data_copy = preprocessing_find_bad_channel(raw_mne_data_copy)
            

        # raw_mne_data_copy.set_channel_types({'EOG':'eog'}) # 試試把 F 設為眼動參考通道?
        # 忘記設完後要做什麼了

        if filter_true:

            
            print("filter start")

            """notch_freqs = [60]

            raw_mne_data_copy = raw_mne_data_copy.copy().notch_filter(freqs=notch_freqs,
                                                                        picks='eeg',       # 或 picks=pick_channels
                                                                        method='spectrum_fit', # 或 'sos' / 'iir' / 'fir'
                                                                        notch_widths=None,  # 或給定每個頻率的帶寬 (Hz) 或比例
                                                                        fir_design='firwin') # 若用 fir"""

            raw_mne_data_copy = preprocess_filter(raw_mne_data_copy, 
                                                l_freq=1., 
                                                h_freq=40., 
                                                fir_design='firwin')
            print("filter finish")


        if reference_true:
            print("reference start")
            raw_mne_data_copy = raw_mne_data_copy.set_eeg_reference('average')
            print("reference finish")


        if ICA_true:
            print("ICA start")
            raw_mne_data_copy = preprocess_ICA(raw_mne_data_copy, 
                                            n_components=20, 
                                            method='fastica', 
                                            random_state=42, 
                                            max_iter='auto',
                                            auto_adj_true=False,
                                            artifact_std_threshold=100e-6,
                                            prt_info=False)

            print("ICA finish")

        raw_mne_data_multi_copy_multi.append(raw_mne_data_copy)

    print("preprocessing_multi finsih")
   
    return raw_mne_data_multi_copy_multi


def preprocessing_find_bad_channel(raw_mne_data):

    print()
    print("####################################")
    print("find_bad_channel")
    print()


    raw_mne_data_copy = raw_mne_data

    # bad_channels: type = list
    bad_channels = find_bad_channels_lof(raw_mne_data_copy)

    raw_mne_data_copy.info['bads'] = bad_channels

    # 插值取代原值 而不是drop，True 表示標記為好，False 表示仍然為bad
    raw_mne_data_copy.interpolate_bads(reset_bads=True)



    if not len(bad_channels):
        print("channels all good")
    
    print("####################################")

    return raw_mne_data_copy  


def preprocess_filter(raw_mne_data, l_freq, h_freq, fir_design):

    '''
    'firwin'	預設方法，使用 SciPy 的 firwin 設計濾波器
    'firwin2'	更靈活，可設計非對稱通帶
    'firls'	    最小平方誤差 FIR 設計
    'kaiser'	Kaiser window 設計法，舊版 MNE 常用

    l_freq = 高通濾波（移除低頻漂移）
    h_freq = 低通濾波（去高頻雜訊）
    None 表示不做該方向濾波

    phase='minimum' → 因果濾波，只使用過去資料，
    與即時端 Streaming Filter 行為一致（離線-線上對齊）

    # ex: 低通濾波，cutoff 30 Hz
    raw.filter(l_freq=None, h_freq=30., fir_design='firwin')
    '''
    raw_mne_data_copy = raw_mne_data

    print(f"[DEBUG] preprocess_filter: l_freq={l_freq}, h_freq={h_freq}, "
          f"fir_design={fir_design}, phase='minimum' (causal)")

    raw_mne_data_copy = raw_mne_data_copy.filter(
        l_freq=l_freq, h_freq=h_freq,
        fir_design=fir_design,
        phase='minimum'  # 因果濾波，與即時端行為一致
    )
    
    return raw_mne_data_copy


def preprocess_ICA(raw_mne_data, 
                   n_components, 
                   method, 
                   random_state, 
                   max_iter,
                   auto_adj_true,
                   artifact_std_threshold,
                   prt_info):

    '''
    n_components	ICA 成分數	通常與 EEG 通道數接近 (ex. 32 channels → 20~30)
    method	ICA 方法	"fastica"（預設）、"picard"（更快）、"infomax"
    max_iter	最大迭代次數	'auto' 通常即可
    reject	設定振幅閾值自動排除爆掉的 epoch	可搭配一起使用
    auto_adj_true: true for auto std adj
    '''

    raw_mne_data_copy = raw_mne_data

    ica = ICA(n_components=n_components, 
              method=method, 
              random_state=random_state, 
              max_iter=max_iter)
    


    ica.fit(raw_mne_data_copy)


    # ica_sources shape: (n_components, n_times)
    ica_sources = ica.get_sources(raw_mne_data_copy)

    if prt_info:
        ica_sources.plot(title='ICA components')




    # type: list
    artifact_idxs = []



    # 標準差太大的是 artifact
    '''
    | 其他方法                              | 說明                                      |
    | ------------------------------- | --------------------------------------- |
    | 峰值振幅 (peak-to-peak)             | 計算 component 峰值差，過大可能是眨眼                |
    | 頻譜特徵 (power spectrum)           | 高頻成分過多 → 肌電雜訊                           |
    | 與 EOG / ECG channel correlation | 如果有 EOG/ECG channel，可自動選相關度高的 component |
    '''

    if auto_adj_true:
        print()
        stds = ica_sources.get_data().std(axis=1)
        print(stds)
        artifact_idxs = list(np.where(stds > artifact_std_threshold)[0])  # 閾值自己調整
        print(f' artifact_idxs = {artifact_idxs}')


    # 手動挑出眼動分量
    # fig = ica.plot_sources(raw_mne_data_copy, show=False, title='raw_ori')         # 看 ICA 成分波形
    # fig.savefig("img/ICA_sources.png", dpi=300)

    fig = ica.plot_components(show=False)                       # topography: red+, blue-
    fig.savefig("img/ICA_topography.png", dpi=300)

    artifact_app = [2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 18, 19]
    artifact_rm = []

    artifact_idxs = artifact_idxs + artifact_app
    
    artifact_idxs = [x for x in artifact_idxs if x not in artifact_rm]

    
    print(f' final artifact_idxs = {artifact_idxs}')

    # get exclude component
    ica.exclude = artifact_idxs


    # ICA apply
    raw_clean = ica.apply(raw_mne_data_copy)


    # 看乾淨訊號
    if prt_info:
        raw_clean.plot(title='raw_clean')                            


    return raw_clean

