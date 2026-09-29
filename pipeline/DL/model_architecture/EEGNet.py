import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, confusion_matrix
import copy
import pickle
from tqdm import tqdm


"""
# 並定會做 全channel S_kernel
## 由於padding的問題，暫且先只選偶數，想試3
## overfit有可能是資料太少的問題


# 第一層：Temporal Convolution（時間濾波器）
## 不同 kernel size 相當於不同頻帶濾波器（如 α, β, γ 頻段）。
'''
| 參數                               | 意義         | 說明                                                                                      |
| -------------------------------- | ----------     | --------------------------------------------------------------------------------------- |
| `in_channels = 1`                | 輸入通道數      | EEG 資料 shape 是 `(batch, 1, n_channels, n_times)`，1 是指「每個 trial 是單一輸入影像」，即一張 2D EEG 時空圖。 |
| `out_channels = F1`              | 輸出濾波器數量   | 表示要學出 F1 個不同的時間濾波器。常設 4 或 8。                                                            |
| `kernel_size = (1, kernLength)`  | 卷積核大小       | `(1, kernLength)` 表示只沿 **時間軸 (n_times)** 卷積，不在 **空間軸 (n_channels)** 上動作。                |
| `padding = (0, kernLength // 2)` | padding 補邊     | 保持時間維度不變（"same" padding）。                                                               |
| `bias=False`                     | 不加偏差項        | 因為後面 BatchNorm 會處理偏移。                                                                   |
'''

# 第二層：Depthwise Spatial Convolution（空間濾波器）
## 模仿傳統 EEG 分析裡的「通道加權」、「空間濾波器」(例如 CSP)
'''
Conv2d 這裡的 kernel size 是 (n_channels, 1)
→ 代表「一次卷積涵蓋所有通道」，但只在時間維度上取 1（不跨時間）。

groups=F1 → 表示這是一個 Depthwise Convolution：
每一組（對應每個 temporal filter）各自學習自己的空間濾波器。

F1 是前一層 temporal filter 的個數。
所以這裡總共有 F1 組濾波器，各自針對整個通道空間卷積。

D 是 depth multiplier，表示對每個 temporal feature 要學多少個空間濾波器。
因此輸出通道數是 F1 * D。

「對每個時間特徵圖（頻帶），學 D 個空間權重組合，用來捕捉不同腦區 pattern。」
'''

"""
class EEGNet(nn.Module):
    def __init__(self, 
                    n_channels=0, 
                    n_samples=0, 
                    n_classes=0, 
                    F1=4, 
                    D=2, 
                    F2=8, 
                    T_kernSizes=[64, 128, 256], 
                    S_kernSizes=[], 
                    dropout=0.55
                    ):

        super(EEGNet, self).__init__()
        ################################################################################
        # 1. Temporal Conv
        self.T_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, F1, (1, k), padding=(0, k//2), bias=False),

                # ************************** 改用 GN 試試看
                # nn.GroupNorm(num_groups=1, num_channels=F1),
                nn.BatchNorm2d(F1),

                nn.ELU()
            )
            for k in T_kernSizes
        ])
        ## 將多kernel concat在channel維度上，因此總輸出通道數 = F1 * len(kernSizes)
        F1 = F1 * len(T_kernSizes)

        ################################################################################
        # 2. Spatial Conv

        # S multi kernel
        S_kernSizes = [n_channels, n_channels//2 +1, 3]  # 固定用這三種
        self.S_kernSizes = S_kernSizes
        print(f"S_kernSizes: {S_kernSizes}")

        
        self.S_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F1, F1 * D, (ks, 1), groups=F1, padding=(ks//2, 0), bias=False),

                # ************************** 改用 GN 試試看
                # nn.GroupNorm(num_groups=1, num_channels=F1 * D),
                nn.BatchNorm2d(F1 * D),


                nn.ELU()
            )
            for ks in S_kernSizes  # e.g. [n_channels, n_channels//2, 3]
        ])
        # concat 後的總通道數
        # F1D_total = F1 * D * len(S_kernSizes)
        F1 = F1 * len(S_kernSizes)
        '''

        # 喔目前只用全channel S conv，沒用 multi kernel阿難怪
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        '''


        # 3. pooling + dropout
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        
        # 4. Separable Conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),

            # ************************** 改用 GN 試試看
            # nn.GroupNorm(num_groups=1, num_channels=F2),
            nn.BatchNorm2d(F2),


            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # flatten feature_dim
        self.feature_dim = self._get_feature_dim(n_channels, n_samples)

        # classifier
        self.classifier = nn.Linear(self.feature_dim, n_classes)



    def _get_feature_dim(self, n_channels, n_samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            
            # 1. T conv
            x = torch.cat([b(x) for b in self.T_branches], dim=1)

            # 2. S conv
            '''
            x = self.depthwise(x)
            x = self.bn2(x)
            '''
            x = torch.cat([b(x) for b in self.S_branches], dim=1)
            


            # 3. pool + dropout
            x = self.pool1(x)
            x = self.dropout1(x)

            # 4. Separable
            x = self.separable(x)
            x = x.flatten(1)
        
        return x.shape[1]

    def forward(self, x):
        # x: (batch, 1, n_channels, n_times)

        # 1. T conv
        x = torch.cat([b(x) for b in self.T_branches], dim=1)

        # 2. S conv
        """
        x = self.depthwise(x)
        x = self.bn2(x)
        """
        x = torch.cat([b(x) for b in self.S_branches], dim=1)  
        x = F.elu(x)


        # 3. pool + dropout
        x = self.pool1(x)
        x = self.dropout1(x)

        # 4. Separable
        x = self.separable(x)

        x = x.flatten(1)
        return self.classifier(x)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
##########################################################################################

# functions

##########################################################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



def augment_eeg(batch, noise_std=0.01, channel_dropout=0.1):
    """
    EEG 資料增強
    batch: (batch, 1, n_channels, n_times)
    noise_std: 高斯噪音標準差
    channel_dropout: 隨機丟棄 channel 的比例
    """
    batch_aug = batch.clone()
    
    # 1. 高斯噪音
    noise = torch.randn_like(batch_aug) * noise_std
    batch_aug += noise
    
    '''
    choose channel 偏少，我認為不適用，但沒試過就是了
    # 2. channel dropout
    if channel_dropout > 0:
        n_channels = batch_aug.shape[2]
        for i in range(batch_aug.shape[0]):  # 遍歷 batch
            mask = (torch.rand(n_channels) > channel_dropout).float()
            mask = mask.view(1, n_channels, 1)
            batch_aug[i, 0] *= mask  # 乘上 mask
    '''
    return batch_aug


def evaluate(model,
                X_test,
                y_test,
                arg_dict
                ):

    ##########################################################################################################
    # Evaluation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []


    n_trials, n_channels, n_times = X_test.shape

    n_classes = arg_dict["n_classes"]
    batch_size = arg_dict["using"]["batch_size"]
    
    
    with torch.no_grad():

        if arg_dict["if_normailze"]:
        # if False:
            print("if_normailze")

            for idx in range(0, n_trials, batch_size):

                batch_slice = slice(idx, idx + batch_size)
                X_test_batch = X_test[batch_slice]   # (B, C, T)
               
                logits_list = []

                ALL_MARKER_MEAN = np.array(arg_dict["ALL_MARKER_MEAN"])
                ALL_MARKER_STD = np.array(arg_dict["ALL_MARKER_STD"])
                

                for c in range(n_classes):

                    MARKER_MEAN = ALL_MARKER_MEAN[c]
                    MARKER_STD = ALL_MARKER_STD[c]
                    
                    Xv = (X_test_batch[:, :, :] - MARKER_MEAN.reshape(1, n_channels, 1)) / (MARKER_STD.reshape(1, n_channels, 1) + 1e-10)
                    Xv = torch.tensor(Xv, dtype=torch.float32).unsqueeze(1).to(device)

                
                    logits = model(Xv)   # shape (B, n_classes)
                    logits = torch.softmax(logits, dim=1)

                    logits_list.append(logits)
                    
                    #print(f"v: {v}, logits: {logits}")

                # stack + mean → ensemble
                logits_mean = torch.stack(logits_list, dim=0).mean(dim=0)  # (B, n_classes)
                pred_labels = logits_mean.argmax(dim=1).cpu().numpy()

                # ground truth
                yb = y_test[batch_slice]

                #print(f"logits_mean: {logits_mean}")
                #print(f"yb: {yb}")

                correct += (pred_labels == yb).sum()
                total += len(yb)

                ## numpy only in cpu
                all_preds.extend(pred_labels)
                all_labels.extend(yb)
        
        else:
            print("no norm")
            test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
            test_loader  = DataLoader(test_ds, batch_size=batch_size)
            
            # pbar = tqdm(test_loader, desc="evaluating")

            # for Xb, yb in pbar:
            for Xb, yb in test_loader:

                Xb, yb = Xb.to(device), yb.to(device)

                ##################################################
                # test no augment_eeg

                preds = model(Xb)
                pred_labels = preds.argmax(1)

                #print(f"preds: {preds}")
                #print(f"yb: {yb}")
                

                correct += (pred_labels == yb).sum().item()
                total += len(yb)

                ## numpy only in cpu
                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        

    #############################################
    # result

    acc = correct / total
    print(f"Test Accuracy: {acc:.2%} ")
    return acc, all_preds, all_labels

# no use 
def calibration(X_calib_ori, 
                    y_calib_ori,
                    X_valid_ori,
                    y_valid_ori,
                    arg_dict,
                    best_acc_all
                    ):
    
    ################################################################
    # 創建 generator
    seed = arg_dict["using"]["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    ##################################################
    # prevent cover data

    X_calib = X_calib_ori.copy()
    y_calib = y_calib_ori.copy()

    X_valid = X_valid_ori.copy()
    y_valid = y_valid_ori.copy()

    ########################################################
    # using parameter
    (n_trials, n_channels, n_times) = X_calib.shape


    n_classes = arg_dict["n_classes"]

    n_epochs = arg_dict["using"]["n_epochs"]
    batch_size = arg_dict["using"]["batch_size"]
    lr = arg_dict["using"]["lr"]
    weight_decay = arg_dict["using"]["weight_decay"]

    # gradient clipping 上限
    # max_grad_norm = arg_dict["using"]["max_grad_norm"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##################################################################################
    # Tensor 化
    calib_ds = TensorDataset(torch.tensor(X_calib, dtype=torch.float32).unsqueeze(1), torch.tensor(y_calib, dtype=torch.long))
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=True, generator=g)

    model = EEGNet(n_channels, n_times, n_classes, 
                    F1=arg_dict["init"]["F1"],
                    D=arg_dict["init"]["D"], 
                    F2=arg_dict["init"]["F2"],
                    T_kernSizes=arg_dict["init"]["T_kernSizes"],
                    S_kernSizes=arg_dict["init"]["S_kernSizes"],
                    dropout=arg_dict["init"]["dropout"]
                    ).to(device)

    model.load_state_dict(torch.load(arg_dict["model_load_path"]))

    best_state_dict = copy.deepcopy(model.state_dict())

    ################################################################
    # 凍結， calib 設定

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    # AdaBN: 只 forward calibration data, 不用 backward

    model.train()   # 讓 BN 更新 running mean/var

    with torch.no_grad():
        for X, _ in calib_loader:
            _ = model(X.to(device))

    # 初始化最後一層 classifier
    old_in_dim = model.classifier[-1].in_features   # 抓原本 linear 輸入維度
    model.classifier = nn.Linear(old_in_dim, n_classes).to(device)


    ###############################################################################
    # optimizer and loss function

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # lr 調整

    # scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # 每20個epoch減半

    ###############################################################################
    # Calibration loop

    best_acc_fold = 0.0

    for epoch in range(1, n_epochs+1):

        model.train()
        total_loss = 0

        for Xb, yb in calib_loader:

            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()

            ############################################################################
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            ############################################################################

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f}, ", end='')
        val_acc_epoch, all_preds, all_labels = evaluate(model,
                                                            X_valid,
                                                            y_valid,
                                                            arg_dict
                                                            )#'''

        ###################################################################################
        # 存 val acc highest model weight

        if val_acc_epoch > best_acc_fold:
            best_acc_fold = val_acc_epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"New best fold model at epoch {epoch} with acc {val_acc_epoch:.4f}")

        ###################################################################################
        # 儲存模型權重
        # 存當前全部最高 acc 的 model

        if val_acc_epoch > best_acc_all:
            best_acc_all = val_acc_epoch
            if arg_dict["model_save_path"]:
                torch.save(model.state_dict(), arg_dict["model_save_path"])
            print(f"New best calib model in all saved at epoch {epoch} with acc {val_acc_epoch:.4f}")


        ##########################################################################################
        # 更新學習率

        # scheduler.step()  
        # print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")



def testing(X_test, 
                y_test,
                arg_dict,
                ):

    (n_trials, n_channels, n_times) = X_test.shape

    model_path = arg_dict["model_load_path"]

    n_classes = arg_dict["n_classes"]
    batch_size = arg_dict["using"]["batch_size"]

    model = EEGNet(n_channels, n_times, n_classes, 
                   F1=arg_dict["init"]["F1"],
                    D=arg_dict["init"]["D"], 
                    F2=arg_dict["init"]["F2"],
                    T_kernSizes=arg_dict["init"]["T_kernSizes"],
                    S_kernSizes=arg_dict["init"]["S_kernSizes"],
                    dropout=arg_dict["init"]["dropout"]
                    )

    # Handle both checkpoint format and direct state_dict
    checkpoint = torch.load(model_path, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_config = checkpoint.get("model_config", {})
        
        # Rebuild model with saved config if available
        if model_config:
            init_saved = model_config.get("init", arg_dict["init"])
            n_classes_saved = model_config.get("n_classes", n_classes)
            n_channels_saved = model_config.get("n_channels", n_channels)
            n_times_saved = model_config.get("n_times", n_times)
            
            model = EEGNet(n_channels_saved, n_times_saved, n_classes_saved, 
                           F1=init_saved["F1"],
                            D=init_saved["D"], 
                            F2=init_saved["F2"],
                            T_kernSizes=init_saved["T_kernSizes"],
                            S_kernSizes=init_saved["S_kernSizes"],
                            dropout=init_saved["dropout"]
                            )
        
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)

    acc, all_preds, all_labels = evaluate(model,
                                            X_test,
                                            y_test,
                                            arg_dict
                                            )

    return acc, all_preds, all_labels
    


def EEGNet_setting_using(X_train_ori, 
                            y_train_ori,
                            X_valid_ori, 
                            y_valid_ori,
                            arg_dict,
                            best_acc_all,
                            best_LOSS_all
                            ):
    """
    X_train_ori = (trail, choose_channel, time) or (trail*nv, choose_channel, time)
    y_train_ori = (trail, ) or (trail*nv, )
    X_valid_ori = (trail, choose_channel, time)
    y_valid_ori = (trail, )

    """
    ################################################################
    # 創建 generator
    seed = arg_dict["using"]["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    ##################################################
    # prevent cover data

    X_train = X_train_ori.copy()
    y_train = y_train_ori.copy()

    X_valid = X_valid_ori.copy()
    y_valid = y_valid_ori.copy()

    ########################################################
    # using parameter

    (n_trials, n_channels, n_times) = X_train.shape
    

    mode = arg_dict["mode"]
    n_classes = arg_dict["n_classes"]

    n_epochs = arg_dict["using"]["n_epochs"]
    batch_size = arg_dict["using"]["batch_size"]
    lr = arg_dict["using"]["lr"]
    weight_decay = arg_dict["using"]["weight_decay"]

    # gradient clipping 上限
    # max_grad_norm = arg_dict["using"]["max_grad_norm"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##############################################################
    # Tensor 化
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    model = EEGNet(n_channels, n_times, n_classes, 
                    F1=arg_dict["init"]["F1"],
                    D=arg_dict["init"]["D"], 
                    F2=arg_dict["init"]["F2"],
                    T_kernSizes=arg_dict["init"]["T_kernSizes"],
                    S_kernSizes=arg_dict["init"]["S_kernSizes"],
                    dropout=arg_dict["init"]["dropout"]
                    ).to(device)

    
    ###############################################################################
    # training mode
    if mode == "training":

        ###############################################################################
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)




    ###############################################################################
    # calibration mode

    elif mode == "calibration":
        model.load_state_dict(torch.load(arg_dict["model_load_path"]))

        for name, param in model.named_parameters():
            if "classifier" not in name:   # 只讓最後一層可更新
                param.requires_grad = False

        # AdaBN: 只 forward calibration data, 不用 backward
        model.train()   # 讓 BN 更新 running mean/var

        with torch.no_grad():
            for X, _ in train_loader:
                _ = model(X.to(device))
        
        # 初始化最後一層 classifier
        old_in_dim = model.classifier.in_features   # 抓原本 linear 輸入維度
        model.classifier = nn.Linear(old_in_dim, n_classes).to(device)


        ###############################################################################
        # optimizer
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    
    ###############################################################################
    # loss function

    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # lr 調整

    # scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # 每20個epoch減半

    ###############################################################################
    # Training or Calibration loop
    
    best_acc_fold = 0.0
    best_LOSS_fold = 100.0

    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, n_epochs+1):

        model.train()
        total_loss = 0

        for Xb, yb in train_loader:

            Xb, yb = Xb.to(device), yb.to(device)

            """
            if mode == "training":
                Xb = augment_eeg(Xb, 
                                noise_std=arg_dict["augment"]["noise_std"], 
                                channel_dropout=arg_dict["augment"]["channel_dropout"])
            """
                
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()

            ############################################################################
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            ############################################################################

            optimizer.step()
            total_loss += loss.item()

        LOSS = total_loss/len(train_loader)
        print(f"Epoch {epoch:02d} | Loss: {LOSS:.4f}, ", end='')
        val_acc_epoch, all_preds, all_labels = evaluate(model,
                                                            X_valid,
                                                            y_valid,
                                                            arg_dict
                                                            )#'''

        ###################################################################################
        # 存 val acc highest model weight
        """
        if val_acc_epoch > best_acc_fold:
            best_acc_fold = val_acc_epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"New best fold model at epoch {epoch} with acc {val_acc_epoch:.4f}")
        """

        if LOSS < best_LOSS_fold:
            best_LOSS_fold = LOSS
            best_acc_fold = val_acc_epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            # New best fold model
            print(f"NBFM at epoch {epoch} with LOSS {LOSS:.4f}, acc {val_acc_epoch:.4f}")


        ###################################################################################
        # 儲存模型權重
        """
        # 存當前全部最高 acc 的 model

        if val_acc_epoch > best_acc_all:
            best_acc_all = val_acc_epoch

            save_path = arg_dict["model_save_path"]
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
            print(f"New best model in all saved at epoch {epoch} with acc {val_acc_epoch:.4f}")
        """

        if LOSS < best_LOSS_all:

            best_LOSS_all = LOSS
            best_acc_all = val_acc_epoch

            save_path = arg_dict["model_save_path"]
            """
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                """

            if save_path:
                checkpoint = {
                                'model_state_dict': model.state_dict(),
                                'model_config': {
                                    'n_classes': n_classes,
                                    'n_channels': n_channels,
                                    'n_times': n_times,
                                    'init': arg_dict["init"]
                                }
                            }
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkpoint, save_path)

            # New best all model
            print(f"NBAM at epoch {epoch} with LOSS {LOSS:.4f}, acc {val_acc_epoch:.4f}")



        ##########################################################################################
        # 更新學習率

        # scheduler.step()  
        # print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    
    ##########################################################################################################
    # Evaluation by fold best model

    model.load_state_dict(best_state_dict)

    val_highest_acc, all_preds, all_labels = evaluate(model,
                                                        X_valid,
                                                        y_valid,
                                                        arg_dict
                                                        )

    # result

    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, digits=4))

    """
    val_highest_acc =====> float
    all_preds ====> list
    all_labels ====> list
    best_acc_all =====> float
    """

    return val_highest_acc, all_preds, all_labels, best_acc_all, best_LOSS_all
    

