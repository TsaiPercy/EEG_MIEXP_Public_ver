import numpy as np
import matplotlib.pyplot as plt
import math
import json
from sklearn.metrics import ConfusionMatrixDisplay


# my
import plot_func as plf


def get_file_path_func(P, SP, FN):

    
    if FN == "all":
        nP = not P
        return f"./result_record/ablation/{SP}/{FN}_{nP}.json"

    else:
        return f"./result_record/ablation/{SP}/{FN}_{P}.json"



def main():

    ################################################################################################
    # AVE ACC part
    plot_path_list = []
    

    member_name_list = ["S01", "S02", "S06", "S04", "S05", "S07", "S03"]
    member_num_list = list(range(len(member_name_list)))

    for i in range(len(member_num_list)):
        member_num_list[i]= "Subject_" + str(member_num_list[i])

    

    TF_list = [True, 
                False
                ]
    RA_list = ["Append", "Remove"]


    note_list = ["process_outlier", 
                    "data_normalization_standardization", 
                    "filter_true", 
                    "reference_true", 
                    "if_normailze",
                    # "embedding_method",
                    "channel_aug",
                    "all"
                    ]
    
    LDA_open = False
    SVM_open = False
    EEGNet_open = True


    if LDA_open:

        plot_path_list = []

        data_dict_keys = ["all_accuracy", "LDA", "AVE_ACC"]
        
        for TF in TF_list:
        
            if TF:
                RA = "Append"
            else:
                RA = "Remove"

            plot_path_list.append(f"./plot_use/plot_img/abl_LDA_{RA}.png")

            
        plf.plot_whole_process(
                            data_dict_keys=data_dict_keys,
                            plot_list=TF_list,
                            subplot_list=member_name_list,
                            file_name_list=note_list,
                            get_file_path_func=get_file_path_func,
                            plot_path_list=plot_path_list,
                            subplot_titles=member_num_list
                            )

        print("LDA finish")

    if EEGNet_open:
        class_name = "EEGNet"

        plot_path_list = []

        data_dict_keys = ["all_accuracy", class_name, "AVE_ACC"]
        
    
        for TF in TF_list:
        
            if TF:
                RA = "Append"
            else:
                RA = "Remove"

            plot_path_list.append(f"./plot_use/plot_img/abl_{class_name}_{RA}.png")

            
        plf.plot_whole_process(
                            data_dict_keys=data_dict_keys,
                            plot_list=TF_list,
                            subplot_list=member_name_list,
                            file_name_list=note_list,
                            get_file_path_func=get_file_path_func,
                            plot_path_list=plot_path_list,
                            subplot_titles=member_num_list
                            )

        print(f"{class_name} finish")

    ################################################################################################
    # CM part

    note_list = ["all_True", "all_False"]
    
    file_path_list = []
    plot_path_list = []

    data_dict_keys = ["all_accuracy", "EEGNet", "confusion_matrix"]


    for member_name in member_name_list:
        for note in note_list:
            file_path_list.append(f"./result_record/ablation/{member_name}/{note}.json")
            plot_path_list.append(f"./plot_use/plot_img/CM_{member_name}_{note}.png")

    
    plot_data_list = plf.get_CM(file_path_list, data_dict_keys)
    
    print(f"plot_path_list len: {len(plot_path_list)}")

    plot_data_np = np.array(plot_data_list)
    print(f"plot_data_np shape: {plot_data_np.shape}")

    

    for cm, plot_path in zip(plot_data_list, plot_path_list):

        cm=np.array(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # 可改成 "viridis", "Reds" 等色盤
        disp.plot(cmap='Blues')

        plt.title("Confusion Matrix")
        plt.savefig(plot_path, dpi=150)

        print(f"plot CM finish: {plot_path}")

    return

if __name__ == "__main__":
    main()