import json




import main_session as MSS
import data_path_update as dpu



def return_all(setting_data, background_data, TF=True):
    setting_data["process_outlier"]["use"] = TF
    setting_data["data_normalization_standardization"]["use"] = TF
    setting_data["preprocessing_trues"]["filter_true"] = TF
    setting_data["preprocessing_trues"]["reference_true"] = TF


    setting_data["traditional_ML_classifier_method"]["LDA"]["if_normailze"] = TF
    setting_data["traditional_ML_classifier_method"]["SVM"]["if_normailze"] = TF

    setting_data["DL_classifier_method"]["EEGNet"]["if_normailze"] = TF

    all_method = []

    if TF:
        all_method = [
                        "CSP",
                        "Riemannian",
                        "Hjorth",
                        "DE",
                        "PSD"
                    ]
                    

    setting_data["traditional_ML_classifier_method"]["LDA"]["embedding_method"]["current_using"] = all_method
    setting_data["traditional_ML_classifier_method"]["SVM"]["embedding_method"]["current_using"] = all_method

    setting_data["DL_classifier_method"]["EEGNet"]["embedding_method"]["current_using"] = all_method

    setting_data["channel_aug"]["ML"] = TF
    setting_data["channel_aug"]["DL"] = TF

    setting_data["choose_channels"].update(background_data["channel"][setting_data["choose_channels"]["MODE"]])


def main():

    
    name_list = ["S02", "S06", "S04", "S05", "S07", "S03"]
    # "S01",

    setting_path = "./setting_json/setting_ablation_use.json"


    background_path = "./json/background.json"
    path_path = "./json/path.json"



    for i in range(len(name_list)):

        name = name_list[i]

        ###########################################################################
        # path.json update
        
        with open(path_path, "r", encoding="utf-8") as f:
            path_data = json.load(f)

        # path_data["data_find_folder_path"] = "./data/no_use"
        path_data["data_find_folder_path"] = "./data/no_use/" + name

        with open(path_path, "w", encoding="utf-8") as f:
            json.dump(path_data, f, ensure_ascii=False, indent=4)
        
        dpu.save_csv_paths_to_json(path_path)

        ###########################################################################


        with open(setting_path, "r", encoding="utf-8") as f:
            setting_data = json.load(f)

        with open(setting_path, "r", encoding="utf-8") as f:
            setting_data = json.load(f)

        with open(background_path, "r", encoding="utf-8") as f:
            background_data = json.load(f)


        setting_data["mode_control"]["current_using"] = "training"


        setting_data["data_info"]["SUBJECT_NAME"] = name
        setting_data["data_info"]["SUBJECT_num"] = i
        setting_data["data_info"]["DESCRIPTION"] = "MI + ME all data, training mode"
        
        setting_data["data_info"]["IF_DATA_RELOAD"] = False
        setting_data["data_info"]["data_find_folder_path"] = "./data/no_use/" + name
        #setting_data["data_info"]["data_find_folder_path"] = "./data/no_use"
        setting_data["data_info"]["IF_RECORD_RESULT"] = True
        setting_data["data_info"]["note"] = "epoch 60"

        

        note_list = ["process_outlier", 
                    "data_normalization_standardization", 
                    "filter_true", 
                    "reference_true", 
                    "if_normailze",
                    #"embedding_method", #維修中
                    "channel_aug",
                    "all"
                    ]
            
        TF_list = [True, 
                    False
                    ]
        

        for note in note_list:

            for TF in TF_list:
    
                nTF = not TF

                return_all(setting_data, background_data, TF)

                setting_data["data_info"]["RESULT_JSON_PATH"] = f"./result_record/ablation/{name}/{note}_{nTF}.json"
                

                if note == "filter_true" or note == "reference_true":
                    setting_data["preprocessing_trues"][note] = nTF

                elif note == "if_normailze":
                    setting_data["traditional_ML_classifier_method"]["LDA"][note] = nTF
                    setting_data["traditional_ML_classifier_method"]["SVM"][note] = nTF

                    setting_data["DL_classifier_method"]["EEGNet"][note] = nTF

                elif note == "embedding_method":

                    all_method = []

                    if nTF:
                        all_method = [
                                        "CSP",
                                        "Riemannian",
                                        "Hjorth",
                                        "DE",
                                        "PSD"
                                    ]

                    setting_data["traditional_ML_classifier_method"]["LDA"]["embedding_method"]["current_using"] = all_method
                    setting_data["traditional_ML_classifier_method"]["SVM"]["embedding_method"]["current_using"] = all_method

                    setting_data["DL_classifier_method"]["EEGNet"]["embedding_method"]["current_using"] = all_method

                elif note == "channel_aug":
                    setting_data["channel_aug"]["ML"] = nTF
                    setting_data["channel_aug"]["DL"] = nTF
                    
                elif note == "all":
                    return_all(setting_data, background_data,nTF)
                else:
                    setting_data[note]["use"] = nTF

                with open(setting_path, "w", encoding="utf-8") as f:
                    json.dump(setting_data, f, ensure_ascii=False, indent=4)

                MSS.main(setting_path, use_input_setting_path=False)
                

                print(f"{note} {str(nTF)}已完成")

        


        print(f"{name} 已完成")



if __name__ == "__main__":
    main()

