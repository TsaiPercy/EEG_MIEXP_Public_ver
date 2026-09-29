import glob
import json

"""
有兩種使用方式：

1. 通常是給 main_session.py 引入使用，開啟 IF_DATA_RELOAD 會自動更新 path.json

2. 先在 path.json 設定 data_find_folder_path，直接 run data_path_update.py 即可更新 path.json

"""

def find_csv_files(data_find_folder_path):

    # 搜尋所有子資料夾中的 .csv 檔案
    csv_files = glob.glob(f"{data_find_folder_path}/**/*.csv", recursive=True)

    # 排序確保順序固定
    csv_files.sort()
    return csv_files



def save_csv_paths_to_json(path_json_path):
    '''
    如果要直接執行此檔案，請事先在 path.json 設定 data_find_folder_path
    '''

    with open(path_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    data_find_folder_path = data["data_find_folder_path"]

    csv_files = find_csv_files(data_find_folder_path)


    data["data_paths"] = {
                            "NUM": len(csv_files),
                            "PATHS": csv_files
                        }
    

    # 儲存成 JSON（使用 utf-8 確保中文不亂碼）
    with open(path_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"已將 {len(csv_files)} 個檔案路徑存入 {path_json_path}")





def main():
         
    path_json_path = "./json/path.json"  

    save_csv_paths_to_json(path_json_path)

   


if __name__ == "__main__":
    main()