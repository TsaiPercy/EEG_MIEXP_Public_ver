import json
import os
import sys
import subprocess

def main():
    # 1. 定義要處理的受試者名字列表 (可替換為羅馬拼音或中文)
    subject_names = [
        "S01", "S02", "S03", "S04", 
        "S05", "S06", "S07"
    ]
    # 

    # 2. 設定相對路徑 (以腳本所在目錄為基準)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    setting_path = os.path.join(BASE_DIR, "json", "setting.json")
    processor_script = os.path.join(BASE_DIR, "data_processor.py")

    # 確保 setting.json 存在
    if not os.path.exists(setting_path):
        print(f"❌ 找不到設定檔: {setting_path}")
        return

    # 3. 開始批次處理迴圈
    for name in subject_names:
        print(f"\n{'='*60}")
        print(f"🚀 開始批次處理受試者: {name}")
        print(f"{'='*60}\n")

        # --- 步驟 A：讀取並修改 setting.json ---
        with open(setting_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # 修改名字與資料夾路徑 (假設您的原始資料放在 ori_csv_data/名字)
        settings["data_info"]["SUBJECT_NAME"] = name
        settings["data_info"]["data_find_folder_path"] = f"ori_csv_data/{name}"
        
        # 寫回 json 檔 (ensure_ascii=False 確保中文不會變成 Unicode 編碼)
        with open(setting_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 已更新 setting.json (SUBJECT_NAME={name})")

        # --- 步驟 B：執行 data_processor.py ---
        try:
            # sys.executable 會使用目前啟動腳本的同一個 Python 環境 (虛擬環境)
            result = subprocess.run(
                [sys.executable, processor_script],
                check=True, # 發生 Error 會觸發 CalledProcessError
                text=True
            )
            print(f"✅ 受試者 {name} 處理完成！")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 執行 data_processor.py 發生錯誤 (受試者: {name})")
            print(f"   錯誤代碼: {e.returncode}")
            # 您可以選擇在此 break 中斷迴圈，或 continue 繼續處理下一個人
            continue 

    print(f"\n🎉 所有受試者批次處理結束！")

if __name__ == "__main__":
    main()