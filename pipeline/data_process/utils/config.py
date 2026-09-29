"""
config.py

路徑使用 __file__ 相對定位，不依賴 CWD
"""
import os

# utils/config.py 所在目錄 → data_process/utils/
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
# 上一層 → data_process/
_BASE_DIR = os.path.dirname(_UTILS_DIR)

background_path = os.path.join(_BASE_DIR, "json", "background.json")
path_path = os.path.join(_BASE_DIR, "json", "path.json")