"""
test.py

MIEXP 離線實驗 model 測試 Pipeline

流程: .pt → testing → results
輸入: .pt 格式同 train.py

使用方式:
    cd MI_data_process/main/DL/testing
    python test.py --pt_path ../path/to/data.pt --setting_path ./setting_test.json
"""


import os
import sys
import argparse
import json
import copy
from datetime import datetime

import numpy as np
import torch
import pickle
from sklearn.preprocessing import LabelEncoder

# ─── 基準目錄：test.py 所在的 testing/ 資料夾 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
import classification_func as cf
from utils import window_slice as ws

# ═══════════════════════════════════════════
# log setting
# ═══════════════════════════════════════════

class TeeLogger:
    """雙向輸出 Logger，同時寫入終端機與檔案"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def setup_dual_logging():
    """設定 Log 雙向輸出 (終端機與檔案)"""
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"process_log_{current_time}.txt")

    original_stdout = sys.stdout
    log_file = open(log_filename, "w", encoding="utf-8")
    sys.stdout = TeeLogger(original_stdout, log_file)

    print(f"✅ Log 將同時儲存至: {log_filename}")
    return log_file, original_stdout

# ═══════════════════════════════════════════
# main class
# ═══════════════════════════════════════════

class Tester:
    """Testing-only runner for MIEXP DL models."""

    def __init__(self, setting_path=None, seed=60):

        self.setting_path = setting_path
        self.seed = seed

        self.setting_data = self._load_json(self.setting_path)
        self._validate_setting()
        self._set_seed(self.seed)

        self.sfreq = self.setting_data["data_info"]["sfreq"]
        self.label_encoder_path = self.setting_data["mode_control"].get(
            "label_encoder_path", "./model/label_encoder.pkl"
        )
        self.classifiers = self.setting_data["DL_classifier_method"]["current_using"]
        self.setting_data["all_accuracy"] = self.setting_data.get("all_accuracy", {})

        print(f"[Tester] 初始化完成")
        print(f"  setting_path={self.setting_path}")
        print(f"  sfreq={self.sfreq}")
        print(f"  classifiers={self.classifiers}")

    # ═══════════════════════════════════════════
    # private helpers
    # ═══════════════════════════════════════════

    @staticmethod
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _set_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_setting(self):
        current_using = self.setting_data["DL_classifier_method"].get("current_using")
        if not isinstance(current_using, list) or len(current_using) == 0:
            raise ValueError("DL_classifier_method.current_using must be a non-empty list")

        # Verify model_load_path exists for each classifier
        for clfr in current_using:
            if "model_load_path" not in self.setting_data["DL_classifier_method"][clfr]:
                print(f"[WARN] model_load_path not specified for {clfr}")
            else:
                model_path = self.setting_data["DL_classifier_method"][clfr]["model_load_path"]
                if not os.path.exists(model_path):
                    raise ValueError(f"Model path not found: {model_path}")

    @staticmethod
    def load_pt(pt_path):
        data = torch.load(pt_path, weights_only=False)
        x_data = data["x_data"]
        y_data = data["y_data"]

        if isinstance(x_data, torch.Tensor):
            x_data = x_data.cpu().numpy()
        else:
            x_data = np.asarray(x_data)

        if isinstance(y_data, torch.Tensor):
            y_data = y_data.cpu().numpy()
        else:
            y_data = np.asarray(y_data)

        return x_data, y_data, data.get("metadata", {})

    @staticmethod
    def load_label_encoder(encoder_path):
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
        return le

    def _resolve_label_encoder_path(self, override_path=None):
        return override_path if override_path is not None else self.label_encoder_path

    def _resolve_result_json_path(self, override_path=None):
        if override_path is not None:
            return override_path
        if self.setting_data["data_info"].get("IF_RECORD_RESULT", False):
            return self.setting_data["data_info"].get("RESULT_JSON_PATH")
        return None

    def save_result_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.setting_data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Saved result JSON to: {path}")

    def test_from_pt(self, pt_path, label_encoder_path=None, result_json_path=None):
        print(f"[INFO] Load processed data from: {pt_path}")
        x_data, y_data, metadata = self.load_pt(pt_path)
        print(f"[INFO] x_data shape: {x_data.shape}")
        print(f"[INFO] y_data shape: {y_data.shape}")

        encoder_path = self._resolve_label_encoder_path(label_encoder_path)
        
        # Check if y_data is already 0-based
        unique_labels = np.unique(y_data)
        is_zero_based = (unique_labels >= 0).all() and (unique_labels < len(unique_labels)).all()
        
        if is_zero_based:
            print(f"[INFO] Labels already 0-based, skipping encoding")
            le = self.load_label_encoder(encoder_path)
        else:
            le = self.load_label_encoder(encoder_path)
            y_data = le.transform(y_data)
            print(f"[INFO] label encoder loaded: {encoder_path}")
        
        print(f"[INFO] mapped labels: {np.unique(y_data).tolist()}")

        # Apply window slice if configured
        window_slice_config = self.setting_data.get("window_slice", {})
        if window_slice_config.get("use", False):
            window_time = window_slice_config.get("window_time", 1.0)
            step_time = window_slice_config.get("step_time", 0.1)
            print(f"[INFO] Applying window slice: window_time={window_time}s, step_time={step_time}s")
            x_data, y_data = ws.window_slice(x_data, y_data, sfreq=self.sfreq, 
                                           window_time=window_time, step_time=step_time)
            print(f"[INFO] After window slice: x_data.shape={x_data.shape}, y_data.shape={y_data.shape}")

        for clfr in self.classifiers:
            clf_args = copy.deepcopy(self.setting_data["DL_classifier_method"][clfr])
            clf_args["n_classes"] = int(np.unique(y_data).shape[0])

            print(f"\n[INFO] Start testing: {clfr}")
            result = cf.test_DL(
                x_data,
                y_data,
                classifier=clfr,
                arg_dict=clf_args,
            )

            print(f"\n[RESULT] {clfr}:")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            self.setting_data["all_accuracy"][clfr] = result

        save_path = self._resolve_result_json_path(result_json_path)
        if save_path is not None:
            self.save_result_json(save_path)

        return self.setting_data["all_accuracy"]

# ═══════════════════════════════════════════
# parse_args
# ═══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Testing only script for MIEXP DL models")
    parser.add_argument(
        "--setting_path",
        type=str,
        default=os.path.join(BASE_DIR, "json", "setting_test.json"),
        help="Path to testing setting JSON",
    )
    parser.add_argument(
        "--pt_path",
        type=str,
        required=True,
        help="Path to processed .pt file containing x_data and y_data",
    )
    parser.add_argument(
        "--label_encoder_path",
        type=str,
        default=None,
        help="LabelEncoder path to load labels",
    )
    parser.add_argument(
        "--result_json_path",
        type=str,
        default=None,
        help="Optional override path to save testing result JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=60,
        help="Random seed",
    )
    return parser.parse_args()

# ═══════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════

def main():
    args = parse_args()
    log_file, original_stdout = setup_dual_logging()

    try:
        tester = Tester(setting_path=args.setting_path, seed=args.seed)
        tester.test_from_pt(
            pt_path=args.pt_path,
            label_encoder_path=args.label_encoder_path,
            result_json_path=args.result_json_path,
        )
    finally:
        sys.stdout = original_stdout
        log_file.close()

if __name__ == '__main__':
    main()