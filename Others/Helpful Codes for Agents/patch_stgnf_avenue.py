"""Patch Pose Extraction and Testing.ipynb for ShanghaiTech + Avenue dataset selection."""

import json
import re
from pathlib import Path


NB_PATH = Path("Pose Extraction and Testing.ipynb")


def set_source(nb, idx, source: str) -> None:
    cell = nb["cells"][idx]
    cell["source"] = source.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Clear stale outputs to shrink the notebook.
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

set_source(
    nb,
    0,
    """# STG-NF Pose Extraction, Training, and Testing on Colab

This notebook extracts AlphaPose pose JSONs using the same command and conversion style as `orhir/STG-NF/gen_data.py`, trains/evaluates STG-NF, and exports frame-level scores.

Set **`DATASET_NAME`** in Section 1 to:
- `\"shanghaitech\"` — original ShanghaiTech Campus layout (train `.avi`, test frame folders)
- `\"avenue\"` — CUHK Avenue dataset from `Avenue_Dataset.zip` (train/test `.avi`)

Outputs are saved under separate Drive folders to avoid collisions:
- ShanghaiTech → `MyDrive/STG-NF/original_shanghaitech`
- Avenue → `MyDrive/STG-NF/Avenue_dataset`
""",
)

set_source(
    nb,
    1,
    """## 1. Mount Drive and Configure Paths

Set **`DATASET_NAME`** to `\"shanghaitech\"` or `\"avenue\"`. The cell below resolves archive paths, local extract roots, source video folders, ground-truth labels, and the Drive output root for the selected dataset.
""",
)

set_source(
    nb,
    2,
    """from google.colab import drive
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

assert torch.cuda.is_available(), "No GPU found. Use Runtime -> Change runtime type -> GPU."
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)

drive.mount("/content/drive")

REPO_URL = "https://github.com/orhir/STG-NF.git"
REPO_DIR = Path("/content/STG-NF")
ALPHAPOSE_DIR = Path("/content/AlphaPose")
# Pin to the AlphaPose commit observed in the matching Colab diagnostic so defaults do not drift.
ALPHAPOSE_COMMIT = "c60106d19afb443e964df6f06ed1842962f5f1f7"
LOCAL_POSE_WORK = Path("/content/stg_nf_alphapose_work")

# --- Dataset selection: "shanghaitech" or "avenue" ---
DATASET_NAME = "shanghaitech"

DATASET_PRESETS = {
    "shanghaitech": {
        "display_name": "ShanghaiTech Campus",
        "stg_nf_dataset_arg": "ShanghaiTech",
        "archive_path": Path("/content/drive/MyDrive/shanghaitech.tar.gz"),
        "extract_root": Path("/content"),
        "extract_marker": Path("/content/shanghaitech/training/videos"),
        "archive_is_zip": False,
        "original_data_root": Path("/content/shanghaitech"),
        "train_relative": Path("training/videos"),
        "test_relative": Path("testing/frames"),
        "train_source_mode": "video",
        "test_source_mode": "images",
        "drive_root": Path("/content/drive/MyDrive/STG-NF/original_shanghaitech"),
        "ground_truth_dir": None,
        "gt_repo_subdir": Path("data/ShanghaiTech/gt/test_frame_mask"),
        "expected_train": 330,
        "expected_test": 107,
    },
    "avenue": {
        "display_name": "CUHK Avenue",
        "stg_nf_dataset_arg": "Avenue",
        "archive_path": Path("/content/drive/MyDrive/Avenue_Dataset.zip"),
        "extract_root": Path("/content/avenue_dataset"),
        "extract_marker": Path("/content/avenue_dataset/Avenue Dataset/training_videos"),
        "archive_is_zip": True,
        "original_data_root": Path("/content/avenue_dataset/Avenue Dataset"),
        "train_relative": Path("training_videos"),
        "test_relative": Path("testing_videos"),
        "train_source_mode": "video",
        "test_source_mode": "video",
        "drive_root": Path("/content/drive/MyDrive/STG-NF/Avenue_dataset"),
        "ground_truth_dir": Path("/content/drive/MyDrive/ground_truth_avenue"),
        "gt_repo_subdir": Path("data/Avenue/gt/test_frame_mask"),
        "expected_train": 16,
        "expected_test": 21,
    },
}

DATASET_KEY = DATASET_NAME.lower().strip()
if DATASET_KEY not in DATASET_PRESETS:
    raise ValueError(
        f"Unsupported DATASET_NAME={DATASET_NAME!r}. Choose one of: {sorted(DATASET_PRESETS)}"
    )

DATASET_CONFIG = DATASET_PRESETS[DATASET_KEY]
DISPLAY_DATASET_NAME = DATASET_CONFIG["display_name"]
STG_NF_DATASET_ARG = DATASET_CONFIG["stg_nf_dataset_arg"]

ORIGINAL_DATA_ROOT = Path(DATASET_CONFIG["original_data_root"])
TRAIN_SOURCE_ROOT = ORIGINAL_DATA_ROOT / DATASET_CONFIG["train_relative"]
TEST_SOURCE_ROOT = ORIGINAL_DATA_ROOT / DATASET_CONFIG["test_relative"]
TRAIN_SOURCE_MODE = DATASET_CONFIG["train_source_mode"]
TEST_SOURCE_MODE = DATASET_CONFIG["test_source_mode"]

# Paper-style extraction: STG-NF paper says AlphaPose with YOLOX detector, then PoseFlow/pose tracking.
# Set False only if you intentionally want the literal repo gen_data.py command where YOLOX is commented out.
USE_YOLOX_DETECTOR = True
YOLOX_X_WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"

DRIVE_ROOT = Path(DATASET_CONFIG["drive_root"])
DRIVE_POSE_TRAIN = DRIVE_ROOT / "pose/train"
DRIVE_POSE_TEST = DRIVE_ROOT / "pose/test"
DRIVE_LOG_DIR = DRIVE_ROOT / "logs"
GROUND_TRUTH_DIR = DATASET_CONFIG["ground_truth_dir"]
GT_REPO_SUBDIR = Path(DATASET_CONFIG["gt_repo_subdir"])
EXPECTED_TRAIN_CLIPS = int(DATASET_CONFIG["expected_train"])
EXPECTED_TEST_CLIPS = int(DATASET_CONFIG["expected_test"])

for directory in [LOCAL_POSE_WORK, DRIVE_POSE_TRAIN, DRIVE_POSE_TEST, DRIVE_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Selected dataset:", DISPLAY_DATASET_NAME, f"({DATASET_KEY})")
print("STG-NF --dataset argument:", STG_NF_DATASET_ARG)
print("Repo target:", REPO_DIR)
print("AlphaPose target:", ALPHAPOSE_DIR)
print("Train source root:", TRAIN_SOURCE_ROOT)
print("Test source root:", TEST_SOURCE_ROOT)
print("Train source mode:", TRAIN_SOURCE_MODE)
print("Test source mode:", TEST_SOURCE_MODE)
print("Drive pose root:", DRIVE_ROOT)
if GROUND_TRUTH_DIR is not None:
    print("Ground-truth directory:", GROUND_TRUTH_DIR)
""",
)

# Find and replace dataset extraction cell (currently shanghaitech tar only).
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "shanghaitech.tar.gz" in src and cell["cell_type"] == "code":
        set_source(
            nb,
            idx,
            """ARCHIVE_PATH = Path(DATASET_CONFIG["archive_path"])
EXTRACT_ROOT = Path(DATASET_CONFIG["extract_root"])
EXTRACT_MARKER = Path(DATASET_CONFIG["extract_marker"])

if EXTRACT_MARKER.exists():
    print(f"Dataset already extracted: {EXTRACT_MARKER}")
else:
    assert ARCHIVE_PATH.exists(), f"Missing dataset archive: {ARCHIVE_PATH}"
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    if DATASET_CONFIG["archive_is_zip"]:
        print(f"Extracting {ARCHIVE_PATH} to {EXTRACT_ROOT} ...")
        !unzip -q "{ARCHIVE_PATH}" -d "{EXTRACT_ROOT}"
    else:
        print(f"Extracting {ARCHIVE_PATH} to {EXTRACT_ROOT} ...")
        !tar -xzvf "{ARCHIVE_PATH}" -C "{EXTRACT_ROOT}"

assert EXTRACT_MARKER.exists(), (
    f"Extraction finished but expected marker path is missing: {EXTRACT_MARKER}\\n"
    f"Check the archive layout for {DISPLAY_DATASET_NAME}."
)
print("Dataset archive ready:", EXTRACT_MARKER)
""",
        )
        if idx > 0 and nb["cells"][idx - 1]["cell_type"] == "markdown":
            set_source(
                nb,
                idx - 1,
                "## 1b. Extract Dataset Archive\n\nExtracts `shanghaitech.tar.gz` or `Avenue_Dataset.zip` from Google Drive on first run only.",
            )
        break

# Patch clip_id helper cell.
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "def clip_id_from_source(source, source_mode):" in src and "def list_video_sources" in src:
        src = src.replace(
            "def clip_id_from_source(source, source_mode):\n"
            "    source = Path(source)\n"
            "    return source.stem if source_mode == \"video\" else source.name\n",
            "def clip_id_from_source(source, source_mode):\n"
            "    source = Path(source)\n"
            "    if DATASET_KEY == \"avenue\":\n"
            "        # STG-NF expects scene_clip ids like 01_0005; Avenue videos are numbered 01..21.\n"
            "        video_num = int(source.stem)\n"
            "        return f\"01_{video_num:04d}\"\n"
            "    return source.stem if source_mode == \"video\" else source.name\n",
        )
        set_source(nb, idx, src)
        break

# Patch compatibility cell to register Avenue in args/scoring and prepare Avenue GT copies.
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "replace_if_present(\"dataset.py\"" in src and "Compatibility patch step complete" in src:
        extra = '''

def patch_once(path, old, new, label):
    path = Path(path)
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        print("patched", label, path)
    else:
        print("already ok", label, path)

# Allow Avenue in train_eval.py argument parser.
patch_once(
    "args.py",
    "choices=['ShanghaiTech', 'ShanghaiTech-HR', 'UBnormal'], help='Dataset for Eval'",
    "choices=['ShanghaiTech', 'ShanghaiTech-HR', 'UBnormal', 'Avenue'], help='Dataset for Eval'",
    "args Avenue choice",
)

# Route Avenue scoring to repo-local GT folder populated below.
patch_once(
    "utils/scoring_utils.py",
    "    else:\\n        per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask/'\\n        clip_list = os.listdir(per_frame_scores_root)",
    "    elif args.dataset == 'Avenue':\\n        per_frame_scores_root = 'data/Avenue/gt/test_frame_mask/'\\n        clip_list = os.listdir(per_frame_scores_root)\\n    else:\\n        per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask/'\\n        clip_list = os.listdir(per_frame_scores_root)",
    "scoring_utils Avenue gt root",
)

# Keep init_sub_args dataset mapping unchanged for Avenue (same loader path style as ShanghaiTech).

def prepare_repo_ground_truth():
    gt_dst = REPO_DIR / GT_REPO_SUBDIR
    gt_dst.mkdir(parents=True, exist_ok=True)
    if DATASET_KEY == "shanghaitech":
        src_root = REPO_DIR / "data/ShanghaiTech/gt/test_frame_mask"
        if src_root.exists():
            for src in src_root.glob("*.npy"):
                dst = gt_dst / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
        print("ShanghaiTech GT folder:", gt_dst)
        return gt_dst

    assert GROUND_TRUTH_DIR is not None, "GROUND_TRUTH_DIR is required for Avenue"
    assert Path(GROUND_TRUTH_DIR).exists(), f"Missing Avenue labels: {GROUND_TRUTH_DIR}"
    for video_num in range(1, 22):
        src = Path(GROUND_TRUTH_DIR) / f"{video_num}.npy"
        dst = gt_dst / f"01_{video_num:04d}.npy"
        if not src.exists():
            raise FileNotFoundError(f"Missing Avenue label file: {src}")
        shutil.copy2(src, dst)
    print("Prepared Avenue GT copies:", gt_dst)
    return gt_dst

prepare_repo_ground_truth()
'''
        set_source(nb, idx, src.rstrip() + extra)
        break

# Replace hard-coded ShanghaiTech in training / loader / export cells.
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    new_src = src
    new_src = new_src.replace('"--dataset", "ShanghaiTech"', '"--dataset", STG_NF_DATASET_ARG')
    new_src = new_src.replace("--dataset ShanghaiTech \\", "--dataset {STG_NF_DATASET_ARG} \\")
    new_src = new_src.replace(
        "frame-level anomaly scores for every ShanghaiTech test video",
        "frame-level anomaly scores for every test video in the selected dataset",
    )
    new_src = new_src.replace(
        'STG_NF_CHECKPOINT_PATH = "/content/drive/MyDrive/STG-NF/original_shanghaitech/logs/ShanghaiTech/Jun26_1803/Jun26_1806__checkpoint.pth.tar"',
        'STG_NF_CHECKPOINT_PATH = None  # Set to your trained checkpoint .pth.tar path after training',
    )
    if new_src != src:
        set_source(nb, idx, new_src)

# Update export markdown to mention both datasets.
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        if "ShanghaiTech_MULDE_Training_GMM.ipynb" in src:
            set_source(
                nb,
                idx,
                src.replace(
                    "MULDE export produced in `ShanghaiTech_MULDE_Training_GMM.ipynb`",
                    "MULDE export produced in the MULDE training notebook",
                ),
            )

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Patched {NB_PATH}")
