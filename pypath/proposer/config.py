import os

from baseline import PROPOSER_BASELINE, REPO_ROOT


LOCAL_ONLY = False

# ---------------------------
#        DIRECTORIES
# ---------------------------

PROJECT_LOCATION = REPO_ROOT
PYCODE_LOCATION = os.path.join(PROJECT_LOCATION, "pypath")
COMPILE_LOCATION = os.path.join(PROJECT_LOCATION, "release")
DATA_LOCATION = os.path.join(PROJECT_LOCATION, "experiments", "default", "proposer")

CDL_PATH = os.path.join(PROJECT_LOCATION, "40nm", "scc40nll_vhsc50_rvt.cdl")
LIB_PATH = os.path.join(PROJECT_LOCATION, "40nm", "l0040ll_v1p15.lib")
NGSPICE_EXECUTABLE = os.environ.get("WMPC_NGSPICE_EXECUTABLE", os.path.join(COMPILE_LOCATION, "src", "ngspice"))

JSON_DIR = "training_data"
NETLIST_DIR = "generated_netlists"
TRAJECTORY_DIR = "trajectory"
FAILED_NETLIST_DIR = "failed_netlists"
CACHE_DIR = "sequence_data"

# LOCAL DIR AND PATH
LOCAL_DATA_LOCATION = os.path.join(DATA_LOCATION, "legacy")
PROPOSER_LEGACY_DATA_LOCATION = LOCAL_DATA_LOCATION
PROPOSER_SOURCE_DATA_LOCATION = os.path.join(DATA_LOCATION, "source_runs")
PROPOSER_MEDIUM_DATA_LOCATION = os.path.join(DATA_LOCATION, "datasets")
PROPOSER_MODEL_LOCATION = os.path.join(DATA_LOCATION, "models")
PROPOSER_EVAL_LOCATION = os.path.join(DATA_LOCATION, "evaluation")
PROPOSER_DOCS_LOCATION = os.path.join(DATA_LOCATION, "docs")
DEFAULT_SOURCE_RUN_NAME = "independent_midscale_500circuits_v1"
DEFAULT_MEDIUM_DATASET_TAG = "independent_midscale_500circuits_v1_onehop_segment_entry"
DEFAULT_TRAIN_RUN_TAG = f"{DEFAULT_MEDIUM_DATASET_TAG}_lg_e80"

# 两个独立的缓存文件，用于训练集和测试集
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_dataset.pkl")
TEST_CACHE_PATH = os.path.join(CACHE_DIR, "test_dataset.pkl")

VALUE_DIR = "value"
MODEL_DIR = "model"

# GLOBAL DIR AND PATH
GLOBAL_DATA_LOCATION = os.path.join(DATA_LOCATION, "aggregator")

# ---------------------------
#         UUT INFO
# ---------------------------

UUT_LIST = ["AND2V0_12TR50"]
BUILDING_BLOCK_CELL = {
    "type": "AND2V0_12TR50",
    "inputs": ["A1", "A2"],
    "outputs": ["Z"],
    "pins": ["A1", "A2", "Z", "VDD", "VSS"],
}

NEIGHBOR_CELL_POOL = {
    "AND2V0_12TR50": {
        "inputs": ["A1", "A2"],
        "outputs": ["Z"],
        "pins": ["A1", "A2", "Z", "VDD", "VSS"],
    }
}

NUM_CIRCUITS_PER_UUT = PROPOSER_BASELINE.num_circuits_per_uut
DEFAULT_LOCAL_VIEW_SIZE = 32
DEFAULT_LOCAL_VIEW_STRIDE = 16
DEFAULT_MIN_TRAJECTORY_LENGTH = 5

# ---------------------------
#          SCALER
# ---------------------------
V_SCALER_TYPE = "standard"
F_SCALER_TYPE = "log"

# ---------------------------
#       MODEL PARAMS
# ---------------------------

BATCH_SIZE = PROPOSER_BASELINE.preprocess_batch_size
LOOK_BACK = PROPOSER_BASELINE.look_back
TEST_SPLIT_RATIO = PROPOSER_BASELINE.test_split_ratio
