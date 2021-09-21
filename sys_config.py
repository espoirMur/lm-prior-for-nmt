import os
from pathlib import Path

import torch

print("torch:", torch.__version__)
# print("Cuda:", torch.backends.cudnn.is_available())
print("CuDNN:", torch.backends.cudnn.version())
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
CACHING = False
# CACHING = False
RANDOM_SEED = 1618

BASE_DIR = Path.cwd()

MODEL_CNF_DIR = os.path.join(BASE_DIR, "configs")


TRAINED_PATH = os.path.join(BASE_DIR, "checkpoints")

DATA_DIR = BASE_DIR.joinpath("data")

EMBS_PATH = BASE_DIR.joinpath("data", "embeddings")


EXP_DIR = os.path.join(BASE_DIR, 'experiments')

MODEL_DIRS = ["models", "modules", "helpers"]

VIS = {
    "server": "http://localhost",
    "enabled": True,
    "port": 8097,
    "base_url": "/",
    "http_proxy_host": None,
    "http_proxy_port": None,
    "log_to_filename": os.path.join(BASE_DIR, "vis_logger.json")
}

VIS["enabled"] = 8097
VIS["server"] = "http://localhost"

























