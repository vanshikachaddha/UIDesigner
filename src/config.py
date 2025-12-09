DATA_ROOT = "data"          # folder that contains vocab.json, train.json, val.json, images/, tokens/
BATCH_SIZE = 8
IMG_SIZE = 224
MAX_SEQ_LEN = 120

EPOCHS = 35                 # change
LR = 1e-4
D_MODEL = 512
NHEAD = 8
DEC_LAYERS = 6
FF_DIM = 2048
DROPPUT = 0.1
RESNET_VERSION = "resnet50"
NO_PRETRAINED = False
FREEZE_BACKBONE = False
MAX_GRAD_NORM = 1.0
DROPOUT = 0.1
