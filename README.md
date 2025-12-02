# UIDesigner

## Setup and Running

### Set up environment
1. Create virtual environment
```bash
python3 -m venv venv
```
2. Activate virtual environment
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```
### Install Requirements
```bash
pip install -r requirements.txt
```
### Training Loop
1. Running
```bash
python src/training/train.py --data_dir data --batch_size 32 --epochs 50 —> running
```
2. Running with custom settings
```bash
python src/training/train.py \
    --data_dir data \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.001 \
    --max_seq_len 512 \
    --hidden_dim 512 \
    --use_amp \
    --save_dir checkpoints
```
3. Resume training at checkpoint
```bash
python src/training/train.py --resume checkpoints/latest.pt (http://latest.pt)
```



