# UIDesigner

## Setup and Running

* Set up environment
### Set up environment
1. Create virtual environment
```bash
python3 -m venv venv
@@ -126,11 +126,11 @@ python3 -m venv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```
* Install Requirements
### Install Requirements
```bash
pip install -r requirements.txt
```
* Running Training
### Training Loop
1. Running
```bash
python src/training/train.py --data_dir data --batch_size 32 --epochs 50 —> running
