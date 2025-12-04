# UI Designer


## Collaborators
---
| Name | Email |
|---|---|
| Leo Phung | leophung@bu.edu |
| Vanshika Chaddha| vchaddha@bu.edu |
| Isabelle Canty | icanty@bu.edu |
| Amrita Kohli | kohlia@bu.edu |
| Aakanksha Lambi | aalambi@bu.edu |

## 1. Goals and Motivation of Project

### Goal
This project aims to rebuild and modernize the Pix2Code architecture, a system that transforms UI screenshots into frontend code. We start by faithfully recreating the original 2017 architecture (CNN + LSTM), and then upgrade the model with modern components like Vision Transformers (ViT) and Transformer Decoders

* Take a screenshot of a mobile/web UI
* Generate clean DSL → convert to HTML/CSS
* Eventually upgrade the model for improved accuracy & generalization

### Motivation
UI design → code generation is one of the most hype computer vision + sequence modeling tasks. Automating this workflow saves hours of manual frontend work and opens the door for rapid prototyping tools.

This project helps us:

* Understand vision encoders (CNNs → ViT upgrade)
* Understand sequence decoders (LSTM → Transformer upgrade)
* Learn dataset processing pipelines
* Build a full end-to-end ML system from preprocessing → modeling → evaluation → generation

## 2. Project Architecture

### Phase 1 — Recreate the Original Pix2Code Architecture (Baseline)

The baseline model exactly follows the 2017 paper:

#### 1. CNN Encoder
Takes a 2D UI screenshot (224×224)
→ passes through convolutional layers
→ outputs a fixed-length feature vector (≈256 dims)

#### 2. LSTM Decoder
Consumes:
* previous output token
* CNN feature vector (as context)
* Predicts the next DSL token until sequence ends.

### 3. Training
* Teacher Forcing
* Cross-Entropy Loss
* BLEU Score evaluation
* DSL → HTML generation for qualitative evaluation

### Phase 2 — Modern Architecture Upgrade (Goal)
Once the baseline is working:

### Upgrade Encoder → ViT (Vision Transformer)
Why?
* Better global understanding
* Better performance on structured UIs
* Eliminates convolution bottlenecks

### Upgrade Decoder → Transformer Decoder
Why?
* Better long-range context modeling
* Handles complex DSL sequences
* Industry standard in seq-to-seq tasks

### Add Attention → Cross Attention
Why?
* Referencing past tokens before generating new ones
* Instead of having one vector -> flatten to patches
* Better inferences rather than model trying to remember 

## 3. Team Responsiblity

### Data Preprocessing
* Image normalization
* DSL cleaning & tokenization
* Vocabulary building
* Creating train/val/test splits

### Model Implementation
* CNN Encoder
* LSTM Decoder
  
### Training Pipeline
* Forward pass
* Loss functions
* Teacher forcing
* Evaluation metrics

### Evaluation + Generation
* Convert predicted DSL → HTML/CSS
* Compare against ground truth
* Calculate BLEU scores
* Produce demo outputs

## 4. Evaluation
### We evaluate models using:
* Token-level accuracy
* BLEU score (reproduction of DSL sequence)
* Render comparison (visual matching HTML outputs)

## 5. Dependencies
* Python
* PyTorch
* NumPy / Pandas
* Matplotlib / Seaborn
* OpenCV
* TorchVision
* 

## 6. FINAL GOALS
* ✔️ A fully working Pix2Code baseline
* ✔️ A modernized Transformer-based UI-to-Code pipeline
* ✔️ Automatic HTML/CSS generation
* ✔️ A full ML workflow from preprocessing → inference
* ✔️ A polished, documented GitHub repo

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


