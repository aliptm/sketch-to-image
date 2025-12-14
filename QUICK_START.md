# Quick Start Guide

## 🚀 Run the Project in 4 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Training Data
```bash
python generate_training_data.py --num_samples 100
```
✅ Creates `dataset/train/sketches/` and `dataset/train/images/`

### Step 3: Train the Model
```bash
python train.py --data_dir ./dataset --epochs 50 --batch_size 4
```
✅ Saves model to `checkpoints/generator_best.pth`

### Step 4: Generate Images
```bash
python inference.py \
    --checkpoint checkpoints/generator_best.pth \
    --input_sketch dataset/train/sketches/sketch_0001.png \
    --output_path result.png
```
✅ Creates `result.png` with generated image

---

## 📊 What Happens During Training?

```
Epoch 1/50: 100%|████████| 25/25 [02:15<00:00]
Generator Loss: 2.3456
Discriminator Loss: 0.8234
Saved best model with loss: 2.3456
```

- **Checkpoints**: Saved every 10 epochs in `checkpoints/`
- **Samples**: Generated images saved in `logs/samples/`
- **Logs**: Training metrics in `logs/` (view with TensorBoard)

---

## 🎯 Common Commands

```bash
# Test if models work
python test_models.py

# Validate your dataset
python prepare_dataset.py --data_dir ./dataset --validate

# View training progress
tensorboard --logdir logs
# Then open http://localhost:6006
```

---

## ⚡ Quick Test (5 minutes)

Want to quickly test if everything works?

```bash
# 1. Generate small dataset
python generate_training_data.py --num_samples 10

# 2. Train for just 2 epochs (quick test)
python train.py --data_dir ./dataset --epochs 2 --batch_size 2

# 3. Generate test image
python inference.py \
    --checkpoint checkpoints/generator_best.pth \
    --input_sketch dataset/train/sketches/sketch_0001.png \
    --output_path test_result.png
```

---

## 📁 Project Files You'll Use

| File | Purpose |
|------|---------|
| `generate_training_data.py` | Create synthetic data |
| `train.py` | Train the model |
| `inference.py` | Generate images |
| `test_models.py` | Verify models work |
| `prepare_dataset.py` | Validate your data |

---

## 🐛 Troubleshooting

**Problem**: "ModuleNotFoundError"  
**Solution**: `pip install -r requirements.txt`

**Problem**: "Out of memory"  
**Solution**: Reduce `--batch_size` to 2 or 1

**Problem**: "No such file or directory"  
**Solution**: Run `python generate_training_data.py` first

---

## 📚 More Help

- **Detailed guide**: See [HOW_TO_RUN.md](HOW_TO_RUN.md)
- **Project overview**: See [README.md](README.md)
- **Interview prep**: See [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)

