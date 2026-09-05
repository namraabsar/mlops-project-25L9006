# MLOps Project - 25L9006

## Description
This project implements a machine learning pipeline for house price prediction, with version control using Git and GitHub.

## Project Structure
mlops-project-25L9006/
├── data/ # Raw dataset (ignored in git)
├── src/ # Training scripts
├── model/ # Saved trained model (ignored in git)
├── requirements.txt
└── README.md


## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the training script
```bash
python src/train_25L9006.py
```

This will load the dataset from `data/dataset.csv`, train a Random Forest model, and save it to `model/trained_model.pkl`.