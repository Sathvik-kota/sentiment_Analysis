# 🚀 HACKATHON STEP 1: Advanced Setup with Transformers (5 minutes)
# State-of-the-art transformer fine-tuning approach

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import xgboost as xgb

# 🔥 TRANSFORMER LIBRARIES (Cutting-edge approach)
!pip install -q transformers torch accelerate datasets evaluate
!pip install -q scikit-learn xgboost

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset
import evaluate

# Set device for GPU acceleration (essential for transformers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Initialize DistilBERT tokenizer (research-proven optimal for hackathons)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("✅ Step 1: Advanced transformer setup complete!")
print("🚀 Ready for state-of-the-art sentiment analysis!")

commit_and_push(commit_message="Step 1: Advanced transformer setup with DistilBERT for competition-grade results")


