"""
Train Malware Detection Model
Script đơn giản để train model với tokenizer mới
"""

import os
import sys

# Set environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("🚀 MALWARE DETECTION MODEL - TRAINING")
print("="*70)

print("\n📦 Checking dependencies...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow {tf.__version__}")
except ImportError:
    print("   ❌ TensorFlow not found. Install: pip install tensorflow")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
    print("   ✅ All dependencies found")
except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")
    sys.exit(1)

# Import our modules
try:
    from tokenizer import MalwareTextTokenizer, create_tokenized_datasets
    print("   ✅ Tokenizer module loaded")
except ImportError:
    print("   ❌ Cannot import tokenizer module")
    sys.exit(1)

# Configuration
CONFIG = {
    "MODEL_NAME": "MalwareDetection_Text_LSTM",
    "MAX_TOKENS": 10000,
    "SEQUENCE_LENGTH": 200,
    "EMBEDDING_DIM": 128,
    "BATCH_SIZE": 128,
    "EPOCHS": 30,
    "OUTPUT_DIR": 'output'
}

print("\n⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   - {key}: {value}")

# Create output directory
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# Check GPU
print("\n🔍 Checking GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✅ Found {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"      - {gpu.name}")
else:
    print("   ⚠️  No GPU found, using CPU (slower)")

# Load data
print("\n" + "="*70)
print("📊 LOADING DATA")
print("="*70)

datasets = {
    'XSS': '../dataset/XSS_dataset.csv',
    'SQL': '../dataset/Modified_SQL_Dataset.csv',
    'DDOS': '../dataset/DDOS_dataset.csv'
}

df_list = []
for source, path in datasets.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['source'] = source
        df_list.append(df)
        print(f"✅ Loaded {len(df):,} samples from {source}")
    else:
        print(f"⚠️  Warning: {path} not found")

if not df_list:
    print("❌ No datasets loaded. Please check file paths.")
    sys.exit(1)

df_all = pd.concat(df_list, ignore_index=True)

# Separate DDoS
df_ddos = df_all[df_all['source'] == 'DDOS'].copy()
df_non_ddos = df_all[df_all['source'] != 'DDOS'].copy()

print(f"\n📈 Dataset Summary:")
print(f"   - DDoS samples: {len(df_ddos):,}")
print(f"   - Non-DDoS samples: {len(df_non_ddos):,}")
print(f"   - Total: {len(df_all):,}")

# Use non-DDoS for training (XSS + SQL)
df = df_non_ddos.copy()

# Remove duplicates
original_size = len(df)
df = df.drop_duplicates(subset=['Sentence'], keep='first')
removed = original_size - len(df)
if removed > 0:
    print(f"\n🔧 Removed {removed:,} duplicate samples ({removed/original_size*100:.1f}%)")

# Filter short texts
df = df[df['Sentence'].str.strip().str.split().str.len() > 2]

print(f"\n✅ Final dataset:")
print(f"   - Total: {len(df):,} samples")
print(f"   - Malware (1): {len(df[df['Label']==1]):,}")
print(f"   - Benign (0): {len(df[df['Label']==0]):,}")

# Prepare data
texts = df['Sentence'].fillna('').astype(str).values
labels = df['Label'].values

# Split data (70-15-15)
print("\n📂 Splitting data...")
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

print(f"   - Train: {len(train_texts):,} samples (70%)")
print(f"   - Val: {len(val_texts):,} samples (15%)")
print(f"   - Test: {len(test_texts):,} samples (15%)")

# Build tokenizer
print("\n" + "="*70)
print("🔤 BUILDING TOKENIZER")
print("="*70)

tokenizer = MalwareTextTokenizer(
    max_tokens=CONFIG["MAX_TOKENS"],
    sequence_length=CONFIG["SEQUENCE_LENGTH"]
)
tokenizer.build_vocabulary(train_texts)

# Create datasets
train_ds, val_ds, test_ds = create_tokenized_datasets(
    tokenizer,
    train_texts, train_labels,
    val_texts, val_labels,
    test_texts, test_labels,
    batch_size=CONFIG["BATCH_SIZE"]
)

# Save tokenizer
tokenizer_path = os.path.join(CONFIG["OUTPUT_DIR"], "tokenizer.pkl")
tokenizer.save(tokenizer_path)

# Build model
print("\n" + "="*70)
print("🏗️  BUILDING MODEL")
print("="*70)

from tensorflow import keras

model = keras.Sequential([
    keras.layers.Embedding(CONFIG["MAX_TOKENS"], CONFIG["EMBEDDING_DIM"]),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
], name="BiLSTM_MalwareDetection")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# Train model
print("\n" + "="*70)
print("🚀 TRAINING MODEL")
print("="*70)

import time
start_time = time.time()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG["EPOCHS"],
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time/60:.2f} minutes")

# Evaluate
print("\n" + "="*70)
print("📊 EVALUATING MODEL")
print("="*70)

# Get predictions
y_pred_probs = model.predict(test_ds, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Metrics
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)

print(f"\n📈 Test Results:")
print(f"   - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   - F1-Score:  {f1:.4f}")
print(f"   - Recall:    {recall:.4f}")
print(f"   - Precision: {precision:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(test_labels, y_pred, target_names=['Benign', 'Malware']))

# Save model
model_path = os.path.join(CONFIG["OUTPUT_DIR"], f"{CONFIG['MODEL_NAME']}.keras")
model.save(model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save results
results = {
    'Model': CONFIG['MODEL_NAME'],
    'Training Time (min)': f"{training_time/60:.2f}",
    'Accuracy': f"{accuracy:.4f}",
    'F1-Score': f"{f1:.4f}",
    'Recall': f"{recall:.4f}",
    'Precision': f"{precision:.4f}",
    'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

results_df = pd.DataFrame([results])
results_path = os.path.join(CONFIG["OUTPUT_DIR"], "evaluation_results.csv")

if os.path.exists(results_path):
    existing_df = pd.read_csv(results_path)
    results_df = pd.concat([existing_df, results_df], ignore_index=True)

results_df.to_csv(results_path, index=False)
print(f"💾 Results saved to: {results_path}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n📁 Output files:")
print(f"   - Model: {model_path}")
print(f"   - Tokenizer: {tokenizer_path}")
print(f"   - Results: {results_path}")
