# Malware Detection - Multi-Class Analysis (XSS vs SQL)
# DDoS dataset contains network traffic data, not suitable for text classification

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_multiclass_data():
    # Only XSS and SQL for text-based classification
    datasets = {
        'XSS': (r'..\dataset\XSS_dataset.csv', 'Sentence'),
        'SQL': (r'..\dataset\Modified_SQL_Dataset.csv', 'Query')
    }

    df_list = []
    for source, (path, text_col) in datasets.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['attack_type'] = source
            # Rename text column to 'Sentence' for consistency
            if text_col != 'Sentence':
                df = df.rename(columns={text_col: 'Sentence'})
            df_list.append(df)
            print(f"Loaded {len(df)} samples from {source}")

    df_all = pd.concat(df_list, ignore_index=True)

    # Filter short texts and handle empty strings
    df_all = df_all[df_all['Sentence'].notna()]  # Remove NaN
    df_all = df_all[df_all['Sentence'].str.strip() != '']  # Remove empty
    df_all = df_all[df_all['Sentence'].str.strip().str.split().str.len() > 2]  # At least 3 words

    # Encode labels: XSS=0, SQL=1
    le = LabelEncoder()
    df_all['attack_label'] = le.fit_transform(df_all['attack_type'])

    print(f"Total samples after filtering: {len(df_all)}")
    print(f"Label distribution: {df_all['attack_type'].value_counts()}")
    print("Note: DDoS dataset contains network traffic data, not suitable for text classification")

    return df_all, le

# Build multi-class model
def build_multiclass_model(vocab_size, num_classes):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, 128),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')  # Multi-class
    ])
    return model

# Plot confusion matrix for multi-class
def plot_multiclass_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Multi-Class Confusion Matrix (XSS vs SQL)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved multi-class confusion matrix to: {save_path}")

# Plot ROC curves for multi-class
def plot_multiclass_roc(y_true, y_pred_probs, class_names, save_path):
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves (XSS vs SQL)')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved multi-class ROC curves to: {save_path}")

# Main analysis
def main():
    print("=== MALWARE DETECTION MULTI-CLASS ANALYSIS (XSS vs SQL) ===")

    # Load data
    df, label_encoder = load_multiclass_data()
    class_names = label_encoder.classes_

    # Prepare text data
    texts = df['Sentence'].values
    labels = df['attack_label'].values

    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Text vectorization
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=10000, output_mode='int', output_sequence_length=200)
    vectorize_layer.adapt(train_texts)

    # Debug: Check vectorization
    sample_vectorized = vectorize_layer(train_texts[:5])
    print(f"Sample vectorized shape: {sample_vectorized.shape}")
    print(f"Sample vectorized: {sample_vectorized.numpy()}")

    # Create datasets
    def vectorize_text(text, label):
        vec = vectorize_layer(text)
        # Ensure no empty sequences
        return vec, label

    # Create datasets
    def vectorize_text(text, label):
        vec = vectorize_layer(text)
        # Filter out empty sequences (all zeros)
        mask = tf.reduce_any(vec != 0)  # Check if sequence has any non-zero tokens
        return vec, label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    train_ds = train_ds.map(vectorize_text).filter(lambda x, y: tf.reduce_any(x != 0)).batch(32).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
    val_ds = val_ds.map(vectorize_text).filter(lambda x, y: tf.reduce_any(x != 0)).batch(32).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    test_ds = test_ds.map(vectorize_text).filter(lambda x, y: tf.reduce_any(x != 0)).batch(32).prefetch(AUTOTUNE)

    # Build and train model
    model = build_multiclass_model(10000, len(class_names))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining multi-class model...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

    # Evaluate
    print("\nEvaluating on test set...")
    y_true = []
    y_pred_probs = []

    for batch_texts, batch_labels in test_ds:
        y_true.extend(batch_labels.numpy())
        preds = model.predict(batch_texts, verbose=0)
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Results
    print("\n=== MULTI-CLASS CLASSIFICATION RESULTS ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save plots
    os.makedirs('output', exist_ok=True)

    cm_path = 'output/multiclass_confusion_matrix.png'
    plot_multiclass_confusion_matrix(y_true, y_pred, class_names, cm_path)

    roc_path = 'output/multiclass_roc_curves.png'
    plot_multiclass_roc(y_true, y_pred_probs, class_names, roc_path)

    # Analyze vulnerabilities
    print("\n=== VULNERABILITY ANALYSIS ===")
    cm = confusion_matrix(y_true, y_pred)

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        print(f"\n{class_name} Detection:")
        print(f"  True Positives: {tp}")
        print(f"  False Negatives: {fn} (missed {class_name})")
        print(f"  False Positives: {fp} (false alarms)")
        if len(class_names) > 1:
            other_classes = [c for c in class_names if c != class_name]
            most_confused = other_classes[np.argmax([cm[j, i] for j in range(len(class_names)) if j != i])]
            print(f"  Most confused with: {most_confused}")

    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == '__main__':
    main()