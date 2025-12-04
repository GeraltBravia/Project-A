# Malware Detection - Multi-Class Analysis
# Phân tích đa nhãn: XSS, SQL Injection, DDoS

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
    datasets = {
        'XSS': (r'..\dataset\XSS_dataset.csv', 'Sentence'),
        'SQL': (r'..\dataset\Modified_SQL_Dataset.csv', 'Query'),
        'DDOS': (r'..\dataset\DDOS_dataset.csv', 'Sentence')  # Assuming DDoS uses Sentence
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
            print(f"Loaded {len(df)} samples from {source} (column: {text_col})")

    df_all = pd.concat(df_list, ignore_index=True)

    # Filter short texts but ensure minimum samples per class
    min_samples_per_class = 1000  # Minimum samples per attack type

    filtered_dfs = []
    for attack_type in df_all['attack_type'].unique():
        df_class = df_all[df_all['attack_type'] == attack_type]
        df_class_filtered = df_class[df_class['Sentence'].str.strip().str.split().str.len() > 2]

        if len(df_class_filtered) < min_samples_per_class:
            # If not enough filtered samples, take some shorter ones
            remaining = min_samples_per_class - len(df_class_filtered)
            df_short = df_class[df_class['Sentence'].str.strip().str.split().str.len() <= 2]
            df_class_filtered = pd.concat([df_class_filtered, df_short.head(remaining)])

        filtered_dfs.append(df_class_filtered)
        print(f"{attack_type}: {len(df_class_filtered)} samples after filtering")

    df_all = pd.concat(filtered_dfs, ignore_index=True)

    # Encode labels: XSS=0, SQL=1, DDoS=2
    le = LabelEncoder()
    df_all['attack_label'] = le.fit_transform(df_all['attack_type'])

    print(f"Total samples: {len(df_all)}")
    print(f"Label distribution: {df_all['attack_type'].value_counts()}")

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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Multi-Class Confusion Matrix')
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
    plt.title('Multi-Class ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved multi-class ROC curves to: {save_path}")

# Main analysis
def main():
    print("=== MALWARE DETECTION MULTI-CLASS ANALYSIS ===")

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

    # Create datasets
    def vectorize_text(text, label):
        return vectorize_layer(text), label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    train_ds = train_ds.map(vectorize_text).batch(32).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
    val_ds = val_ds.map(vectorize_text).batch(32).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    test_ds = test_ds.map(vectorize_text).batch(32).prefetch(AUTOTUNE)

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
        print(f"  Most confused with: {class_names[np.argmax(cm[:, i])]}")

    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == '__main__':
    main()