# File: MobileVit_fixed.py
# Đã fix hoàn toàn lỗi "KerasTensor cannot be used..." cho Keras 3.x

import os
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

# Giảm log TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =========================
# CẤU HÌNH
# =========================
CONFIG = {
    "MODEL_NAME": "MobileViT_XXS_FromScratch",
    "IMG_HEIGHT": 256,
    "IMG_WIDTH": 256,
    "BATCH_SIZE": 16,
    "EPOCHS": 10,
    "TRAIN_DIR": r'D:\doan\dataset_split_past1/train',
    "VAL_DIR": r'D:\doan\dataset_split_past1/val',
    "TEST_DIR": r'D:\doan\dataset_split_past1/test',
    "OUTPUT_DIR": 'output'
}

# =========================
# CUSTOM LAYERS & UTILS
# =========================


class CustomCutoutLayer(layers.Layer):
    """Cutout đơn giản, chạy ở chế độ training."""

    def __init__(self, factor=0.4, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, images, training=False):
        if training:
            return tf.map_fn(self._cutout_image, images)
        return images

    def _cutout_image(self, image):
        img_shape = tf.shape(image)
        img_h = img_shape[0]
        img_w = img_shape[1]
        ch = img_shape[2]

        cut_h = tf.cast(tf.cast(img_h, tf.float32) * self.factor, tf.int32)
        cut_w = tf.cast(tf.cast(img_w, tf.float32) * self.factor, tf.int32)

        y0 = tf.random.uniform([], 0, img_h - cut_h, dtype=tf.int32)
        x0 = tf.random.uniform([], 0, img_w - cut_w, dtype=tf.int32)

        mask = tf.ones([img_h, img_w, ch], dtype=image.dtype)
        zero_patch = tf.zeros([cut_h, cut_w, ch], dtype=image.dtype)

        paddings = [[y0, img_h - y0 - cut_h], [x0, img_w - x0 - cut_w], [0, 0]]
        zero_mask = tf.pad(zero_patch, paddings,
                           mode="CONSTANT", constant_values=0.0)

        # Giữ nguyên pixel ngoài vùng cutout
        mask = tf.where(zero_mask == 0.0, 1.0, 0.0)
        return image * mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"factor": self.factor})
        return cfg


def get_device_info():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_info = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = gpu_info.get('device_name', 'GPU')
            print(f"Đã tìm thấy {len(gpus)} GPU. Sử dụng: {gpu_name}")
            return gpu_name
        except RuntimeError as e:
            print("Lỗi khởi tạo GPU:", e)
            return "GPU (lỗi khởi tạo)"
    print("Không tìm thấy GPU. Sử dụng CPU (có thể chậm).")
    return "CPU"


def load_and_prepare_datasets(augmentation_pipeline):
    print("\nĐang tải dữ liệu...")
    train_ds = keras.utils.image_dataset_from_directory(
        CONFIG["TRAIN_DIR"],
        shuffle=True,
        image_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
        batch_size=CONFIG["BATCH_SIZE"]
    )
    val_ds = keras.utils.image_dataset_from_directory(
        CONFIG["VAL_DIR"],
        shuffle=False,
        image_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
        batch_size=CONFIG["BATCH_SIZE"]
    )
    test_ds = keras.utils.image_dataset_from_directory(
        CONFIG["TEST_DIR"],
        shuffle=False,
        image_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
        batch_size=CONFIG["BATCH_SIZE"]
    )

    class_names = train_ds.class_names
    print(f"Đã tải xong. Tìm thấy {len(class_names)} lớp: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE

    # Aug chỉ áp lên train; đã gồm Rescaling bên trong pipeline
    train_ds = train_ds.map(
        lambda x, y: (augmentation_pipeline(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).cache().prefetch(AUTOTUNE)

    # Val/Test chỉ chuẩn hoá
    rescale = layers.Rescaling(1.0 / 255.0)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y),
                        num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y),
                          num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# =========================
# MOBILEViT ARCHITECTURE
# =========================


def conv_block(x, filters=16, kernel_size=3, strides=2):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=True,
                      activation="swish")(x)
    return x


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    """MobileNetV2-style Inverted Residual."""
    in_channels = x.shape[-1]  # Keras 3: lấy trực tiếp

    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)

    # Depthwise
    m = layers.DepthwiseConv2D(3, strides=strides,
                               padding="same" if strides == 1 else "same",
                               use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)

    # Project
    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    # Residual nếu cùng kênh và stride=1
    if (in_channels is not None) and (in_channels == output_channels) and (strides == 1):
        m = layers.Add()([x, m])
    return m


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="swish")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, projection_dim, num_heads=4):
    """Transformer encoder block tối giản (pre-norm)."""
    # x: (B, T, C)
    c = x.shape[-1]  # C (thường = projection_dim)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
    x2 = layers.Add()([x, attn_out])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # Dùng dựa trên projection_dim để tránh C=None
    x3 = mlp(x3, hidden_units=[projection_dim *
             2, projection_dim], dropout_rate=0.1)
    return layers.Add()([x2, x3])


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    """
    MobileViT block:
    - Local rep: conv to projection_dim
    - Global rep: reshape (H*W, C) -> transformer -> reshape back
    - Fuse back with 1x1 conv
    """
    # Nếu cần downsample trước block
    if strides == 2:
        ch = x.shape[-1]
        x = inverted_residual_block(x, expanded_channels=int(
            ch) * 2, output_channels=int(ch), strides=strides)

    # Local representation
    patches = layers.Conv2D(projection_dim, 3, strides=1,
                            padding="same", activation="swish")(x)

    # Lấy kích thước tĩnh (do input fixed 256x256)
    h = patches.shape[1]
    w = patches.shape[2]
    c = patches.shape[3]  # = projection_dim
    if any(dim is None for dim in [h, w, c]):
        raise ValueError(
            "H/W/C không xác định tĩnh. Hãy đảm bảo input size cố định và dùng image_size trong dataset.")

    # (B, H, W, C) -> (B, H*W, C)
    tokens = layers.Reshape((int(h) * int(w), int(c)))(patches)

    # Global representation via Transformer
    for _ in range(num_blocks):
        tokens = transformer_block(tokens, projection_dim)

    # (B, H, W, C)
    patches_back = layers.Reshape((int(h), int(w), int(c)))(tokens)

    # Fuse back
    out = layers.Conv2D(x.shape[-1], 1, padding="same")(patches_back)
    return out


def build_mobilevit_xxs(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], 3))
    x = conv_block(inputs, filters=16, kernel_size=3,
                   strides=2)               # 128x128
    x = inverted_residual_block(
        x, 32, 16, strides=1)                          # 128x128
    x = inverted_residual_block(
        x, 64, 24, strides=2)                          # 64x64
    x = inverted_residual_block(
        x, 80, 24, strides=1)                          # 64x64
    x = inverted_residual_block(
        x, 96, 48, strides=2)                          # 32x32
    x = mobilevit_block(x, num_blocks=2, projection_dim=64,
                        strides=1)         # 32x32
    x = inverted_residual_block(
        x, 160, 64, strides=2)                         # 16x16
    x = mobilevit_block(x, num_blocks=4, projection_dim=80,
                        strides=1)         # 16x16
    x = inverted_residual_block(
        x, 240, 80, strides=2)                         # 8x8
    x = mobilevit_block(x, num_blocks=3, projection_dim=96,
                        strides=1)         # 8x8
    x = conv_block(x, filters=320, kernel_size=1,
                   strides=1)                   # 8x8
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="MobileViT_XXS")

# =========================
# TRAINING / EVAL UTILS
# =========================


def plot_and_save_history(history, save_path):
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.title("Kết quả huấn luyện")
    plt.xlabel("Epoch")
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ huấn luyện tại: {save_path}")

# =========================
# MAIN
# =========================


def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    print("--- BẮT ĐẦU HUẤN LUYỆN ---")

    device_name = get_device_info()

    # Augmentation pipeline (có Rescaling đầu vào)
    augmentation_pipeline = keras.Sequential([
        layers.Rescaling(1.0 / 255.0),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(height_factor=(-0.2, 0.3), width_factor=(-0.2, 0.3)),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        layers.RandomContrast(0.3),
        CustomCutoutLayer(factor=0.3),
    ], name="data_augmentation_pipeline")

    train_ds, val_ds, test_ds, class_names = load_and_prepare_datasets(
        augmentation_pipeline)

    model = build_mobilevit_xxs(num_classes=len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()

    print("\nHuấn luyện model...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"]
    )
    t1 = time.time()
    train_secs = t1 - t0
    print(
        f"Huấn luyện xong trong {train_secs:.2f} giây ({train_secs/60:.2f} phút).")

    # Lưu biểu đồ & model
    fig_path = os.path.join(
        CONFIG["OUTPUT_DIR"], f'{CONFIG["MODEL_NAME"]}_training_history.png')
    plot_and_save_history(history, fig_path)

    model_path = os.path.join(
        CONFIG["OUTPUT_DIR"], f'{CONFIG["MODEL_NAME"]}.keras')
    model.save(model_path)
    print(f"Đã lưu model tại: {model_path}")

    # ========== TEST ==========
    print("\n--- BẮT ĐẦU TEST ---")
    loaded_model = keras.models.load_model(
        model_path,
        custom_objects={'CustomCutoutLayer': CustomCutoutLayer}
    )

    # Gom nhãn thật
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    # Dự đoán
    t2 = time.time()
    y_pred_probs = loaded_model.predict(test_ds)
    t3 = time.time()

    y_pred = np.argmax(y_pred_probs, axis=1)

    num_test_samples = len(y_true)
    total_pred_time = t3 - t2
    time_per_sample = total_pred_time / max(1, num_test_samples)

    # Metrics
    accuracy = float(np.sum(y_pred == y_true) / num_test_samples)
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    results = {
        "thuat_toan": CONFIG["MODEL_NAME"],
        "GPU": device_name,
        "thoi_gian_huan_luyen (s)": round(train_secs, 2),
        "thoi_gian_huan_luyen (m)": round(train_secs / 60, 2),
        "accuracy": f"{accuracy:.4f}",
        "F1_score": f"{f1:.4f}",
        "recall": f"{recall:.4f}",
        "precision": f"{precision:.4f}",
        "thoi_gian_du_doan_1_mau (s)": f"{time_per_sample:.6f}",
        "model_size (MB)": round(model_size_mb, 2),
        "ngay_thuc_hien": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
    for k, v in results.items():
        print(f"{k:<32}: {v}")
    print("-----------------------------------------")
    print("\nChi tiết theo từng lớp:")
    print(classification_report(y_true, y_pred,
          target_names=class_names, zero_division=0))

    csv_path = os.path.join(CONFIG["OUTPUT_DIR"], 'evaluation_results.csv')
    df = pd.DataFrame([results])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Đã cập nhật kết quả vào file: {csv_path}")
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"Đã lưu kết quả vào file mới: {csv_path}")

    print("\n--- TOÀN BỘ QUÁ TRÌNH HOÀN TẤT ---")


if __name__ == "__main__":
    main()
