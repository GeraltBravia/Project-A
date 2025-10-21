# File: SqueezeNet.py
# Phien ban hoan chinh, da sua loi 'AUTOTUNE' not defined

import os
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

# Tat cac log khong can thiet cua TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- PHAN 1: CAU HINH VA CAC HAM HO TRO ---

# --- Cau hinh cho mo hinh ---
CONFIG = {
    "MODEL_NAME": "SqueezeNet_Standalone_Augmented",
    "IMG_HEIGHT": 227,
    "IMG_WIDTH": 227,
    "BATCH_SIZE": 16,
    "EPOCHS": 25,
    "TRAIN_DIR": r'D:\doan\dataset_split_past1/train',
    "VAL_DIR": r'D:\doan\dataset_split_past1/val',
    "TEST_DIR": r'D:\doan\dataset_split_past1/test',
    "OUTPUT_DIR": 'output'
}

# THAY DOI 1: Dinh nghia AUTOTUNE o pham vi toan cuc
AUTOTUNE = tf.data.AUTOTUNE

# Dinh nghia lop Cutout o pham vi toan cuc


class CustomCutoutLayer(keras.layers.Layer):
    def __init__(self, factor=0.4, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, images, training=False):
        if training:
            return tf.map_fn(self.cutout_image, images)
        return images

    def cutout_image(self, image):
        img_shape = tf.shape(image)
        img_height = img_shape[0]
        img_width = img_shape[1]
        cutout_height = tf.cast(
            tf.cast(img_height, tf.float32) * self.factor, tf.int32)
        cutout_width = tf.cast(
            tf.cast(img_width, tf.float32) * self.factor, tf.int32)
        coord_y = tf.random.uniform(
            shape=[], minval=0, maxval=img_height - cutout_height, dtype=tf.int32)
        coord_x = tf.random.uniform(
            shape=[], minval=0, maxval=img_width - cutout_width, dtype=tf.int32)
        cutout_area = tf.zeros(
            [cutout_height, cutout_width, 3], dtype=image.dtype)
        paddings = [[coord_y, img_height - coord_y - cutout_height],
                    [coord_x, img_width - coord_x - cutout_width], [0, 0]]
        mask = tf.pad(cutout_area, paddings, "CONSTANT", constant_values=1.0)
        return image * mask

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


def get_device_info():
    # ... (ham nay giu nguyen)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_info = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = gpu_info.get('device_name', 'GPU')
            print(f"Da tim thay {len(gpus)} GPU. Se su dung: {gpu_name}")
            return gpu_name
        except RuntimeError as e:
            print(e)
            return "GPU (Loi khi khoi tao)"
    else:
        print("Khong tim thay GPU. Se su dung CPU (qua trinh co the rat cham).")
        return "CPU"


def load_and_prepare_datasets(augmentation_pipeline):
    # ... (ham nay giu nguyen)
    print("\nDang tai du lieu...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG["TRAIN_DIR"], shuffle=True, image_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
        batch_size=CONFIG["BATCH_SIZE"]
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG["VAL_DIR"], shuffle=False, image_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
        batch_size=CONFIG["BATCH_SIZE"]
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG["TEST_DIR"], shuffle=False, image_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
        batch_size=CONFIG["BATCH_SIZE"]
    )
    class_names = train_ds.class_names
    print(f"Da tai xong. Tim thay {len(class_names)} lop: {class_names}")

    # THAY DOI 2: Xoa dong dinh nghia AUTOTUNE o day
    # AUTOTUNE = tf.data.AUTOTUNE # <- XOA DONG NAY

    train_ds = train_ds.map(lambda x, y: (augmentation_pipeline(
        x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


def fire_module(x, fire_id, squeeze=16, expand=64):
    f_id = 'fire' + str(fire_id) + '_'
    channel_axis = -1
    x = keras.layers.Conv2D(squeeze, (1, 1), activation='relu',
                            padding='valid', name=f_id + 'squeeze1x1')(x)
    left = keras.layers.Conv2D(
        expand, (1, 1), activation='relu', padding='valid', name=f_id + 'expand1x1')(x)
    right = keras.layers.Conv2D(
        expand, (3, 3), activation='relu', padding='same', name=f_id + 'expand3x3')(x)
    x = keras.layers.Concatenate(
        axis=channel_axis, name=f_id + 'concat')([left, right])
    return x


def build_squeezenet_model(num_classes):
    inputs = keras.layers.Input(
        shape=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], 3))
    x = keras.layers.Conv2D(96, (7, 7), strides=(
        2, 2), padding='valid', activation='relu', name='conv1')(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), name='maxpool1')(x)
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = keras.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), name='maxpool4')(x)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = keras.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), name='maxpool8')(x)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = keras.layers.Dropout(0.5, name='drop9')(x)
    x = keras.layers.Conv2D(num_classes, (1, 1),
                            padding='valid', name='conv10')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Activation('softmax', name='softmax')(x)
    model = keras.Model(inputs, outputs, name='squeezenet')
    return model


def plot_and_save_history(history, save_path):
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.title("Ket qua Huan luyen")
    plt.xlabel("Epoch")
    plt.savefig(save_path)
    plt.close()
    print(f"Da luu bieu do ket qua huan luyen tai: {save_path}")

# --- PHAN 2: QUY TRINH CHINH ---


def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    print("--- BAT DAU GIAI DOAN HUAN LUYEN ---")

    device_name = get_device_info()

    preprocess_input_caffe = keras.layers.Lambda(
        lambda x: tf.keras.applications.imagenet_utils.preprocess_input(
            x, mode='caffe')
    )

    augmentation_pipeline = keras.Sequential([
        preprocess_input_caffe,
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.3),
        keras.layers.RandomZoom(height_factor=(-0.2, 0.3),
                                width_factor=(-0.2, 0.3)),
        keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        keras.layers.RandomBrightness(factor=0.3),
        keras.layers.RandomContrast(factor=0.3),
        CustomCutoutLayer(factor=0.3)
    ], name="data_augmentation_pipeline")

    train_ds, val_ds, test_ds, class_names = load_and_prepare_datasets(
        augmentation_pipeline)

    model = build_squeezenet_model(num_classes=len(class_names))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    print("\nBat dau huan luyen mo hinh...")
    start_time_train = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"]
    )

    end_time_train = time.time()
    training_time_s = end_time_train - start_time_train
    training_time_m = training_time_s / 60
    print(
        f"Huan luyen hoan tat trong {training_time_s:.2f} giay ({training_time_m:.2f} phut).")

    plot_path = os.path.join(
        CONFIG["OUTPUT_DIR"], f'{CONFIG["MODEL_NAME"]}_training_history.png')
    plot_and_save_history(history, plot_path)

    model_path = os.path.join(
        CONFIG["OUTPUT_DIR"], f'{CONFIG["MODEL_NAME"]}.keras')
    model.save(model_path)
    print(f"Da luu mo hinh da huan luyen tai: {model_path}")

    print("\n--- BAT DAU GIAI DOAN KIEM TRA (TEST) ---")

    print("Dang tai lai mo hinh da luu de kiem tra...")
    loaded_model = keras.models.load_model(
        model_path,
        custom_objects={'CustomCutoutLayer': CustomCutoutLayer}
    )

    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    print("Dang thuc hien du doan tren tap test...")
    start_time_pred = time.time()

    # Can ap dung chuan hoa Caffe cho tap test truoc khi du doan
    test_ds_preprocessed = test_ds.map(lambda x, y: (
        preprocess_input_caffe(x), y), num_parallel_calls=AUTOTUNE)
    y_pred_probs = loaded_model.predict(test_ds_preprocessed)

    end_time_pred = time.time()

    y_pred = np.argmax(y_pred_probs, axis=1)

    num_test_samples = len(y_true)
    total_pred_time = end_time_pred - start_time_pred
    time_per_sample = total_pred_time / num_test_samples

    accuracy = np.sum(y_pred == y_true) / num_test_samples
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    execution_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = {
        "thuat_toan": CONFIG["MODEL_NAME"],
        "GPU": device_name,
        "thoi_gian_huan_luyen (s)": round(training_time_s, 2),
        "thoi_gian_huan_luyen (m)": round(training_time_m, 2),
        "accuracy": f"{accuracy:.4f}",
        "F1_score": f"{f1:.4f}",
        "recall": f"{recall:.4f}",
        "precision": f"{precision:.4f}",
        "thoi_gian_du_doan_1_mau (s)": f"{time_per_sample:.6f}",
        "model_size (MB)": round(model_size_mb, 2),
        "ngay_thuc_hien": execution_date,
    }

    print("\n--- KET QUA DANH GIA TREN TAP TEST ---")
    for key, value in results.items():
        print(f"{key.replace('_', ' ').capitalize():<30}: {value}")
    print("-----------------------------------------")
    print("\nChi tiet theo tung lop:")
    print(classification_report(y_true, y_pred,
          target_names=class_names, zero_division=0))

    results_df = pd.DataFrame([results])
    csv_path = os.path.join(CONFIG["OUTPUT_DIR"], 'evaluation_results.csv')

    if os.path.exists(csv_path):
        results_df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Da cap nhat ket qua vao file: {csv_path}")
    else:
        results_df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"Da luu ket qua vao file moi: {csv_path}")

    print("\n--- TOAN BO QUA TRINH HOAN TAT ---")


if __name__ == '__main__':
    main()
