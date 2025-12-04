import tensorflow as tf

# Đường dẫn đến file model của bạn
model_path = r"D:\Project A\model\output\MalwareDetection_Text_LSTM.keras"

try:
    print("🔍 Đang tải mô hình TensorFlow...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Mô hình đã tải thành công!")

    # In ra kiến trúc của mô hình để xác minh
    model.summary()

except Exception as e:
    print("❌ Lỗi khi tải mô hình:")
    print(e)
