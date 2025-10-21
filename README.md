# Dự Án Phát Hiện Malware Dựa Trên Text

Dự án này sử dụng các mô hình học máy để phát hiện malware từ dữ liệu text, bao gồm XSS, SQL injection, và DDoS attacks.

## Cấu Trúc Dự Án

```
Project-A/
├── dataset/
│   ├── DDOS_dataset.csv
│   ├── extract_positives.py
│   ├── merged.csv
│   ├── merged.py
│   ├── merged_positive.csv
│   └── XSS_dataset.csv
└── model/
    ├── EfficientNetB0/
    │   ├── EfficientNetB0.py
    │   └── output/
    │       ├── EfficientNetB0_Augmented.keras
    │       └── evaluation_results.csv
    ├── MalwareDetection_Text.py
    ├── MobileViT/
    │   ├── MobileVit.py
    │   └── output/
    │       ├── evaluation_results.csv
    │       └── MobileViT_XXS_FromScratch.keras
    ├── mobilenetv2/
    │   ├── mobilenetv2.py
    │   └── output/
    │       ├── evaluation_results.csv
    │       ├── MobileNetV2_Augmented_NoAddons.h5
    │       ├── MobileNetV2_Augmented_v2.keras
    │       └── MobileNetV2.h5
    ├── output/
    │   ├── evaluation_results.csv
    │   ├── MalwareDetection_Text_LSTM.keras
    │   └── MalwareDetection_Text_LSTM_training_history.png
    ├── SqueezeNet/
    │   ├── SqueezeNet.py
    │   └── output/
    │       ├── evaluation_results.csv
    │       └── SqueezeNet_Standalone_Augmented.keras
    └── Swin_Transformer/
        ├── output/
        │   ├── evaluation_results.csv
        │   └── SwinTransformer_NewImpl_fixed.keras
        ├── swin_tiny_finetuned.pth
        └── Swin_Transformer.py
```

## Yêu Cầu Hệ Thống

- Python 3.11+
- TensorFlow 2.x
- PyTorch (cho Swin Transformer)
- pandas
- scikit-learn
- matplotlib

## Cài Đặt

1. Tạo virtual environment:
```bash
python -m venv .venv
```

2. Kích hoạt virtual environment:
```bash
.venv\Scripts\activate  # Windows
```

3. Cài đặt dependencies:
```bash
pip install tensorflow pandas scikit-learn matplotlib torch torchvision timm
```

## Chuẩn Bị Dữ Liệu

### 1. Hợp Nhất Dữ Liệu
Chạy script để hợp nhất các file CSV:
```bash
cd dataset
python merged.py
```
Script này sẽ tạo `merged.csv` từ `XSS_dataset.csv`, `Modified_SQL_Dataset.csv`, và `DDOS_dataset.csv`.

### 2. Lọc Dữ Liệu Positive
Để lọc các mẫu positive (Label=1):
```bash
python extract_positives.py
```
Script này tạo `merged_positive.csv` từ `merged.csv`.

## Huấn Luyện Mô Hình

### Mô Hình Text-Based Malware Detection
Chạy script chính để huấn luyện mô hình phát hiện malware từ text:
```bash
cd model
python MalwareDetection_Text.py
```

**Thông tin mô hình:**
- Kiến trúc: BiLSTM với Embedding
- Input: Text sequences (max 200 tokens)
- Output: Binary classification (Malware/Benign)
- Thời gian huấn luyện: ~7 phút trên CPU
- Độ chính xác: ~99.5%

**Output:**
- `MalwareDetection_Text_LSTM.keras`: Mô hình đã huấn luyện
- `evaluation_results.csv`: Kết quả đánh giá
- `MalwareDetection_Text_LSTM_training_history.png`: Biểu đồ huấn luyện

### Các Mô Hình Hình Ảnh Khác
Dự án cũng bao gồm các mô hình CNN cho phân loại hình ảnh:

- **EfficientNetB0**: `model/EfficientNetB0/EfficientNetB0.py`
- **MobileNetV2**: `model/mobilenetv2/mobilenetv2.py`
- **MobileViT**: `model/MobileViT/MobileVit.py`
- **SqueezeNet**: `model/SqueezeNet/SqueezeNet.py`
- **Swin Transformer**: `model/Swin_Transformer/Swin_Transformer.py`

Để chạy các mô hình này:
```bash
cd model/[ModelName]
python [ModelName].py
```

## Đánh Giá Mô Hình

Sau khi huấn luyện, kiểm tra kết quả trong file `evaluation_results.csv`:
- Accuracy, F1-score, Precision, Recall
- Thời gian huấn luyện và dự đoán
- Kích thước mô hình

## Sử Dụng Mô Hình Đã Huấn Luyện

Để sử dụng mô hình đã huấn luyện cho dự đoán:

```python
import tensorflow as tf
from tensorflow import keras

# Tải mô hình
model = keras.models.load_model('model/output/MalwareDetection_Text_LSTM.keras')

# Chuẩn bị text
text = "your text here"
# (Cần tiền xử lý text tương tự như trong script huấn luyện)

# Dự đoán
prediction = model.predict([text])
is_malware = prediction[0][0] > 0.5
```

## Ghi Chú

- Các mô hình hình ảnh yêu cầu dữ liệu hình ảnh, không phải text
- Mô hình text sử dụng TextVectorization để xử lý text
- Đảm bảo kích hoạt virtual environment trước khi chạy
- Thời gian huấn luyện có thể khác nhau tùy thuộc vào phần cứng

## Liên Hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng kiểm tra code và logs để debug.