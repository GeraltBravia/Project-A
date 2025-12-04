# Phân Tích Mô Hình Phát Hiện Malware

## 1. Bài Toán Phân Loại

### Phân Loại Nhị Phân (Binary Classification)
- **Định nghĩa**: Phân loại thành 2 lớp (malware vs benign)
- **Ứng dụng**: Phát hiện có malware hay không
- **Ví dụ**: Label = 1 (malware), Label = 0 (benign)

### Phân Loại Đa Nhãn (Multi-Class Classification)
- **Định nghĩa**: Phân loại thành nhiều lớp (>2)
- **Ứng dụng**: Phân biệt loại attack (XSS, SQL, DDoS)
- **Ví dụ**: XSS=0, SQL=1, DDoS=2

## 2. Phương Pháp Trích Xuất Dữ Liệu

### Text Vectorization
```python
vectorize_layer = keras.layers.TextVectorization(
    max_tokens=10000,        # Vocab size
    output_mode='int',       # Integer encoding
    output_sequence_length=200  # Fixed length
)
```

### BiLSTM Architecture
```
Input Text → Embedding → BiLSTM → Dense → Output
```

- **Embedding**: Chuyển từ thành vector
- **BiLSTM**: Học ngữ cảnh hai chiều
- **Dense**: Classification layer

## 3. Metrics Đánh Giá

### Binary Classification
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * Precision * Recall / (Precision + Recall)

### Multi-Class Classification
- **Macro Average**: Trung bình cộng các lớp
- **Weighted Average**: Trung bình có trọng số theo support

## 4. Sơ Đồ Đánh Giá

### Confusion Matrix
- Hiển thị dự đoán đúng/sai cho từng lớp
- Diagonal: True Positives
- Off-diagonal: Errors

### ROC Curve
- True Positive Rate vs False Positive Rate
- AUC càng cao càng tốt (1.0 = perfect)

## 5. Lỗ Hổng Trong Mô Hình

### Binary Classification Issues
- **False Positives**: Báo malware khi là benign
- **False Negatives**: Bỏ sót malware thật
- **Class Imbalance**: Ít malware hơn benign

### Multi-Class Issues
- **Class Confusion**: Nhầm lẫn giữa các loại attack
- **Rare Classes**: Một số attack ít mẫu
- **Overfitting**: Học quá tốt trên train, kém trên test

## 6. Cách Chạy Phân Tích

### Binary Classification (mặc định)
```bash
cd model
python MalwareDetection_Text.py
```

### Multi-Class Analysis
```bash
cd model
python multiclass_analysis.py
```

### Output Files
- `*_training_history.png`: Learning curves
- `*_confusion_matrix.png`: Confusion matrix
- `*_roc_curve.png`: ROC curves
- `evaluation_results.csv`: Metrics summary

## 7. Cải Thiện Mô Hình

### Đối với Binary
- Thêm data augmentation
- Sử dụng pretrained embeddings
- Thêm regularization

### Đối với Multi-Class
- Oversampling rare classes
- Sử dụng class weights
- Thêm cross-validation

## 8. Kết Luận

- **Binary**: Tốt cho detection yes/no
- **Multi-Class**: Tốt cho phân loại chi tiết
- **Trade-off**: Accuracy vs Granularity

Chọn phương pháp phù hợp với use case cụ thể!