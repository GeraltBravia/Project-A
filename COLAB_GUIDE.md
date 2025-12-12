# 🚀 Hướng Dẫn Chạy Model Trên Google Colab

## 📋 Bước 1: Chuẩn Bị

### Upload Notebook
1. Truy cập: https://colab.research.google.com/
2. File > Upload notebook
3. Chọn file `Malware_Detection_Colab.ipynb` từ máy tính

### Bật GPU (Quan Trọng!)
1. Runtime > Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (khuyến nghị)
4. Nhấn **Save**

## 📊 Bước 2: Upload Dataset

Trong notebook, chạy cell "Upload Dataset" và upload 3 files:
- `XSS_dataset.csv`
- `Modified_SQL_Dataset.csv`
- `DDOS_dataset.csv`

**💡 Tip**: Nén 3 files thành 1 file zip để upload nhanh hơn!

## 🏃 Bước 3: Chạy Model

### Cách 1: Chạy Tất Cả (Khuyến Nghị)
```
Runtime > Run all
```

### Cách 2: Chạy Từng Cell
1. Chạy từng cell từ trên xuống dưới
2. Đợi cell hiện tại chạy xong (biểu tượng ✅)
3. Tiếp tục cell tiếp theo

## ⏱️ Thời Gian Training

- **Với GPU T4**: ~3-5 phút
- **Với CPU**: ~30-40 phút (không khuyến nghị)

## 📈 Kết Quả

Sau khi training xong, bạn sẽ có:

### 1. Metrics
- Accuracy: >99%
- F1-Score: >99%
- Precision & Recall
- ROC AUC

### 2. Visualizations
- Training history plots
- Confusion matrix
- ROC curve

### 3. Model File
- `MalwareDetection_Text_LSTM.keras`

## 📥 Bước 4: Download Kết Quả

Chạy cell cuối cùng để download:
- Model file (.keras)
- Evaluation results (.csv)
- All plots (.png)

Tất cả sẽ được nén trong file `malware_detection_results.zip`

## ⚠️ Lưu Ý Quan Trọng

### 1. Thời Gian Session
- Colab session: **12 giờ** (miễn phí)
- Sau 12h cần chạy lại
- **💡 Tip**: Download kết quả ngay sau khi training xong!

### 2. RAM & Disk
- RAM: 12GB (đủ cho model)
- Disk: 100GB temporary storage
- Colab tự động xóa files sau khi đóng session

### 3. GPU Quota
- Miễn phí: ~15-20 giờ GPU/tháng
- Nếu hết quota, đợi 24h hoặc nâng cấp Colab Pro

## 🔧 Troubleshooting

### ❌ "No GPU available"
**Giải pháp**:
1. Runtime > Change runtime type > GPU
2. Disconnect and delete runtime
3. Connect lại

### ❌ "Out of Memory"
**Giải pháp**:
1. Giảm BATCH_SIZE: 128 → 64
2. Runtime > Factory reset runtime

### ❌ "Session crashed"
**Giải pháp**:
1. Runtime > Factory reset runtime
2. Chạy lại từ đầu

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra GPU đã bật chưa
2. Kiểm tra 3 files CSV đã upload đúng chưa
3. Xem error message trong cell
4. Restart runtime và thử lại

## 🎯 Checklist

Trước khi chạy, đảm bảo:
- ✅ Đã upload notebook lên Colab
- ✅ Đã bật GPU (Runtime > Change runtime type)
- ✅ Đã upload 3 files CSV
- ✅ Đã kiểm tra GPU hoạt động (cell đầu tiên)

## 🚀 Bắt Đầu Ngay!

1. Upload `Malware_Detection_Colab.ipynb` lên Colab
2. Bật GPU
3. Upload 3 files CSV
4. Runtime > Run all
5. Đợi 3-5 phút
6. Download kết quả

**Chúc bạn training thành công! 🎉**
