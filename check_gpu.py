#!/usr/bin/env python3
"""
GPU Check Script for Malware Detection Model
Kiểm tra GPU và hướng dẫn cài đặt nếu cần
"""

import sys
import subprocess

def check_gpu():
    """Kiểm tra GPU và TensorFlow"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                gpu_info = tf.config.experimental.get_device_details(gpu)
                gpu_name = gpu_info.get('device_name', f'GPU {i}')
                print(f"   - {gpu_name}")
            print("🎉 Model sẽ training bằng GPU!")
            return True
        else:
            print("❌ Không tìm thấy GPU NVIDIA.")
            print("\n📋 Hướng dẫn cài đặt GPU support:")
            print("1. Kiểm tra card đồ họa NVIDIA:")
            print("   - Mở Device Manager > Display adapters")
            print("   - Nếu có NVIDIA GPU, tiếp tục bước 2")
            print("\n2. Cài đặt CUDA Toolkit 11.2:")
            print("   - Tải: https://developer.nvidia.com/cuda-11-2-0-download-archive")
            print("   - Chọn: Windows > exe (local) > Download")
            print("\n3. Cài đặt cuDNN 8.1:")
            print("   - Tải: https://developer.nvidia.com/cudnn")
            print("   - Đăng ký tài khoản NVIDIA (miễn phí)")
            print("   - Download cuDNN for CUDA 11.x")
            print("   - Giải nén và copy files vào C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2")
            print("\n4. Khởi động lại máy")
            print("\n5. Chạy lại script này")
            print("\n💡 Khuyến nghị: Sử dụng Google Colab để training với GPU miễn phí!")
            return False

    except ImportError:
        print("❌ TensorFlow chưa được cài đặt.")
        print("Chạy: pip install -r requirements.txt")
        return False

def check_cuda():
    """Kiểm tra CUDA toolkit"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'release' in line:
                    print(f"✅ CUDA: {line}")
                    break
        else:
            print("❌ CUDA toolkit chưa được cài đặt.")
    except FileNotFoundError:
        print("❌ CUDA toolkit chưa được cài đặt.")

if __name__ == "__main__":
    print("🔍 Kiểm tra GPU cho Malware Detection Model")
    print("=" * 50)

    check_cuda()
    gpu_available = check_gpu()

    if not gpu_available:
        print("\n⚠️  Model sẽ training bằng CPU (chậm hơn nhiều)")
        print("💡 Khuyến nghị sử dụng Google Colab với GPU!")

    print("\n" + "=" * 50)