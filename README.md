# 🧬 Implement-StarDist

## 1️⃣ Giới thiệu

Dự án **Implement-StarDist** tái hiện mô hình **StarDist** để phân đoạn nhân tế bào (cell nuclei segmentation) trên bộ dữ liệu **Data Science Bowl 2018 (DSB2018)**.  
Mô hình dựa trên U-Net và được điều chỉnh để dự đoán **xác suất vật thể (object probability map)** và **khoảng cách xuyên tâm (star-convex distances)**.

---

## 2️⃣ Cài đặt môi trường

### ⚙️ Yêu cầu hệ thống
- Python 3.10  
- GPU (nếu có CUDA)
- Hệ điều hành: Ubuntu / Windows / macOS

### 📦 Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## 3️⃣ Clone repository

```bash
git clone https://github.com/TaiTranDang145/Implement-StarDist.git
cd Implement-StarDist
```

---

## 4️⃣ Chuẩn bị dataset DSB2018

1. Tải dữ liệu từ Kaggle:  
   🔗 [https://www.kaggle.com/competitions/data-science-bowl-2018/data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)

2. Giải nén hai file:
   - `stage1_train.zip`
   - `stage1_test.zip`

3. Đặt vào thư mục `data/` của dự án:
```
Implement-StarDist/
│
├── data/
│   ├── stage1_train/
│   └── stage1_test/
```

---

## 5️⃣ Chạy huấn luyện (Train)

Huấn luyện mô hình từ đầu:
```bash
python train.py     --root data     --epochs 200     --batch-size 4     --img-size 256     --n-rays 32     --lr 1e-4
```

### 💾 Kết quả huấn luyện
- Checkpoint sẽ được lưu tại: `trained_models/`
  - `best_model.pt`: mô hình tốt nhất theo IoU
  - `last_model.pt`: mô hình sau epoch cuối
- Theo dõi tiến trình bằng TensorBoard:
  ```bash
  tensorboard --logdir tensorboard/
  ```

---

## 6️⃣ Chạy kiểm thử (Test)

Dùng mô hình đã train để dự đoán và trực quan hóa:
```bash
python test.py     --checkpoint trained_models/best_model.pt     --split val     --prob-thresh 0.5     --nms-thresh 0.4     --save-dir results
```

### 📊 Kết quả
Ảnh kết quả lưu tại:
```
results/
├── visualizations/
│   ├── result_000.png
├── comparison_000.png
```

---

## 7️⃣ Đánh giá mô hình (Evaluation)

Đánh giá định lượng mô hình bằng mAP và AP@IoU:
```bash
python evaluate.py     --checkpoint trained_models/best_model.pt     --split val     --prob-thresh 0.5     --nms-thresh 0.4     --save-dir evaluation_results
```

### 📈 Output:
```
evaluation_results/
├── evaluation_results.json
├── evaluation_report.txt
├── ap_curve.png
```

Ví dụ output:
```
Final Results:
  mAP (IoU 0.5:0.95): 0.7132
  AP50 (IoU 0.5):     0.8657
  AP75 (IoU 0.75):    0.6983
  Precision: 0.87 | Recall: 0.72 | F1-Score: 0.79
```

---

## 8️⃣ Cấu trúc thư mục dự án

```
Implement-StarDist/
│
├── data/
│   ├── stage1_train/
│   └── stage1_test/
│
├── models.py            # Định nghĩa kiến trúc mô hình StarDist
├── my_datasets.py       # Chuẩn bị dataset & star distance
├── post_processing.py   # Hậu xử lý & visualize
├── train.py             # Huấn luyện mô hình
├── test.py              # Kiểm thử và trực quan hóa
├── evaluate.py          # Đánh giá định lượng (mAP, AP50, ...)
├── requirements.txt
└── README.md
```

---

## 9️⃣ Gợi ý cải thiện

| Hướng cải thiện | Mô tả |
|------------------|--------|
| **Dice / Focal Loss** | Giúp mô hình tập trung vào vùng nhỏ, mất cân bằng |
| **Data Augmentation mạnh hơn** | Dùng RandomBrightnessContrast, ElasticTransform |
| **Pretrained backbone (ResNet)** | Fine-tune để hội tụ nhanh hơn |
| **Cosine LR Scheduler** | Điều chỉnh learning rate mềm mại hơn |
| **Multi-scale training** | Huấn luyện đa kích thước tế bào |

---
