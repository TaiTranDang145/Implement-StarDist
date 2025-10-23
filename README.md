# 🧬 Implement-StarDist

## 📁 1. Dataset Preparation (DSB2018)

Bộ dữ liệu được sử dụng trong dự án là **Data Science Bowl 2018 (DSB2018)** – bộ dữ liệu phổ biến cho bài toán **segmentation nhân tế bào**.

### 🔹 Bước 1. Tải dataset
Bạn cần tải dữ liệu tại liên kết Kaggle sau:

🔗 [Data Science Bowl 2018 - Kaggle](https://www.kaggle.com/competitions/data-science-bowl-2018/data)

Sau khi đăng nhập Kaggle và đồng ý điều khoản, tải về 2 tệp sau:
- `stage1_train.zip`
- `stage1_test.zip`

### 🔹 Bước 2. Giải nén và tổ chức thư mục
Giải nén 2 thư mục này rồi đặt chúng vào thư mục `data/` của dự án:

```
project_root/
│
├── data/
│   ├── stage1_train/
│   └── stage1_test/
│
├── my_datasets.py
├── models.py
├── train.py
├── requirements.txt
└── README.md
```

---

## ⚙️ 2. Cài đặt thư viện

Trước tiên, tạo môi trường ảo (nếu chưa có):

```bash
python3 -m venv venv
source venv/bin/activate
```

Sau đó cài toàn bộ thư viện cần thiết từ file `requirements.txt`:

```bash
pip install -r requirements.txt
```

Một số thư viện quan trọng:
- `torch`, `torchvision`: xây dựng và huấn luyện mô hình.
- `opencv-python`, `scikit-image`: xử lý ảnh y sinh.
- `albumentations`: augment dữ liệu.
- `tqdm`, `tensorboard`: theo dõi tiến trình huấn luyện.

---

## 🧠 3. Cấu trúc các file chính

### 🗂️ `my_datasets.py`
- Định nghĩa lớp `DSB2018Datasets` kế thừa `torch.utils.data.Dataset`.
- Chịu trách nhiệm **đọc ảnh gốc và mask**, resize, augment và trả về tensor.
- Quản lý chia tập train/val/test.

### 🧩 `models.py`
- Chứa định nghĩa mô hình **StarDist hoặc U-Net cải tiến**.
- Gồm các module `ResidualBlock`, `Encoder`, `Decoder`, hoặc các khối `StarConv`.
- Có thể mở rộng để huấn luyện các phiên bản khác nhau.

### 🚀 `train.py`
- Là file chính để huấn luyện mô hình.
- Chịu trách nhiệm:
  - Load dataset từ `my_datasets.py`
  - Khởi tạo mô hình từ `models.py`
  - Tạo optimizer, scheduler, criterion
  - Lưu checkpoint (`best_loss.pth`)
  - Ghi log TensorBoard để theo dõi loss và metric

Chạy train:
```bash
python train.py --root data --epochs 100 --batch_size 8
```

---

## 📊 4. Kết quả dự kiến

- Mô hình được huấn luyện để **phân vùng (segment) nhân tế bào** trên ảnh huỳnh quang.
- Đầu ra mỗi mẫu gồm:
  - Ảnh gốc
  - Mask dự đoán (đa giác hình sao)
  - Đánh giá IoU hoặc Dice Score

