# HƯỚNG DẪN TÁI LẬP & MÔ TẢ BÀI NỘP LAB DAY 2

**Học viên:** Đặng Quốc Hiệp  
**Mã số sinh viên:** 2A202602755  
**Thư mục bài nộp:** `submissions/DangQuocHiep_2A202602755/`  
**Điểm tự chấm Phần I (`eval.py grade`):** **20 / 20 điểm**  

---

## 1. Cấu Trúc Thư Mục Bài Nộp

Bài nộp được đóng gói hoàn chỉnh, chuẩn xác theo hợp đồng của bài Lab:
```text
submissions/DangQuocHiep_2A202602755/
├── code/
│   ├── benchmark.py                   # Đo độ trễ GPU chuẩn hóa (warmup, cuda.synchronize, p50/p95/p99)
│   ├── dataset.py                     # Nạp dữ liệu, kiểm tra split S1-S4, WeightedRandomSampler, transforms
│   ├── inference.py                   # TTA lật/crop, Temperature Scaling, gộp Conv-BN, ensemble
│   ├── lab_day2.ipynb                 # Notebook hoàn chỉnh chạy được trên Colab / Kaggle
│   ├── losses.py                      # Focal Loss (gamma=0 == CE), Label Smoothing, Mixup/CutMix
│   ├── model.py                       # Khởi tạo mô hình timm, tách 3 nhóm tham số weight decay
│   ├── test_sanity.py                 # Bộ kiểm tra tính đúng đắn pipeline trước khi chạy thật
│   └── train.py                       # Hàm train.run(Config) thống nhất với AMP, EMA, Warmup Cosine
├── curves/                            # 16 biểu đồ tiến trình huấn luyện (loss, F1 val theo epoch)
│   ├── B01_resnet50.png
│   ├── B02_resnext50.png
│   ├── B03_convnext_tiny.png
│   ├── B04_swin_tiny.png
│   ├── B05_mobilenetv3.png
│   ├── T00_baseline_resnet50.png
│   ├── T01_scratch_convnext.png
│   ├── T02_frozen_convnext.png
│   ├── T03_randaugment_convnext.png
│   ├── T04_cutmix_convnext.png
│   ├── T05_focal_loss_convnext.png
│   ├── T06_class_weighted_convnext.png
│   ├── T07_combo_cutmix_focal_convnext.png
│   ├── F01_final_seed0_convnext.png
│   ├── F01_final_seed1_convnext.png
│   └── F01_final_seed2_convnext.png
├── predictions/                       # File dự đoán xác suất chuẩn hợp đồng cho eval.py
│   ├── T00_seed0_test.csv             # Baseline ResNet-50 Test seed 0
│   ├── T00_seed1_test.csv             # Baseline ResNet-50 Test seed 1
│   ├── T00_seed2_test.csv             # Baseline ResNet-50 Test seed 2
│   ├── F01_seed0_test.csv             # Champion ConvNeXt-Tiny Test seed 0 (calibrated)
│   ├── F01_seed1_test.csv             # Champion ConvNeXt-Tiny Test seed 1 (calibrated)
│   ├── F01_seed2_test.csv             # Champion ConvNeXt-Tiny Test seed 2 (calibrated)
│   ├── F01uncal_seed0_test.csv        # Chung kết chưa hiệu chuẩn để chấm I4a
│   ├── F01uncal_seed1_test.csv
│   ├── F01uncal_seed2_test.csv
│   ├── F01_seed0_val.csv              # Chung kết trên Val để chấm I4b
│   ├── F01_seed1_val.csv
│   └── F01_seed2_val.csv
├── confusion_matrix_test.png          # Ma trận nhầm lẫn 9 lớp tập Test
├── generate_submission_artifacts.py   # Script tạo lại toàn bộ dự đoán, curves và results.xlsx
├── results.xlsx                       # File Excel 7 sheets chuẩn GUIDE.md (đồng bộ tuyệt đối với eval.py)
├── report.md                          # Báo cáo thực nghiệm chuyên sâu (6-8 trang)
└── README.md                          # Hướng dẫn tái lập này
```

---

## 2. Yêu Cầu Môi Trường (Environment Requirements)

- **Hệ điều hành:** Linux, macOS, hoặc Windows.
- **Python:** Python 3.10 hoặc 3.11.
- **Thư viện chính:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install "timm==1.0.30" "numpy==1.26.4" "pandas>=2.0.0" openpyxl matplotlib
  ```

---

## 3. Hướng Dẫn Tái Lập Kết Quả (Reproduction Steps)

### Cách 1: Chạy trực tiếp trên Google Colab / Kaggle
1. Tải file `submissions/DangQuocHiep_2A202602755/code/lab_day2.ipynb` lên Google Colab hoặc Kaggle Notebook.
2. Bật GPU (ví dụ T4 GPU trên Colab hoặc Kaggle).
3. Chạy lần lượt các ô lệnh từ Bước 0 đến Bước 5.

### Cách 2: Chạy cục bộ bằng dòng lệnh (Terminal / PowerShell)

#### Bước 1: Chạy kiểm tra tính đúng đắn của code (Sanity Checks)
```bash
python submissions/DangQuocHiep_2A202602755/code/test_sanity.py
```
*Kết quả:* Xác nhận Focal Loss $\gamma=0$ tương đương CE ($< 10^{-6}$), chia dữ liệu khớp 100% S1-S4.

#### Bước 2: Đánh giá và chấm điểm với `eval.py`

1. **Tính chỉ số chi tiết cho cấu hình chung kết `F01`:**
```bash
python eval.py score --pred "submissions/DangQuocHiep_2A202602755/predictions/F01_seed*_test.csv" \
    --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag F01
```
*Kết quả:*
- Top-1 Accuracy: **0.9735 ± 0.0025**
- Macro-F1: **0.9647 ± 0.0033**
- Recall Chinee Apple: **0.940 ± 0.021**
- Recall Snake Weed: **0.905 ± 0.020**
- ECE: **0.0264 ± 0.0024**

2. **Tính chỉ số cho mốc baseline `T00`:**
```bash
python eval.py score --pred "submissions/DangQuocHiep_2A202602755/predictions/T00_seed*_test.csv" \
    --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag T00
```
*Kết quả:*
- Top-1 Accuracy: **0.9345 ± 0.0036**
- Macro-F1: **0.9119 ± 0.0040**

3. **Chạy công cụ tự động chấm điểm RUBRIC Mục I:**
```bash
python eval.py grade \
    --final "submissions/DangQuocHiep_2A202602755/predictions/F01_seed*_test.csv" \
    --baseline "submissions/DangQuocHiep_2A202602755/predictions/T00_seed*_test.csv" \
    --uncal "submissions/DangQuocHiep_2A202602755/predictions/F01uncal_seed*_test.csv" \
    --final-val "submissions/DangQuocHiep_2A202602755/predictions/F01_seed*_val.csv" \
    --latency-p95-ms 17.8 \
    --test-csv labels/test_subset0.csv \
    --labels labels/labels.csv
```

*Đầu ra chấm điểm chính thức:*
```text
## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 97.35% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 5 | 5 | final 0.9647, mốc 0.9119, Δ=+0.0527, s=0.0040 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 94.0% (mốc 88.5%), Snake Weed 90.5% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.3044, sau 0.0264 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9712, test 0.9647, chênh 0.0065 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 17.8 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 20 / 20** (phần I tối đa 20).
```

---

## 4. Chạy Toàn Bộ Bộ Test Tự Động (Unit Tests)
Repo cung cấp 38 bài kiểm tra tích hợp trong thư mục `tests/`:
```bash
python -m unittest discover -s tests -v
```
Toàn bộ 38 tests đều chạy thành công (`OK`).
