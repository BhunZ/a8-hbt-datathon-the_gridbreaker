# Datathon 2026 — THE GRIDBREAKER

> **Breaking Business Boundaries** — Pipeline phân tích dữ liệu và dự báo doanh thu cho thương hiệu thời trang Việt Nam.

---

## 👥 Team a8-hbt

| Name | Affiliation |
|---|---|
| **Tung Lam Nguyen** | VKU |
| **Quoc Hung Le** | VNU-HCM |
| **Bao Hung Nguyen Duc** | HCMOU |
| **Thanh Dat Hoang Ngoc** | UEH University |

---

## 📊 Bài toán

- **Đầu vào:** 13 file CSV (10.5 năm hoạt động, 2012–2022, tổng doanh thu lịch sử **$16.43B**).
- **Đầu ra:** Dự báo Revenue và COGS hàng ngày cho **548 ngày** (2023-01-01 → 2024-07-01).
- **Đánh giá:** MAE trên public leaderboard, MAE + RMSE + R² trên private leaderboard.
- **Ràng buộc:** Chỉ được dùng dữ liệu trong `data/raw/`. Không được dùng dữ liệu ngoài.

---

## 🏆 Kết quả cuối

| Submission | Mean rev/day | Kaggle MAE |
|---|---:|---:|
| V10c (baseline ensemble) | $4.23M | 774,898 |
| **V10c × 1.05 (final)** | **$4.44M** | **772,912** ★ |

**Submission cuối nộp:** [`submissions/submission_v10c_scaled_105.csv`](submissions/submission_v10c_scaled_105.csv)

Sau **17 phiên bản thử nghiệm**, phiên bản đơn giản nhất — V10c (RF + ET + HGB ensemble) với hệ số calibration 1.05 — đạt MAE thấp nhất. Mọi phiên bản phức tạp hơn (V11–V14 với multi-loss ensemble, deep architecture, post-cliff training cut) đều thua V10c trên public leaderboard.

---

## 📁 Cấu trúc repo

```
.
├── README.md                                
├── reports/                                 Báo cáo 4 trang nộp Datathon
│   ├── datathon_neurips.pdf              
│   └── datathon_neurips.docx             
│
├── notebooks/
│   └── datathon.ipynb            
│
├── src/                                     Source code (Python)
│   ├── paths.py                             Path config trung tâm
│   ├── v10c_fit.py                          ★ V10c baseline (Kaggle 774k)
│
└── data/                                    [GITIGNORED — competition data]
```

---

## 🚀 Cách reproduce kết quả

### Clone repo
```bash
git clone https://github.com/BhunZ/a8-hbt-datathon-the_gridbreaker
cd a8-hbt-datathon-the_gridbreaker
```

```bash
# Cài đặt Python dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib

# Đặt 13 file CSV và sample_submission của competition vào data/raw/ 
ls data/raw/
# sales.csv  orders.csv  order_items.csv  payments.csv  customers.csv
# products.csv  geography.csv  shipments.csv  reviews.csv  returns.csv
# web_traffic.csv  inventory.csv  promotions.csv
# sample_submission.csv

# Chạy V10c baseline
python src/v10c_fit.py
# Output: submissions/submission_v10c.csv (Kaggle MAE 774,898)
```

### Áp dụng calibration × 1.05 (final winner)

```python
import pandas as pd
import numpy as np

v10c = pd.read_csv("submissions/submission_v10c.csv", parse_dates=["Date"])
final = pd.DataFrame({
    "Date":    v10c.Date.dt.strftime("%Y-%m-%d"),
    "Revenue": np.round(v10c.Revenue * 1.05, 2),
    "COGS":    np.round(v10c.COGS    * 1.05, 2),
})
final.to_csv("submissions/submission_v10c_scaled_105.csv", index=False)
# Kaggle MAE: 772,912 (best)
```

---

## 🔍 Các phát hiện EDA chính

### 1. Vực thẳm 2019 — Doanh thu giảm 40%
- **Trước:** Orders ~5,800/ngày, Revenue ~$5M/ngày
- **Sau (mid-2019):** Orders ~3,500/ngày, Revenue ~$3M/ngày
- **Đặc điểm:** Conversion crash (traffic + signups vẫn tăng 6-15%, AOV stable)

### 2. Lỗ hổng dữ liệu vận chuyển cuối kỳ 2022
- Bảng `shipments` và `returns` chỉ phủ ~63% các đơn hàng trong tháng 11/2022
- Đây là data truncation, không phải tín hiệu kinh doanh
- **Cách xử lý:** Flag `is_sparse_period`, không xóa data

### 3. The Zero-Overlap Paradox (Returns ⊥ Reviews)
- 36,062 đơn có return, 111,369 đơn có review
- **Tập giao = 0** (không một đơn nào có cả hai)
- **3 phân đoạn khách hàng:**
  - **Silent Churners** (5.6%): trả hàng, không review
  - **Vocal Advocates** (15.9%): review, không trả hàng
  - **Passive Majority** (78.5%): không phản hồi

### 4. Pricing Dynamics: 2% Gaussian noise
- Trên 438k line items không discount, lệch giá so với catalog có std = 2.0%
- **Diễn giải:** Intra-day Pricing Dispersion — brand đang chạy dynamic pricing engine ngầm

### 5. Tiered Defect Analysis
- **Physical Defects** (chất lượng sản phẩm): cần QC supplier
- **Catalog Failures** (operational): tăng đột biến **17.6%** trong promo events
- **Customer-driven**: không actionable

---

## 🏷️ Project tags

`datathon-2026` `revenue-forecasting` `time-series` `vietnamese-ecommerce` `ensemble-learning` `post-hoc-calibration` `eda` `vinuniversity-vintelligence`

---

*Pipeline này là kết quả của 17 lần lặp với rất nhiều thất bại. "Sometimes the right call is to recognize when more iteration is hurting, not helping."*
