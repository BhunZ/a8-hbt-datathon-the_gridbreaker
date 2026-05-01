# PROJECT_MEMORY — Datathon 2026 (VinTelligence, VinUni)

> File này là "bộ nhớ dự án" để Claude session mới có thể đọc một lần và nắm được toàn bộ context, plan, và các quyết định đã/đang treo. Giữ ngắn gọn, cập nhật khi có thay đổi.

**Cập nhật lần cuối:** 2026-04-20

---

## 1. Bối cảnh cuộc thi

- **Cuộc thi:** Datathon 2026 — The Gridbreaker, tổ chức bởi VinTelligence (VinUni DS&AI Club).
- **Vòng:** Vòng 1 (hiện tại).
- **Chung kết:** 23/05/2026 tại VinUni, Hà Nội — cần ít nhất 1 thành viên tham gia offline.
- **Bối cảnh dữ liệu:** Mô phỏng doanh nghiệp thời trang thương mại điện tử VN.
- **Giai đoạn dữ liệu:** Train 04/07/2012 → 31/12/2022; Test 01/01/2023 → 01/07/2024.

### Thang điểm (tổng 100)
- **Phần 1 (20đ):** 10 câu MCQ, mỗi câu 2đ, không trừ điểm sai.
- **Phần 2 (60đ):** EDA & visualization — chấm theo 4 cấp (Descriptive → Diagnostic → Predictive → Prescriptive). Tiêu chí: Chất lượng viz (15) + Chiều sâu (25) + Insight kinh doanh (15) + Sáng tạo (5).
- **Phần 3 (20đ):** Forecast Revenue cho sales_test. Metrics: MAE, RMSE, R². Model (12đ) + Report (8đ). Kaggle leaderboard.

### Ràng buộc Phần 3 (vi phạm = loại bài)
1. Không dùng Revenue/COGS của test làm feature.
2. Không dùng dữ liệu ngoài.
3. Phải có mã nguồn tái lập được.

### Deliverables
- submission.csv (Kaggle, giữ đúng thứ tự sample_submission.csv)
- Báo cáo NeurIPS LaTeX, tối đa 4 trang (không tính ref + appendix)
- GitHub repo public có README

---

## 2. Dataset — 15 file, 4 lớp

| # | File | Lớp | PK / key | Ghi chú |
|---|---|---|---|---|
| 1 | products.csv | Master | product_id | Ràng buộc cogs < price |
| 2 | customers.csv | Master | customer_id | FK: zip → geography |
| 3 | promotions.csv | Master | promo_id | stackable_flag, applicable_category |
| 4 | geography.csv | Master | zip | region, district |
| 5 | orders.csv | Transaction | order_id | FK: customer_id, zip (SHIPPING) |
| 6 | order_items.csv | Transaction | (order_id, product_id) | FK: promo_id, promo_id_2 — "twist" |
| 7 | payments.csv | Transaction | order_id | 1:1 với orders |
| 8 | shipments.csv | Transaction | order_id | 1:0..1, chỉ có với shipped/delivered/returned |
| 9 | returns.csv | Transaction | return_id | FK: order_id + product_id |
| 10 | reviews.csv | Transaction | review_id | FK: order_id + product_id + customer_id |
| 11 | sales.csv | Analytical | Date | TARGET — Revenue, COGS daily aggregate |
| 12 | sample_submission.csv | Analytical | Date | Template |
| 13 | inventory.csv | Operational | (product_id, snapshot_date) | MONTHLY snapshot cuối tháng |
| 14 | web_traffic.csv | Operational | date (có thể × traffic_source) | DAILY, có thể nhiều row/ngày |

### Cardinality chính
- orders ↔ payments: **1 : 1**
- orders ↔ shipments: **1 : 0 hoặc 1** (status-dependent)
- orders ↔ returns: **1 : 0..N**
- orders ↔ reviews: **1 : 0..N** (~20% delivered)
- order_items ↔ promotions: **N : 0 hoặc 1** (qua promo_id); stacking qua promo_id_2
- products ↔ inventory: **1 : N** (1 dòng/sp/tháng)

### Sơ đồ ER
Xem file `schema_diagram.mermaid` (cùng folder).

---

## 3. Quyết định kiến trúc pipeline

### 3.1 Hai master dataset song song (không gộp làm một)

- **df_txn** — cấp line-item. Spine = order_items. Dùng cho EDA Phần 2.
- **df_daily** — cấp ngày toàn công ty. Spine = date range 2012-07-04 → 2024-07-01. Dùng cho Forecast Phần 3.

Lý do tách: sales.csv ở daily/company-wide, không có chiều product/customer. Gộp chung sẽ hoặc leak hoặc mất thông tin.

### 3.2 Thứ tự join cho df_txn

```
order_items (spine, tạo line_item_id = reset_index)
  ← orders ON order_id
      ← customers ON customer_id
      ← geography AS geo_ship ON orders.zip       -- SHIPPING zip
      ← geography AS geo_home ON customers.zip    -- REGISTRATION zip
  ← products ON product_id
  ← promotions AS p1 ON promo_id                  -- self-join
  ← promotions AS p2 ON promo_id_2                -- self-join
  ← payments ON order_id                          -- 1:1, an toàn
  ← shipments ON order_id                         -- 1:0..1, an toàn
  ← returns_agg ON (order_id, product_id)         -- đã aggregate trước
  ← reviews_agg ON (order_id, product_id)         -- đã aggregate trước
```

**Bắt buộc aggregate returns và reviews trước khi join** về cặp key (order_id, product_id). Nếu không, fan-out sẽ nhân dòng.

### 3.3 Xử lý "xoắn đôi" promotions

Self-join 2 lần với alias p1 và p2. Giữ riêng biệt, không union. Tạo feature tổng hợp:
- `total_discount = coalesce(p1.discount, 0) + coalesce(p2.discount, 0)`
- `is_stacked = promo_id_2 IS NOT NULL`

### 3.4 Granularity alignment (inventory × web_traffic × sales)

- **web_traffic:** nếu có nhiều row/ngày theo traffic_source → group by date trước (sum sessions, weighted mean bounce_rate).
- **inventory (monthly → daily):** CHỌN **Chiến lược B** — aggregate inventory về toàn công ty theo tháng, `shift(1)` (dùng cuối tháng T−1 làm feature cho tháng T), rồi broadcast xuống daily. An toàn leakage.
- **sales daily:** kiểm tra gap ngày trước khi build lag features.

### 3.5 Zip disambiguation

- `geo_ship` join qua `orders.zip` → dùng cho phân tích doanh thu theo vùng (Q7).
- `geo_home` join qua `customers.zip` → dùng cho phân tích demographic.
- Flag `is_shipping_to_home = (orders.zip == customers.zip)`.

---

## 4. Cấu trúc Notebook (10 sections)

| # | Section | Nội dung chính |
|---|---|---|
| 0 | Setup | Imports, helpers (load_csv, audit_fk, assert_no_fanout) |
| 1 | Load & Schema Inspection | Load 14 CSV vào dict `raw`, parse dates, shape/dtypes |
| 2 | Per-table Audit | PK uniqueness, range check, categorical cardinality |
| 3 | Referential Integrity Audit | audit_fk() cho 10 cặp FK |
| 4 | Business Logic Audit | cogs<price, date order, rating range, promo consistency |
| 5 | Pre-join Aggregation | returns_agg, reviews_agg, promo_effective |
| 6 | Build df_txn | Join theo 3.2, assert_no_fanout sau mỗi bước |
| 7 | Financial Reconciliation | Test 5 công thức Revenue vs sales.csv, chốt công thức đúng |
| 8 | Build df_daily | date_spine → aggregate → align inventory shifted → merge |
| 9 | Feature Engineering | date, lag, rolling, trend features (tất cả shift để tránh leak) |
| 10 | Leakage Guard & Export | check_leakage(), time-split train/valid, export parquet |

---

## 5. Risk checklist (các điểm đã xác định)

### 5.1 Khóa & toàn vẹn
- Duplicate PK trong master tables
- Orphan FK: product_id, customer_id, zip, promo_id
- Duplicate order_id trong payments (vi phạm 1:1)

### 5.2 Logic kinh doanh
- cogs ≥ price (vi phạm ràng buộc đề)
- delivery_date < ship_date < order_date (đảo ngược)
- review_date/return_date < order_date
- signup_date > first order_date
- rating ∉ [1,5], quantity/unit_price ≤ 0
- Promo áp ngoài [start_date, end_date], sai category, promo_id_2 khi stackable_flag=0
- discount_amount không khớp công thức

### 5.3 Status consistency
- cancelled orders có shipments (không nên)
- cancelled orders có payments (cần verify)
- non-delivered có reviews (không nên)
- non-returned có returns (không nên)

### 5.4 Granularity
- web_traffic có nhiều row/ngày
- inventory monthly vs daily target (leakage risk)
- gap ngày trong sales.csv phá vỡ lag features

### 5.5 Join
- Cột trùng tên: city, payment_method, product_name/category/segment
- Fan-out khi join 1:N chưa aggregate
- Double join promotions phải dùng alias

### 5.6 Leakage (Phần 3)
- Revenue/COGS test làm feature → loại bài
- Lag vượt forecast horizon
- Inventory snapshot cùng tháng với target → leak
- Target encoding phải out-of-fold
- CV phải TimeSeriesSplit, không K-fold ngẫu nhiên

---

## 6. Mapping risk → MCQ

| Câu | Điểm cần chú ý |
|---|---|
| Q1 | Filter customer >1 đơn; median gap nội bộ rồi median tổng |
| Q2 | Margin per row → mean theo segment (không phải margin của tổng) |
| Q3 | Join returns×products, filter category='Streetwear' (check case) |
| Q4 | web_traffic có thể long format; simple mean bounce_rate |
| Q5 | Chỉ đếm promo_id (không tính promo_id_2) |
| Q6 | Filter age_group notnull; orders/unique customers per group |
| Q7 | sales không có zip → build từ orders×order_items×geography. **Dùng orders.zip, không phải customers.zip** |
| Q8 | Dùng orders.payment_method (cancelled có thể không có row payments) |
| Q9 | Tử: returns×products; mẫu: order_items×products; filter size ∈ {S,M,L,XL} |
| Q10 | Group payments theo installments, mean payment_value |

---

## 7. Các công thức audit quan trọng

### 7.1 Financial reconciliation — test 5 công thức Revenue
| Ký hiệu | Công thức |
|---|---|
| F1 | Σ(qty × unit_price) group by orders.order_date |
| F2 | F1 − Σ refund_amount group by returns.return_date |
| F3 | F1 − refund theo order_date của returned items |
| F4 | F1 chỉ với status ∈ {delivered, shipped} |
| F5 | F4 − refund |

Chọn công thức có MAE thấp nhất so với sales.csv → định nghĩa "Revenue" cho toàn pipeline.

### 7.2 Fan-out guard
```python
def assert_no_fanout(df_before, df_after, spine_key, step_name):
    n_before = df_before[spine_key].nunique()
    n_after_unique = df_after[spine_key].nunique()
    assert n_after_unique == n_before
    if len(df_after) != n_before:
        print(f'WARN {step_name}: fan-out {n_before}→{len(df_after)}')
```

### 7.3 Leakage guard
```python
def check_leakage(features_df, target_date_col, max_lookahead_days=0):
    for col in features_df.columns:
        if col.endswith('_date'):
            bad = features_df[col] > features_df[target_date_col] - pd.Timedelta(days=max_lookahead_days)
            if bad.any():
                raise ValueError(f'Leakage in {col}: {bad.sum()} rows')
```

---

## 8. Quyết định đang treo (cần user chốt)

1. **Orphan key policy:** drop hay giữ với NaN? *Khuyến nghị Claude: giữ với NaN.*
2. **Cancelled orders:** đưa vào df_txn không? *Khuyến nghị: giữ với flag is_cancelled.*
3. **Inventory alignment:** Chiến lược A (merge_asof) hay B (monthly+shift)? *Khuyến nghị: B.*
4. **df_daily date range:** chỉ train (→2022-12-31) hay extend đến test end (→2024-07-01)? *Khuyến nghị: extend, mark is_test.*

---

## 9. Trạng thái hiện tại

- [x] Đọc đề, hiểu bối cảnh
- [x] Vẽ sơ đồ ER (schema_diagram.mermaid)
- [x] Thiết kế schema join logic
- [x] Lập risk checklist
- [x] Cấu trúc notebook
- [ ] User chốt 4 quyết định treo (mục 8)
- [ ] Viết code Section 0 (Setup)
- [ ] Viết code Section 1-2 (Load & Audit)
- [ ] Viết code Section 3-4 (FK & business logic audit)
- [ ] Viết code Section 5 (Pre-join aggregation)
- [ ] Viết code Section 6 (df_txn)
- [ ] Viết code Section 7 (Financial reconciliation) ← **quan trọng nhất cho Phần 3**
- [ ] Viết code Section 8 (df_daily)
- [ ] Viết code Section 9 (Feature engineering)
- [ ] Viết code Section 10 (Leakage guard & export)
- [ ] Phần 1 MCQ — trả lời từ df_txn
- [ ] Phần 2 EDA — visualization & storytelling
- [ ] Phần 3 Forecasting — model training, Kaggle submission
- [ ] Báo cáo NeurIPS 4 trang
- [ ] GitHub repo + README

---

## 10. Lưu ý vận hành

- File này cần được cập nhật sau mỗi mốc lớn.
- Khi Claude session mới: đọc file này TRƯỚC, rồi đọc `schema_diagram.mermaid`, rồi mới bắt đầu công việc.
- Nếu công thức Revenue (mục 7.1) đã được chốt: ghi vào mục 9 dưới dạng "REVENUE_FORMULA = F?" để session sau khỏi test lại.
- Notebook file hiện có: `baseline.ipynb` (chưa đọc nội dung).
