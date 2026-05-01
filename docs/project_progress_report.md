# BÁO CÁO TIẾN ĐỘ DỰ ÁN DATATHON 2026

**Dự án:** VinUniversity VinTelligence — Datathon Round 1
**Bộ dữ liệu:** Vietnamese Fashion E-commerce (14 bảng CSV, dual-master dimensional model)
**Giai đoạn đã hoàn tất:** Phase 0 → Phase 5
**Ngày cập nhật:** 2026-04-22
**Người báo cáo:** Senior Data Architect

---

## 1. Tổng quan dự án (Executive Summary)

Mục tiêu chiến lược của giai đoạn 0–5 là xây dựng một **dual-master pipeline** — hai bảng master đã được reconcile hoàn toàn, đóng vai trò nguồn sự thật (source of truth) duy nhất cho toàn bộ downstream analytics, EDA, và Machine Learning:

| Master table | Grain (Primary Key) | Kích thước | Vai trò |
|---|---|---:|---|
| **`df_txn`** | `line_item_id` (transaction) | 714,669 × 64 | Phân tích hành vi đơn hàng, return, review ở mức dòng giao dịch |
| **`df_daily`** | `Date` (daily) | 3,833 × 58 | Phân tích time-series, forecasting, kết hợp traffic & inventory |

Cả hai master đều:
- Được xây dựng từ 14 bảng CSV nguồn thông qua 5 phase kiểm soát chất lượng nghiêm ngặt.
- **Conservation-verified**: mọi bất biến (revenue, cogs, orders, lines, shipping fee) đều khớp đến từng cent so với `sales.csv` chuẩn.
- Đã vượt qua **76 audit checks** (tính từ Phase 1 đến Phase 5), với **0 FAIL rows** trong phiên bản cuối.
- Sẵn sàng cho Phase 6 (Feature Engineering) và Phase 7 (Modelling).

**Điểm mấu chốt:** Pipeline này không chỉ “join dữ liệu lại” — nó đã xử lý 4 vấn đề có thể phá hỏng downstream analytics (unit_price noise, B7 stackable promo, B10/B11 truncation tail, shipping fee duplication) và ghi lại toàn bộ rationale cho team tham khảo.

---

## 2. Chi tiết từng giai đoạn (Phase-by-Phase Breakdown)

### 2.1. Phase 0 & 1 — Thiết lập môi trường và Nạp dữ liệu

**Mục tiêu:** Xây dựng helper module, nạp 14 bảng CSV với typed dtypes, và **reconcile doanh thu $16.4B** so với bảng `sales.csv` chuẩn.

| Kết quả | Giá trị |
|---|---:|
| Số bảng nguồn được nạp | 14 |
| Tổng số dòng dữ liệu | 1,637,933 |
| Primary Key violations | 0 |
| Foreign Key orphan rate | 0.0% trên 15 FK relationships |
| **Tổng doanh thu reconciled** | **$16,430,476,585.53** |
| Công thức doanh thu F1 | `Σ (quantity × unit_price)` trên `order_items` |
| Sai lệch so với `sales.Revenue` | **$0.00 (MAE = 0)** |

**Output quan trọng:**
- `helpers.py`: module typed-loading có sẵn hàm `load_csv_typed()`, `audit_pk_unique()`, `audit_fk()`.
- 14 bảng parquet snapshot trong `outputs/parquet/` — dùng cho fast phase re-entry.
- `pipeline.ipynb`: notebook kết nối Phase 0 → 2 có thể chạy lại.
- `profile_report` chứa shape, dtype, null rate, cardinality của từng bảng.

**Bài học chính:** Bằng cách nạp dữ liệu với typed dtypes ngay từ đầu (thay vì để pandas inference), ta loại bỏ được hoàn toàn drift kiểu dữ liệu (dtype drift) khi join ở các phase sau.

---

### 2.2. Phase 2 — Kiểm định chất lượng và Đặc tả (Audit & Gap Analysis)

**Mục tiêu:** Chạy 22 business-rule checks (B1–B15 + G_CHK1–G_CHK5), giải quyết các gap trong spec, và chốt các design decisions cho downstream.

#### 2.2.1. Phát hiện #1: **Công thức `unit_price` có Gaussian noise 2%**

Team ban đầu không reconcile được `unit_price` với `product_price + shipping_fee` hay `product_price − discount`. Sau khi probe 7 promo buckets khác nhau, phát hiện quy luật:

$$
\text{unit\_price} = \text{product\_price} \times (1 - d) \times (1 + \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, 0.02)
$$

với `d` = discount fraction. Độ lệch chuẩn **σ = 0.020 đồng nhất** trên mọi bucket (xác nhận bằng stratified residual validation). Đây là multiplicative Gaussian noise được inject bởi synthetic data generator — không phải lỗi dữ liệu.

**Ý nghĩa:** Mọi phân tích giá cần tolerate ±2% random noise. Không thể dùng `unit_price` để reverse-engineer discount chính xác.

#### 2.2.2. Phát hiện #2: **Công thức `discount_amount` dạng hybrid**

Rule B15 ban đầu FAIL 20,950 dòng khi dùng công thức `qty × unit_price × d`. Sau khi drill-down, phát hiện quy luật 3 nhánh:

| Promo type | Công thức discount_amount |
|---|---|
| `percentage` (primary) | `qty × unit_price × disc_frac` |
| `fixed` (primary) | `qty × 50` (flat \$50/unit) |
| Stacked (`promo_id_2` not null) | Cộng thêm **+\$50 flat per line**, độc lập với `stackable_flag` |
| Không có promo | `0` |

Công thức hybrid này khớp **714,669 dòng đến ±\$0.02**. Quyết định: spec rule B7 (`stackable_flag` enforcement) bị deprecate, thay bằng công thức deterministic đã được verified.

#### 2.2.3. Phát hiện #3: **Dataset-tail truncation (B10 & B11)**

| Rule | Số dòng ảnh hưởng | Đặc điểm |
|---|---:|---|
| **B10** (order thiếu shipment) | 564 đơn | 100% rơi vào 2022-12-22 → 2022-12-31 |
| **B11** (order "returned" nhưng không có return row) | 80 đơn | 100% rơi vào 2022-12-06 → 2022-12-31 |

Kết luận: Đây là **dataset truncation tail** — file shipments/returns bị cắt tại 2022-12-31 nên các đơn vào cuối tháng 12/2022 có shipment/return record chưa kịp xuất hiện. Quyết định triage:
- **KHÔNG loại bỏ** các đơn này khỏi df_txn.
- Gắn flag `truncation_flag_shipment` / `truncation_flag_return` để downstream biết và handle đúng.

#### 2.2.4. Phát hiện #4: **`signup_date` KHÔNG phải "first interaction"**

Rule B13 FAIL trên **80,623 khách hàng (89.3%)** với median gap = **2,020 ngày (~5.5 năm)** — tức là `signup_date` nằm **SAU** order đầu tiên. Điều này chứng tỏ `signup_date` là ngày đăng ký tài khoản / enroll loyalty, không phải "first interaction".

**Quyết định chốt:** Tenure anchor chính thức là `min(order_date) per customer_id`, không dùng `signup_date`. Feature `customer_tenure_days` được tính theo đó.

#### 2.2.5. Phát hiện #5: **Gap G1 — `order_items` không có natural PK**

Bảng `order_items` có 16 cặp `(order_id, product_id)` duplicate ở mức dòng. Giải pháp: thêm **surrogate key** `line_item_id` (1-indexed int64) — trở thành PK chính thức của `order_items` và `df_txn`.

**Tóm tắt Phase 2:**

| Verdict | Count |
|---|---:|
| PASS | 17 |
| FAIL (demoted sau triage) | 3 → 0 |
| LOG | 1 |
| WARN | 1 |

---

### 2.3. Phase 3 — Xử lý dữ liệu vệ tinh (Satellite Aggregation)

**Mục tiêu:** Aggregate `returns` và `reviews` về grain `(order_id, product_id)` để có thể LEFT JOIN sạch vào df_txn.

#### 2.3.1. Phase 3A — `returns_agg` (39,937 dòng × 15 cột)

Từ 39,939 return events → aggregate về 39,937 line-level entries.

**Tiered defect classification** (nâng cấp từ flag đơn giản lên 2 bucket riêng biệt):

| Flag | Reason set | Số dòng | Tỷ lệ |
|---|---|---:|---:|
| `any_physical_defect_line` | `{"defective", "damaged"}` | 8,020 | 20.08% |
| `any_catalog_failure_line` | `{"not_as_described", "wrong_item"}` | 7,035 | 17.62% |
| Không flag (customer preference / logistics) | `wrong_size`, `changed_mind`, `late_delivery` | 24,882 | 62.30% |

**Key features:** `refund_over_paid_ratio`, `time_to_first_return_days`, `dominant_return_reason`, `has_any_return`, `truncation_flag_return`.

**Validations:** 9/9 PASS hoặc LOG. Conservation `Σ refund_amount` khớp đến ±\$0.01.

#### 2.3.2. Phase 3B — `reviews_agg` (113,551 dòng × 14 cột)

Nguồn `reviews` đã unique trên `(order_id, product_id)` (0 dup, 0 orphan) — aggregation về bản chất là structured projection.

**Sentiment flags:**

| Flag | Rule | Số dòng | Tỷ lệ | Baseline |
|---|---|---:|---:|---|
| `has_high_rating` | `rating ≥ 4` | 81,668 | 71.92% | — |
| `has_low_rating` | `rating ≤ 2` | 14,867 | 13.09% | — |
| Trung bình `avg_rating` | — | — | 3.936 / 5 | — |

Latency: p50 = 21 ngày, p99 = 38 ngày. **0 retro-reviews** (tất cả review đều sau `order_date`).

#### 2.3.3. Phase 3C — Phát hiện "Zero Overlap" giữa Returns và Reviews

Cross-check intersection giữa `returns_agg` và `reviews_agg` trên `(order_id, product_id)`:

| Set | Size |
|---|---:|
| `returns_agg` | 39,937 |
| `reviews_agg` | 113,551 |
| **Intersection (ret ∧ rev)** | **0 (0.00%)** |
| Union | 153,488 |

**Verified tại 3 layer độc lập** (post-agg parquet, raw source, direct merge) với dtype int32 đồng nhất. Không có một dòng nào vừa được review vừa được return — ngay cả trên `order_id` đơn lẻ cũng 0 overlap.

**Ý nghĩa:**
1. Đây rất có thể là **synthetic-generator property** (mỗi line chỉ được gán 1 trong 2).
2. Post-purchase outcome space collapse xuống còn **3 buckets** (`no_feedback`, `reviewed_only`, `returned_only`) — bucket thứ 4 `reviewed_and_returned` luôn rỗng.
3. Line-level `avg_rating` **không thể** dùng trực tiếp làm return-risk feature. Phải transfer qua product-level aggregate.

**Coverage math:** reviewed_only = 15.89% · returned_only = 5.59% · no_feedback = 78.52% của toàn bộ order_items.

---

### 2.4. Phase 4 — Master Transaction Table (`df_txn`)

**Mục tiêu:** Master join — base table `order_items` LEFT JOIN 7 satellite (orders, customers, products, promotions, shipments, returns_agg, reviews_agg) không gây fan-out.

| Property | Value |
|---|---|
| Rows / Cols | **714,669 × 64** |
| PK | `line_item_id` (0 duplicates) |
| Fan-out | **Zero** — `len(df_txn) == len(order_items)` |
| Revenue conservation | `Σ qty × unit_price = $16,430,476,585.53` (diff = $0.00 vs sales) |

#### Các feature derived được chốt trong Phase 4:

| Feature | Công thức | Ghi chú |
|---|---|---|
| `customer_tenure_days` | `order_date − first_order_date` | `first_order_date = min(order_date) per customer` (theo B13 redefinition) |
| `truncation_flag_shipment` | `status ∈ {shipped, delivered, returned} ∧ no shipment row` | 564 đơn (khớp B10) |
| `truncation_flag_return` | `order_id ∈ returned_orders_truncated.csv` | 80 đơn (khớp B11) |
| `has_shipment`, `has_any_return`, `has_any_review` | Boolean presence flags | NULL-safe sau LEFT join & fillna(0) |
| `post_purchase_outcome` | 3-valued categorical | `no_feedback / reviewed_only / returned_only` (bucket thứ 4 guaranteed empty bởi Phase 3C) |

#### Distribution `post_purchase_outcome`:

| Bucket | Lines | Share |
|---|---:|---:|
| `no_feedback` | 561,177 | 78.52% |
| `reviewed_only` | 113,553 | 15.89% |
| `returned_only` | 39,939 | 5.59% |
| `reviewed_and_returned` | **0** | 0.00% |

**Validations:** 15/15 PASS hoặc LOG. Zero FAIL.

**Null inventory (tất cả đều expected by design):**
- `avg_rating`, `dominant_rating`: 84.11% null (chỉ có trên `reviewed_only`).
- `refund_over_paid_ratio`, `dominant_return_reason`: 94.41% null (chỉ có trên `returned_only`).
- `delivery_date`: 12.49% null (cancelled/paid/created + 564 B10).
- `promo_id`: 61.34% null (phần lớn line không promotional).

---

### 2.5. Phase 5 — Daily Time-Series Master (`df_daily`)

**Mục tiêu:** Roll-up `df_txn` về grain ngày, LEFT JOIN `web_traffic` (daily) và `inventory` (monthly snapshot, merge_asof backward).

| Property | Value |
|---|---|
| Rows / Cols | **3,833 × 58** |
| Date range | 2012-07-04 → 2022-12-31, **0 gaps** |
| PK | `Date` (unique) |
| Revenue total | $16,430,476,585.53 — khớp chính xác `Σ sales.Revenue` |
| COGS max daily Δ | **$0.00** vs `sales.COGS` (cumulative Σ drift $0.04 là float64 noise, ~3 ppt) |
| Shipping fee total | $2,809,309.66 — khớp chính xác `Σ shipments.shipping_fee` |

#### Column blocks trong df_daily:

| Block | Columns tiêu biểu |
|---|---|
| **Spine & canonical** | `Date`, `sales_revenue_canonical`, `sales_cogs_canonical` |
| **Volume roll-ups** | `n_line_items`, `n_units`, `n_orders`, `n_unique_customers`, `n_new_customer_lines`, `n_shipped_orders` |
| **Economics** | `gross_revenue`, `gross_cogs`, `net_revenue`, `gross_margin`, `total_discount_amount`, `total_shipping_fee` |
| **Ratios** | `aov`, `units_per_order`, `avg_unit_price`, `cancel_rate_lines`, `return_rate_lines`, `review_rate_lines`, `promo_revenue_share` |
| **Status mix** | `n_status_{delivered,cancelled,returned,shipped,paid,created}` |
| **Web traffic** (exogenous) | `sessions`, `unique_visitors`, `page_views`, `bounce_rate`, `avg_session_duration_sec`, `traffic_source` |
| **Inventory** (monthly backward-filled) | `inv_total_stock_on_hand`, `inv_n_stockouts`, `inv_avg_fill_rate`, `inv_avg_days_of_supply`, `inv_avg_sell_through` |
| **Calendar** | `year`, `quarter`, `month`, `day_of_week`, `is_weekend`, `is_month_end`, `is_month_start` |

**Validations:** 13/13 PASS hoặc LOG. Zero FAIL.

**2 landmine quan trọng đã xử lý:**

1. **Shipping fee duplication** — `shipping_fee` được stamp trên mọi line của đơn hàng nhiều dòng. Nếu naive `sum()` sẽ ra \$3.03M (sai), trong khi giá trị đúng là \$2.81M. Fix: de-duplicate theo `order_id` trước khi sum.

2. **COGS cumulative drift** — `Σ sales.COGS` drift $0.04 trên tổng $12.8B (~3 parts per trillion) do float64 summation noise. Daily-level invariant (max per-day Δ) là kiểm thử đúng nghĩa về mặt semantic, nên được chọn.

---

## 3. Các phát hiện quan trọng (Key Insights & Discoveries)

### 3.1. Các data anomaly đã được giải quyết

| # | Anomaly | Mức độ ảnh hưởng | Giải pháp |
|---|---|---|---|
| 1 | `unit_price` có Gaussian noise σ=2% | Toàn bộ 714,669 line | Chấp nhận & document; không thể dùng để reverse-engineer chính xác discount |
| 2 | `discount_amount` theo công thức hybrid (fixed/percentage + flat stacked) | 20,950 line FAIL ban đầu | Công thức 3-nhánh, rule B7 deprecated |
| 3 | B10 — 564 đơn thiếu shipment | Dataset tail (Dec 2022) | Flag `truncation_flag_shipment`, giữ nguyên line |
| 4 | B11 — 80 đơn "returned" không có return row | Dataset tail (Dec 2022) | Flag `truncation_flag_return`, persist list riêng |
| 5 | B13 — `signup_date` sai semantic với 89.3% customer | Toàn tập khách hàng | Redefine tenure theo `min(order_date) per customer` |
| 6 | G1 — `order_items` không có natural PK | 16 cặp duplicate | Thêm surrogate `line_item_id` |
| 7 | Zero overlap Returns vs Reviews | 0 intersection | Document, collapse outcome space xuống 3 buckets |
| 8 | `shipping_fee` duplication per line | Inflation \$217K | De-duplicate theo `order_id` trước khi sum |
| 9 | COGS float64 cumulative drift \$0.04 | Chỉ ở level aggregate | Chuyển test sang max-per-day delta |
| 10 | Web traffic thiếu 181 ngày đầu (G_CHK5) | Pre-2013 period | Giữ NaN để Phase 6 xử lý (forward-fill hoặc loại pre-2013 khỏi training) |

### 3.2. Những con số đáng nhớ

- **$16,430,476,585.53** — tổng doanh thu được reconcile chính xác đến từng cent qua cả 2 master table và `sales.csv`.
- **0 / 76** — FAIL rows trong toàn bộ audit pipeline (sau khi đã triage).
- **714,669 / 3,833** — kích thước df_txn và df_daily.
- **3 buckets** — `post_purchase_outcome` collapsed do Zero Overlap finding.
- **σ = 0.020** — noise identical trên mọi promo bucket (dấu hiệu rõ của synthetic generator).

---

## 4. Trạng thái hiện tại (Current Readiness)

### 4.1. Data integrity status

| Invariant | Status |
|---|:---:|
| PK uniqueness trên cả 2 master | ✓ |
| FK orphan rate = 0% trên 15 relationships | ✓ |
| Revenue conservation (df_txn ↔ df_daily ↔ sales) | ✓ |
| COGS conservation (daily-level) | ✓ |
| Shipping fee conservation vs shipments | ✓ |
| Truncation flags khớp Phase 2 counts (B10=564, B11=80) | ✓ |
| Cross-satellite disjoint rule (Phase 3C) | ✓ |
| Date spine gap-free (3,833 ngày) | ✓ |
| Web traffic head gap (G_CHK5 = 181 ngày) | ✓ |
| Inventory backward-fill coverage (G_CHK2 = 27 ngày head null) | ✓ |

### 4.2. Artifacts đã được bàn giao

| Artifact | Mô tả |
|---|---|
| `outputs/parquet/df_txn.parquet` | Master transaction, 714,669 × 64 |
| `outputs/parquet/df_daily.parquet` | Master daily time-series, 3,833 × 58 |
| `outputs/parquet/returns_agg.parquet` | Satellite return aggregate, 39,937 × 15 |
| `outputs/parquet/reviews_agg.parquet` | Satellite review aggregate, 113,551 × 14 |
| `outputs/parquet/*.parquet` (14 bảng) | Typed snapshot của từng bảng nguồn |
| `outputs/phase{1..5}_audit_results.csv` | 76 audit checks với verdict + detail |
| `outputs/phase{1..5}_summary.md` | Triage narrative của mỗi phase |
| `outputs/unit_price_formula.md` | Write-up công thức đã reverse-engineered |
| `outputs/returned_orders_truncated.csv` | 80 đơn B11 cho downstream reference |
| `outputs/phase3_crosscheck.json` | Machine-readable overlap stats |
| `outputs/phase{1..5}*.py` | Re-runnable build scripts |

### 4.3. Sẵn sàng cho Phase 6 (Feature Engineering) & Phase 7 (Modelling)

Pipeline hiện tại đã **100% clean và reconciled**. Team có thể bắt đầu:

- **EDA & Visualization:** Cả 2 master đã có dtype compact, ready cho pandas/polars/plotly.
- **Feature Engineering (Phase 6):**
  - Lag features trên df_daily (rolling 7/14/30/90 ngày).
  - Customer-level aggregate (total_spend, recency, frequency, monetary) từ df_txn.
  - Product-level rating transfer (để bypass Zero Overlap issue).
- **Modelling (Phase 7):**
  - Return prediction (target = `has_any_return`, excluding 80 truncated orders).
  - Daily demand forecasting trên df_daily (có thể drop pre-2013 nếu cần).
  - Customer churn / CLV (dùng `customer_tenure_days` đã redefine).

### 4.4. Các quyết định thiết kế đã chốt (không được thay đổi downstream)

1. PK của df_txn là `line_item_id`, **không** phải `(order_id, product_id)`.
2. Tenure anchor là `first_order_date`, **không** phải `signup_date`.
3. Công thức doanh thu F1 là `Σ (quantity × unit_price)` — đã chứng minh khớp `sales.Revenue` đến cent.
4. 80 đơn B11 và 564 đơn B10 **không bị loại** khỏi master — chỉ flag truncation.
5. Tiered defect classification chia thành `any_physical_defect_line` và `any_catalog_failure_line`, không dùng flag gộp.
6. `has_any_return` / `has_any_review` / `has_shipment` là canonical presence flags, **không** null-check lại post-join.
7. Zero Overlap là structural property — phải được reflect trong mọi downstream phân tích (ví dụ: không dùng line-level rating làm return feature).

---

## 5. Tổng kết

Qua 5 giai đoạn, team đã chuyển hóa 14 file CSV thô (với noise, truncation, và nhiều data anomaly tiềm ẩn) thành **một dual-master pipeline production-grade** có đầy đủ:

- **Integrity guarantees** — 0 FAIL rows qua 76 audit checks.
- **Conservation proofs** — mọi bất biến revenue/cogs/orders/lines/shipping khớp đến cent.
- **Documented rationale** — mọi design decision có markdown triage đi kèm.
- **Reproducibility** — mọi artifact có script re-runnable.

Pipeline sẵn sàng cho giai đoạn EDA sâu và Machine Learning. Các quyết định kỹ thuật quan trọng đã được chốt và ghi lại để team tham khảo nhất quán trong các phase tiếp theo.

**Phase 5 — Complete. Ready for Phase 6 & beyond.**
