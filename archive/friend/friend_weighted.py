import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

def get_bf_date(year):
    nov_1st = pd.Timestamp(year, 11, 1)
    thanksgiving = nov_1st + pd.Timedelta(days=(3 - nov_1st.weekday() + 7) % 7 + 21)
    return thanksgiving + pd.Timedelta(days=1)

def train_weighted_god_mode():
    print("--- ⚖️ KÍCH HOẠT WEIGHTED GOD MODE: SURGICAL OPTIMIZATION ⚖️ ---")
    
    # 1. Load Data
    sales = pd.read_csv('sales.csv', parse_dates=['Date'])
    traffic = pd.read_csv('web_traffic.csv', parse_dates=['date'])
    test = pd.read_csv('sample_submission.csv', parse_dates=['Date'])
    
    # 2. Xử lý Traffic Profile
    daily_traffic = traffic.groupby('date')[['sessions', 'page_views']].sum().reset_index()
    daily_traffic.rename(columns={'date': 'Date'}, inplace=True)
    daily_traffic['month'] = daily_traffic['Date'].dt.month
    daily_traffic['dow'] = daily_traffic['Date'].dt.dayofweek
    # Giữ nguyên logic God Mode: Lấy trung bình toàn dải để ổn định
    traffic_avg = daily_traffic.groupby(['month', 'dow'])[['sessions', 'page_views']].mean().reset_index()

    # 3. Lịch các ngày bùng nổ (Vietnam Context)
    tet_dates = pd.to_datetime(['2017-01-28', '2018-02-16', '2019-02-05', '2020-01-25', '2021-02-12', '2022-02-01', '2023-01-22', '2024-02-10'])

    def get_features(df_input, is_train=True):
        df = df_input.copy()
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dow'] = df['Date'].dt.dayofweek
        
        # Merge Traffic (Consistent with God Mode)
        if is_train:
            df = pd.merge(df, daily_traffic[['Date', 'sessions', 'page_views']], on='Date', how='left').fillna(0)
        else:
            df = pd.merge(df, traffic_avg, on=['month', 'dow'], how='left').fillna(0)
        
        df['is_payday'] = ((df['day'] >= 25) | (df['day'] <= 5)).astype(int)
        df['is_double_day'] = (df['month'] == df['day']).astype(int)
        
        df['is_tet_season'] = 0
        for tet_date in tet_dates:
            mask = (df['Date'] >= (tet_date - pd.Timedelta(days=21))) & (df['Date'] < tet_date)
            df.loc[mask, 'is_tet_season'] = 1
            
        df['is_holiday_period'] = 0
        holidays = [(1, 1), (4, 30), (5, 1), (9, 2), (12, 24), (12, 25), (12, 31)]
        for m, d in holidays:
            for offset in range(0, 4):
                df.loc[(df['month'] == m) & (df['day'] == (d - offset)), 'is_holiday_period'] = 1
        
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['dow_sin'] = np.sin(2 * np.pi * df['dow']/7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow']/7)
        
        return df

    # 4. Training
    train_data = sales[sales['Date'].dt.year >= 2017].reset_index(drop=True)
    train_df = get_features(train_data, is_train=True)
    test_df = get_features(test, is_train=False)
    
    features = ['month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'day', 'is_payday', 
                'is_double_day', 'is_tet_season', 'sessions', 'page_views', 'is_holiday_period']
    
    y_rev = np.log1p(train_df['Revenue'])
    y_cogs = np.log1p(train_df['COGS'])
    
    # --- PHƯƠNG PHÁP 1: SAMPLE WEIGHTING ---
    # Tăng trọng số cho dữ liệu gần đây (2021-2022) để mô hình ưu tiên học hiệu suất mới
    # Càng xa năm 2016, trọng số càng tăng theo hàm mũ nhẹ
    weights = (train_df['year'] - 2016) ** 1.2
    
    print("Huấn luyện Weighted Ensemble (Trọng số ưu tiên hiện đại)...")
    rf = RandomForestRegressor(n_estimators=600, max_depth=15, random_state=42, n_jobs=-1)
    et = ExtraTreesRegressor(n_estimators=800, max_depth=16, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=42)
    
    # Fit với sample_weight
    rf.fit(train_df[features], y_rev, sample_weight=weights)
    et.fit(train_df[features], y_rev, sample_weight=weights)
    hgb.fit(train_df[features], y_rev, sample_weight=weights)
    
    # Fit cho COGS (Dùng trọng số tương tự)
    rf_c = RandomForestRegressor(n_estimators=600, max_depth=15, random_state=42, n_jobs=-1)
    et_c = ExtraTreesRegressor(n_estimators=800, max_depth=16, random_state=42, n_jobs=-1)
    hgb_c = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=42)
    rf_c.fit(train_df[features], y_cogs, sample_weight=weights)
    et_c.fit(train_df[features], y_cogs, sample_weight=weights)
    hgb_c.fit(train_df[features], y_cogs, sample_weight=weights)

    # 5. Predict & Blending (Cộng hưởng 30% RF, 50% ET, 20% HGB)
    print("Tổng hợp dự báo (Tối ưu hóa ET)...")
    def blend_predict(models, X, w):
        return sum(m.predict(X) * weight for m, weight in zip(models, w))

    p_rev = blend_predict([rf, et, hgb], test_df[features], [0.3, 0.5, 0.2])
    p_cogs = blend_predict([rf_c, et_c, hgb_c], test_df[features], [0.3, 0.5, 0.2])
    
    test['Revenue'] = np.expm1(p_rev).round(2)
    test['COGS'] = np.expm1(p_cogs).round(2)

    # 6. Export
    test[['Date', 'Revenue', 'COGS']].to_csv('submission_weighted.csv', index=False)
    print("--- ✅ WEIGHTED GOD MODE HOÀN TẤT. KẾT QUẢ TẠI SUBMISSION_WEIGHTED.CSV! ✅ ---")

if __name__ == "__main__":
    train_weighted_god_mode()
