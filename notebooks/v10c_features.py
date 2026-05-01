"""V10c full feature builder — đầy đủ 23 features.
Import vào notebook: from v10c_features import build_features, V10C_FEATS, TET
"""
import numpy as np
import pandas as pd

TET = [pd.Timestamp(d) for d in [
    "2017-01-28","2018-02-16","2019-02-05","2020-01-25",
    "2021-02-12","2022-02-01","2023-01-22","2024-02-10"
]]

def nth_weekday_of_month(y, m, wd, n):
    """Tìm ngày thứ `wd` lần thứ `n` trong tháng `m` năm `y`."""
    d = pd.Timestamp(y, m, 1)
    shift = (wd - d.weekday()) % 7
    return d + pd.Timedelta(days=shift + 7*(n-1))

def build_features(df, traffic_df=None, traffic_avg=None):
    """Build V10c's full 23 features từ Date.
    
    Parameters
    ----------
    df : DataFrame cần có cột 'Date'
    traffic_df : DataFrame daily traffic (dùng cho train) — cần cột Date, sessions, page_views, unique_visitors, bounce_rate, avg_sess
    traffic_avg : DataFrame traffic trung bình theo month-dow (dùng cho test khi chưa có actual traffic)
    """
    df = df.copy()
    df["year"]  = df.Date.dt.year
    df["month"] = df.Date.dt.month
    df["day"]   = df.Date.dt.day
    df["dow"]   = df.Date.dt.dayofweek

    # --- Web traffic ---
    if traffic_df is not None:
        df = df.merge(traffic_df[["Date","sessions","page_views","unique_visitors","bounce_rate","avg_sess"]],
                      on="Date", how="left")
    elif traffic_avg is not None:
        df = df.merge(traffic_avg, on=["month","dow"], how="left")
    for c in ["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            df[c] = 0

    # --- Cyclical encoding (sin/cos cho biến tuần hoàn) ---
    df["month_sin"] = np.sin(2*np.pi*df.month/12)
    df["month_cos"] = np.cos(2*np.pi*df.month/12)
    df["dow_sin"]   = np.sin(2*np.pi*df.dow/7)
    df["dow_cos"]   = np.cos(2*np.pi*df.dow/7)

    # --- Event flags ---
    df["is_payday"]      = ((df.day >= 25) | (df.day <= 5)).astype(int)
    df["is_double_day"]  = (df.month == df.day).astype(int)

    # Tết season (21 ngày trước Tết âm lịch)
    df["is_tet_season"] = 0
    for td in TET:
        m = (df.Date >= td - pd.Timedelta(days=21)) & (df.Date < td)
        df.loc[m, "is_tet_season"] = 1

    # Holiday periods (lễ lớn ± 3 ngày trước)
    df["is_holiday_period"] = 0
    for mh, dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0, 4):
            df.loc[(df.month == mh) & (df.day == (dh - off)), "is_holiday_period"] = 1

    # Singles Day, 9/9, 12/12
    df["is_singles_day"]   = ((df.month == 11) & (df.day == 11)).astype(int)
    df["is_nine_nine"]     = ((df.month == 9) & (df.day == 9)).astype(int)
    df["is_twelve_twelve"] = ((df.month == 12) & (df.day == 12)).astype(int)

    # Black Friday (thứ 6 tuần thứ 4 tháng 11)
    df["is_black_friday"] = 0
    for y in df.year.unique():
        bf = nth_weekday_of_month(int(y), 11, 4, 4)
        df.loc[df.Date == bf, "is_black_friday"] = 1

    # Cyber Monday (thứ 2 sau Thanksgiving = thứ 5 tuần thứ 4 tháng 11 + 4 ngày)
    df["is_cyber_monday"] = 0
    for y in df.year.unique():
        thx = nth_weekday_of_month(int(y), 11, 3, 4)
        df.loc[df.Date == thx + pd.Timedelta(days=4), "is_cyber_monday"] = 1

    # Mother's Day (Chủ nhật tuần thứ 2 tháng 5)
    df["is_mothers_day"] = 0
    for y in df.year.unique():
        md = nth_weekday_of_month(int(y), 5, 6, 2)
        df.loc[df.Date == md, "is_mothers_day"] = 1

    # Near flash event (±3 ngày quanh các ngày sale lớn)
    flash = (df["is_singles_day"] + df["is_twelve_twelve"] + df["is_nine_nine"]
             + df["is_black_friday"] + df["is_cyber_monday"])
    df["near_flash_event"] = 0
    for off in range(-3, 4):
        shifted = flash.shift(off).fillna(0)
        df["near_flash_event"] = np.maximum(df["near_flash_event"], (shifted > 0).astype(int))

    return df


V10C_FEATS = [
    "month_sin", "month_cos", "dow_sin", "dow_cos", "day", "month", "dow",
    "is_payday", "is_double_day", "is_tet_season", "is_holiday_period",
    "is_singles_day", "is_nine_nine", "is_twelve_twelve",
    "is_black_friday", "is_cyber_monday", "is_mothers_day", "near_flash_event",
    "sessions", "page_views", "unique_visitors", "bounce_rate", "avg_sess",
]
