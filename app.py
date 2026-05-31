import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from scipy import stats
from statsmodels.tsa.stattools import acf

from src.data_prep import (
    load_and_prepare_data,
    detect_and_smooth_anomalies
)
from src.features import (
    ensure_stationarity,
    build_feature_matrix,
    filter_multicollinearity
)
from src.modeling import (
    fit_ols,
    evaluate_residuals,
    generate_dynamic_forecast,
    rolling_window_backtest
)

st.set_page_config(
    page_title="Прогнозування кількості відвідувачів на сайті",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Прогнозування кількості відвідувачів на сайті")
st.markdown("Курсова робота з аналітики даних")

if "horizon" not in st.session_state:
    st.session_state.horizon = 12
if "max_lags" not in st.session_state:
    st.session_state.max_lags = 3
if "train_window" not in st.session_state:
    st.session_state.train_window = 48
if "z_thresh" not in st.session_state:
    st.session_state.z_thresh = 3.0
if "ema_alpha" not in st.session_state:
    st.session_state.ema_alpha = 0.3
if "seasonal_style" not in st.session_state:
    st.session_state.seasonal_style = "Fourier & Peak Dummies"
if "include_exog" not in st.session_state:
    st.session_state.include_exog = False
if "include_trend" not in st.session_state:
    st.session_state.include_trend = False

st.sidebar.header("Конфігурація набору даних")
available_files = ["big-5-scaled.csv", "big-5-visitors.csv"]
selected_file = st.sidebar.selectbox("Оберіть набір даних", available_files)

DATA_PATH = os.path.join("datasets", selected_file)

if not os.path.exists(DATA_PATH):
    st.error(f"Набір даних не знайдено за шляхом `{DATA_PATH}`. Будь ласка, розмістіть файл у директорії datasets.")
    st.stop()


@st.cache_data
def get_clean_data(path):
    return load_and_prepare_data(path)


try:
    df_raw = get_clean_data(DATA_PATH)
except Exception as e:
    st.error(f"Помилка завантаження даних: {e}")
    st.stop()

st.sidebar.subheader("Параметри часового горизонту")
min_date = df_raw.index.min().to_pydatetime()
max_date = df_raw.index.max().to_pydatetime()
start_date, end_date = st.sidebar.slider(
    "Оберіть активний діапазон дат",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM"
)

df_filtered = df_raw.loc[start_date:end_date]

if len(df_filtered) < 24:
    st.error(
        "Обраний діапазон дат містить менше 24 місяців. Будь ласка, розширте вибір дат на бічній панелі, щоб забезпечити достатню кількість ступенів свободи.")
    st.stop()

target_cols = [col for col in df_filtered.columns if col not in ["Year", "Month", "USD"]]
target_company = st.sidebar.selectbox("Оберіть цільову компанію", target_cols)
exog_col = "USD"

y_raw = df_filtered[target_company]
exog_series = df_filtered[exog_col]

rolling_mean = y_raw.rolling(window=12, min_periods=1).mean()
rolling_std = y_raw.rolling(window=12, min_periods=1).std().fillna(0.0)
std_safe = np.where(rolling_std == 0.0, 1e-9, rolling_std)
z_scores = (y_raw - rolling_mean) / std_safe
detected_anomaly_mask = np.abs(z_scores) > st.session_state.z_thresh
detected_dates = y_raw.index[detected_anomaly_mask]
formatted_detected = [d.strftime("%Y-%m") for d in detected_dates]

if ("selected_anomalies" not in st.session_state
        or st.session_state.get("prev_target_company") != target_company
        or st.session_state.get("prev_file") != selected_file):
    st.session_state.selected_anomalies = formatted_detected
    st.session_state.prev_target_company = target_company
    st.session_state.prev_file = selected_file
else:
    st.session_state.selected_anomalies = [
        d for d in st.session_state.selected_anomalies if d in formatted_detected
    ]

y_active = y_raw.copy()
ema_series = y_raw.ewm(alpha=st.session_state.ema_alpha, adjust=False).mean()
for date_str in st.session_state.selected_anomalies:
    target_date = pd.to_datetime(date_str)
    if target_date in y_active.index:
        y_active.loc[target_date] = ema_series.loc[target_date]

y_stat, d = ensure_stationarity(y_active, max_diff=1)

best_beta = None
best_features = None
chosen_lags = st.session_state.max_lags
residuals = None
diagnostics = {}

for lag in range(st.session_state.max_lags, st.session_state.max_lags + 4):
    X, y_aligned = build_feature_matrix(
        y_stat,
        df_filtered[exog_col],
        max_lags=lag,
        include_trend=st.session_state.include_trend,
        include_exog=st.session_state.include_exog,
        seasonal_style=st.session_state.seasonal_style
    )
    X_filtered = filter_multicollinearity(X)
    beta, res = fit_ols(X_filtered, y_aligned)
    diag = evaluate_residuals(res)

    best_beta = beta
    best_features = list(X_filtered.columns)
    chosen_lags = lag
    residuals = res
    diagnostics = diag

    if (1.5 <= diag["dw_stat"] <= 2.5) and (diag["ljung_box_p"] > 0.05):
        break

trend_start = len(df_filtered) - len(y_stat)
future_dates = pd.date_range(
    start=df_filtered.index[-1] + pd.DateOffset(months=1),
    periods=st.session_state.horizon,
    freq="MS"
)

future_exog = pd.Series(index=future_dates, dtype=np.float64).fillna(exog_series.iloc[-1])
exog_combined = pd.concat([df_filtered[exog_col], future_exog])

y_active_slice = y_active.loc[y_stat.index]

_, pred_w = generate_dynamic_forecast(
    history_z=y_stat,
    history_w=y_active_slice,
    exog_all=exog_combined,
    beta=best_beta,
    feature_cols=best_features,
    max_lags=chosen_lags,
    d=d,
    horizon=st.session_state.horizon,
    trend_offset=trend_start,
    include_trend=st.session_state.include_trend,
    include_exog=st.session_state.include_exog,
    seasonal_style=st.session_state.seasonal_style
)

pred_raw = pd.Series(pred_w, index=future_dates)

try:
    y_fit_z = X_filtered @ beta
    y_active_aligned_for_shift = y_active.loc[y_fit_z.index]
    if d == 1:
        y_fit_w = y_active_aligned_for_shift.shift(1) + y_fit_z
    elif d == 2:
        y_fit_w = 2 * y_active_aligned_for_shift.shift(1) - y_active_aligned_for_shift.shift(2) + y_fit_z
    else:
        y_fit_w = y_fit_z

    y_fit_w = y_fit_w.dropna()
    y_active_aligned = y_active.loc[y_fit_w.index]
    y_fit_raw = y_fit_w
    raw_residual_std = np.std(y_active_aligned - y_fit_raw)

    if np.isnan(raw_residual_std) or raw_residual_std <= 0:
        raw_residual_std = np.std(y_active) * 0.1
except Exception:
    raw_residual_std = np.std(y_active) * 0.1

if d > 0:
    se_vector_raw = raw_residual_std * np.sqrt(np.arange(1, st.session_state.horizon + 1))
else:
    se_vector_raw = raw_residual_std * np.ones(st.session_state.horizon)

dof = max(1, len(residuals) - len(best_beta))

t_crit_95 = stats.t.ppf(0.95, dof)
pred_raw_upper_95 = pred_raw + t_crit_95 * se_vector_raw
pred_raw_lower_95 = pred_raw - t_crit_95 * se_vector_raw

t_crit_80 = stats.t.ppf(0.80, dof)
pred_raw_upper_80 = pred_raw + t_crit_80 * se_vector_raw
pred_raw_lower_80 = pred_raw - t_crit_80 * se_vector_raw

lower_limit = 0.0 if y_active.min() >= 0 else float(y_active.min()) * 1.5
pred_raw_upper_95 = np.clip(pred_raw_upper_95, lower_limit, None)
pred_raw_upper_80 = np.clip(pred_raw_upper_80, lower_limit, None)
pred_raw_lower_95 = np.clip(pred_raw_lower_95, lower_limit, None)
pred_raw_lower_80 = np.clip(pred_raw_lower_80, lower_limit, None)

tab_eda, tab_outliers, tab_diagnostics, tab_forecast = st.tabs([
    "Розвідувальний аналіз даних",
    "Обробка викидів",
    "Діагностична аналітика",
    "Прогнозування та оцінка точності",
])

COLOR_MAP = {
    "Епіцентр К": "#3174f1",
    "Epicentr K": "#3174f1",
    "Фокстрот": "#e65c00",
    "Foxtrot": "#e65c00",
    "OLX": "#00f0ff",
    "Prom.ua": "#a83dfa",
    "Prom": "#a83dfa",
    "Розетка": "#00cc66",
    "Rozetka": "#00cc66"
}

with tab_eda:
    st.subheader("Розвідувальний аналіз даних")
    st.markdown(
        "Порівняння базових моделей трафіку, довгострокових трендів та структурних зсувів між усіма платформами.")

    lbl_title_raw = "Порівняння пошукових запитів на сайти онлайн-платформ за середнім значенням на місяць"
    lbl_title_smooth = "Згладжений тренд пошукових запитів на сайти як 12-місячне ковзне середнє"
    lbl_xaxis = "Час"
    lbl_yaxis = "Індекс відносної популярності"
    lbl_shading = "Сезонний сплеск запитів у листопаді та грудні"

    show_shading = st.checkbox("Виділити сезонні піки в листопаді-грудні", value=True)

    eda_sub1, eda_sub2 = st.tabs(["Фактичний активний трафік", "12-місячний згладжений тренд"])

    shading_shapes = []
    if show_shading:
        years = sorted(list(set(df_filtered.index.year)))
        for y in years:
            start_str = f"{y}-11-01"
            end_str = f"{y}-12-31"
            start_ts = pd.Timestamp(start_str)
            end_ts = pd.Timestamp(end_str)
            if start_ts >= df_filtered.index.min() and end_ts <= df_filtered.index.max():
                shading_shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start_str,
                    x1=end_str,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(128, 128, 128, 0.15)",
                    layer="below",
                    line=dict(width=0),
                ))

    with eda_sub1:
        fig_raw_eda = go.Figure()

        if show_shading:
            fig_raw_eda.add_trace(go.Scatter(
                x=[], y=[],
                mode="markers",
                marker=dict(size=10, color="rgba(128, 128, 128, 0.3)", symbol="square"),
                name=lbl_shading,
                showlegend=True
            ))

        for col in target_cols:
            line_color = COLOR_MAP.get(col, None)
            fig_raw_eda.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered[col],
                name=col,
                line=dict(width=2, color=line_color)
            ))

        fig_raw_eda.update_layout(
            title=dict(text=lbl_title_raw, font=dict(size=16)),
            template="plotly_dark",
            xaxis_title=lbl_xaxis,
            yaxis_title=lbl_yaxis,
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            shapes=shading_shapes
        )
        st.plotly_chart(fig_raw_eda, use_container_width=True)

    with eda_sub2:
        fig_smooth_eda = go.Figure()

        if show_shading:
            fig_smooth_eda.add_trace(go.Scatter(
                x=[], y=[],
                mode="markers",
                marker=dict(size=10, color="rgba(128, 128, 128, 0.3)", symbol="square"),
                name=lbl_shading,
                showlegend=True
            ))

        for col in target_cols:
            line_color = COLOR_MAP.get(col, None)
            smoothed_series = df_filtered[col].rolling(window=12, min_periods=1).mean()
            fig_smooth_eda.add_trace(go.Scatter(
                x=df_filtered.index,
                y=smoothed_series,
                name=f"{col}",
                line=dict(width=2.5, color=line_color)
            ))

        fig_smooth_eda.update_layout(
            title=dict(text=lbl_title_smooth, font=dict(size=16)),
            template="plotly_dark",
            xaxis_title=lbl_xaxis,
            yaxis_title=lbl_yaxis,
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            shapes=shading_shapes
        )
        st.plotly_chart(fig_smooth_eda, use_container_width=True)

with tab_outliers:
    st.subheader("Очищення часового ряду")
    st.markdown("Аналіз послідовності стабілізації даних перед виконанням")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.slider("Поріг Z-оцінки для аномалій", min_value=1.5, max_value=5.0, step=0.1, key="z_thresh")
    with col_cfg2:
        st.slider("Коефіцієнт згладжування EMA α", min_value=0.05, max_value=1.0, step=0.05, key="ema_alpha")

    st.markdown("#### Обробка аномалій")
    st.write(f"Виявлено аномалій в активному вікні: **{len(formatted_detected)}**")

    selected_anomalies_str = st.multiselect(
        "Оберіть конкретні дати для згладжування, необрані дати залишаться без змін:",
        options=formatted_detected,
        default=st.session_state.selected_anomalies,
        key="anomaly_multiselect_widget"
    )

    if selected_anomalies_str != st.session_state.selected_anomalies:
        st.session_state.selected_anomalies = selected_anomalies_str
        st.rerun()

    step_col1, step_col2 = st.columns(2)

    with step_col1:
        st.markdown("**Графічне зображення аномалій**")
        fig_step1 = go.Figure()
        fig_step1.add_trace(
            go.Scatter(x=df_filtered.index, y=df_filtered[target_company], name="Вихідні дані",
                       line=dict(color="#64748b")))
        fig_step1.add_trace(
            go.Scatter(x=df_filtered.index, y=y_active, name="Згладжений сигнал", line=dict(color="#38bdf8")))
        fig_step1.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(fig_step1, use_container_width=True)

    with step_col2:
        st.markdown("**Стаціонарний графік**")
        fig_step3 = go.Figure()
        fig_step3.add_trace(go.Scatter(x=y_stat.index, y=y_stat, name="Стаціонарний ряд", line=dict(color="#c084fc")))
        fig_step3.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(fig_step3, use_container_width=True)

with tab_forecast:
    st.subheader(f"Прогноз та оцінка точності для веб-сайту {target_company}")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        st.slider("Горизонт прогнозування у місяцях", min_value=1, max_value=24, key="horizon")
    with col_cfg2:
        st.slider("Базові авторегресійні лаги", min_value=1, max_value=12, key="max_lags")
    with col_cfg3:
        max_train_months = max(12, len(df_filtered) - st.session_state.horizon - 1)
        if st.session_state.train_window > max_train_months:
            st.session_state.train_window = max_train_months
        st.slider("Ширина навчального вікна у місяцях", min_value=24, max_value=max_train_months, key="train_window")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        seasonal_style_map = {
            "Fourier & Peak Dummies": "Фур'є та фіктивні змінні піків",
            "Full Monthly Dummies": "Повні місячні фіктивні змінні",
            "None": "Немає"
        }
        reverse_seasonal_style_map = {v: k for k, v in seasonal_style_map.items()}

        current_style_ukr = seasonal_style_map.get(st.session_state.seasonal_style, "Фур'є та фіктивні змінні піків")
        selected_style_ukr = st.selectbox(
            "Сезонні ознаки",
            list(reverse_seasonal_style_map.keys()),
            index=list(reverse_seasonal_style_map.keys()).index(current_style_ukr)
        )
        st.session_state.seasonal_style = reverse_seasonal_style_map[selected_style_ukr]
    with col_opt2:
        st.markdown("")
        st.markdown("")
        st.checkbox("Врахувати курс долара", key="include_exog")

    try:
        backtest_results = rolling_window_backtest(
            df_filtered,
            target_col=target_company,
            exog_col=exog_col,
            train_window=st.session_state.train_window,
            horizon=st.session_state.horizon,
            max_lags=st.session_state.max_lags,
            include_trend=st.session_state.include_trend,
            include_exog=st.session_state.include_exog,
            seasonal_style=st.session_state.seasonal_style,
            selected_anomalies=st.session_state.selected_anomalies,
            ema_alpha=st.session_state.ema_alpha
        )
        metrics = backtest_results["metrics"]
        backtest_error = None
    except Exception as e:
        metrics = None
        backtest_error = f"Моделювання бектесту завершилося помилкою: {e}"

    st.markdown("---")

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Середня абсолютна помилка MAE", f"{metrics['MAE']:.4f}")
        m2.metric("Середньоквадратична помилка RMSE", f"{metrics['RMSE']:.4f}")
        m3.metric("Симетрична середня помилка SMAPE", f"{metrics['SMAPE']:.2f}%")
        m4.metric("Зважена відсоткова помилка WMAPE", f"{metrics['WMAPE']:.2f}%")
    elif backtest_error:
        st.error(backtest_error)

    st.subheader("Прогнозний розрахунок")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(pred_raw_upper_95) + list(pred_raw_lower_95)[::-1],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="95% довірчий інтервал"
    ))

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(pred_raw_upper_80) + list(pred_raw_lower_80)[::-1],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="80% довірчий інтервал"
    ))

    fig.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered[target_company], name="Фактичні історичні",
                   line=dict(color="#64748b")))

    if len(st.session_state.selected_anomalies) > 0:
        fig.add_trace(
            go.Scatter(x=df_filtered.index, y=y_active, name="Згладжені аномалії",
                       line=dict(color="#0ea5e9", dash="dot")))

    fig.add_trace(
        go.Scatter(x=future_dates, y=pred_raw, name="Середня лінія прогнозу", line=dict(color="#10b981", width=3)))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Дата",
        yaxis_title="Показники обсягу",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_diagnostics:
    st.subheader("Параметризація моделі та діагностична аналітика")
    st.markdown(
        "Оцінка припущень класичної лінійної регресійної моделі для OLS залишків $\\epsilon_t$ за допомогою формальних статистичних тестів.")

    jb_stat, jb_p = stats.jarque_bera(residuals)
    half_split = len(residuals) // 2
    lev_stat, lev_p = stats.levene(residuals.iloc[:half_split], residuals.iloc[half_split:])
    lb_p = diagnostics["ljung_box_p"]

    st.markdown("---")
    st.markdown("### 1. Припущення про нормальність розподілу залишків")
    col_norm_left, col_norm_right = st.columns([1, 1])

    with col_norm_left:
        st.markdown("**Тест Харке-Бера**")
        st.latex(
            r"\begin{aligned} H_0 &: \text{Залишки розподілені нормально} \\ H_1 &: \text{Залишки не є нормально розподіленими} \end{aligned}")
        st.markdown("**Статистичні показники**")
        st.write(f"- Статистика тесту: `{jb_stat:.4f}`")
        st.write(f"- p-значення: `{jb_p:.4e}`")

        st.markdown("**Рішення та висновки**")
        if jb_p < 0.05:
            st.error(r"$H_0 \text{ відхиляється} \text{ на рівні значущості } \alpha = 0.05$")
            st.info(
                "Залишки не мають суто нормального розподілу. Це типово для часових рядів трафіку через волатильні коливання. Точкові оцінки OLS залишаються незміщеними, але стандартні помилки та довірчі інтервали можуть відхилятися від теоретичних значений.")
        else:
            st.success(r"$H_0 \text{ не відхиляється} \text{ на рівні значущості } \alpha = 0.05$")
            st.info(
                "Розподіл залишків статистично не відрізняється від нормального розподілу. Параметричні довірчі межі та статистичні висновки є надійними.")

    with col_norm_right:
        res_mean, res_std = residuals.mean(), residuals.std()
        norm_x = np.linspace(residuals.min(), residuals.max(), 100)
        norm_y = stats.norm.pdf(norm_x, res_mean, res_std)

        fig_hist = px.histogram(
            residuals,
            nbins=30,
            title="Розподіл залишків",
            labels={'value': 'Значення залишку'},
            template="plotly_dark"
        )
        fig_hist.add_trace(go.Scatter(x=norm_x, y=norm_y * len(residuals) * (residuals.max() - residuals.min()) / 30,
                                      name="Нормальний розподіл", line=dict(color="#f43f5e", width=2)))
        fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=280)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("### 2. Припущення про гомоскедастичність залишків")
    col_hom_left, col_hom_right = st.columns([1, 1])

    with col_hom_left:
        st.markdown("**Тест Лівена**")
        st.latex(
            r"\begin{aligned} H_0 &: \text{Постійна дисперсія} \\ H_1 &: \text{Дисперсія змінюється з часом} \end{aligned}")
        st.markdown("**Статистичні показники**")
        st.write(f"- Статистика тесту Лівена: `{lev_stat:.4f}`")
        st.write(f"- p-значення: `{lev_p:.4f}`")

        st.markdown("**Рішення та висновки**")
        if lev_p < 0.05:
            st.error(r"$H_0 \text{ відхиляється} \text{ на рівні значущості } \alpha = 0.05$")
            st.info(
                "Дисперсія залишків не є постійною. Це свідчить про наявність гетероскедастичності, що може потребувати ширших довірчих інтервалів для окремих періодів волатильності.")
        else:
            st.success(r"$H_0 \text{ не відхиляється} \text{ на рівні значущості } \alpha = 0.05$")
            st.info(
                "Дисперсія залишків залишається стабільною. Оцінки стандартних помилок OLS є надійними.")

    with col_hom_right:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=residuals.index,
            y=residuals,
            mode="markers",
            marker=dict(color="#38bdf8", size=5),
            name="Залишки"
        ))
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="#ef4444")
        fig_scatter.update_layout(
            title="Значення залишків на часовій шкалі",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            height=280
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### 3. Припущення про відсутність автокореляції залишків")
    col_auto_left, col_auto_right = st.columns([1, 1])

    with col_auto_left:
        st.markdown("**Тест Льюнга-Бокса та Дарбіна-Уотсона**")
        st.latex(
            r"\begin{aligned} H_0 &: \text{Відсутність серійної кореляції} \\ H_1 &: \text{Залишки демонструють серійну автокореляцію} \end{aligned}")
        st.markdown("**Статистичні показники**")
        st.write(f"- Статистика Дарбіна-Уотсона: `{diagnostics['dw_stat']:.4f}`")
        st.write(f"- p-значення Ljung-Box: `{lb_p:.4f}`")

        st.markdown("**Рішення та висновки**")
        if lb_p < 0.05:
            st.error(r"$H_0 \text{ відхиляється} \text{ на рівні значущості } \alpha = 0.05$")
            st.info(
                "Залишки демонструють автокореляцію. Це означає, що деякі циклічні коливання не повністю враховані параметрами моделі.")
        else:
            st.success(r"$H_0 \text{ не відхиляється} \text{ на рівні значущості } \alpha = 0.05$")
            st.info(
                "Помилки є незалежними. Усі закономірності часового ряду враховані у параметрах моделі.")

        if chosen_lags > st.session_state.max_lags:
            st.warning(
                f"Примітка: активовано адаптивне пом'якшення, лаги збільшено до {chosen_lags} для забезпечення властивостей білого шуму.")

    with col_auto_right:
        acf_vals = acf(residuals, nlags=20, fft=True)
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=np.arange(len(acf_vals)), y=acf_vals, marker_color="#34d399", name="ACF"))

        conf_int = 1.96 / np.sqrt(len(residuals))
        fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="#ef4444")
        fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="#ef4444")
        fig_acf.update_layout(
            title="Автокореляційна функція ACF з 95% довірчими межами",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            height=280
        )
        st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown("---")
    st.markdown("### 4. Зведена таблиця параметрів регресії")
    feature_df = pd.DataFrame({
        "Регресори": best_features,
        "Коефіцієнти": best_beta
    })
    st.dataframe(feature_df.set_index("Регресори"), use_container_width=True)

    st.markdown("### 5. Підсумок")
    st.info(r"$\text{На рівні значущості } \alpha = 0.05$:")
    if jb_p < 0.05:
        st.error("Розподіл залишків суттєво відрізняється від нормального.")
    else:
        st.success("Гіпотезу про нормальний розподіл залишків не було відхилено.")

    if lev_p < 0.05:
        st.error("Виявлено гетероскедастичність - дисперсія залишків не є сталою.")
    else:
        st.success("Ознак гетероскедастичності не виявлено - дисперсія залишків стала.")

    if lb_p < 0.05:
        st.error("Виявлено автокореляцію в залишках.")
    else:
        st.success("Автокореляції в залишках не виявлено.")