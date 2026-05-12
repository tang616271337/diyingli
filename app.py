import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util


def nearest_pressure(df, time_value):
    """Return pressure at the row nearest to a selected time."""
    idx = (df['T'] - time_value).abs().idxmin()
    return float(df.loc[idx, 'P'])


def piecewise_linear_closure(x, y, min_side=4):
    """Find a pressure-closure point by two-segment linear fitting."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < min_side * 2 + 1 or np.nanmax(x) == np.nanmin(x):
        return None

    best = None
    for split in range(min_side, len(x) - min_side):
        left_coef = np.polyfit(x[:split], y[:split], 1)
        right_coef = np.polyfit(x[split:], y[split:], 1)
        left_fit = np.polyval(left_coef, x[:split])
        right_fit = np.polyval(right_coef, x[split:])
        sse = float(np.sum((y[:split] - left_fit) ** 2) + np.sum((y[split:] - right_fit) ** 2))
        if best is None or sse < best['sse']:
            best = {'idx': split, 'x': float(x[split]), 'p': float(y[split]), 'sse': sse}

    return best


def detect_slope_change_pressure(x, y, min_side=4):
    """Find reopening pressure from the slope-break point on a rising curve."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < min_side * 2 + 1 or np.nanmax(x) == np.nanmin(x):
        if len(y):
            idx = int(np.nanargmax(y))
            return {'idx': idx, 'x': float(x[idx]), 'p': float(y[idx]), 'left_coef': None, 'right_coef': None}
        return None

    y_smooth = pd.Series(y).rolling(5, center=True, min_periods=1).median().to_numpy()
    best = None
    fallback = None
    for split in range(min_side, len(x) - min_side):
        left_coef = np.polyfit(x[:split], y_smooth[:split], 1)
        right_coef = np.polyfit(x[split:], y_smooth[split:], 1)
        left_fit = np.polyval(left_coef, x[:split])
        right_fit = np.polyval(right_coef, x[split:])
        sse = float(np.sum((y_smooth[:split] - left_fit) ** 2) + np.sum((y_smooth[split:] - right_fit) ** 2))
        slope_drop = float(left_coef[0] - right_coef[0])
        left_r2 = fit_quality_from_coef(x[:split], y_smooth[:split], left_coef)
        right_r2 = fit_quality_from_coef(x[split:], y_smooth[split:], right_coef)
        candidate = {
            'idx': split,
            'x': float(x[split]),
            'p': float(y[split]),
            'sse': sse,
            'slope_drop': slope_drop,
            'left_coef': left_coef,
            'right_coef': right_coef,
            'r2': float((left_r2 + right_r2) / 2.0),
            'left_r2': float(left_r2),
            'right_r2': float(right_r2),
        }
        if fallback is None or sse < fallback['sse']:
            fallback = candidate
        if left_coef[0] > 0 and slope_drop > 0:
            slope_ratio = right_coef[0] / max(left_coef[0], 1e-9)
            score = candidate['r2'] + min(slope_drop / max(left_coef[0], 1e-9), 1.0) * 0.20
            if slope_ratio > 0.95:
                score -= 0.25
            candidate['score'] = score
            if best is None or score > best['score']:
                best = candidate

    return best or fallback


def cumulative_pressurization_volume(time_s, flow_rate):
    """Return relative injected volume for reopening-pressure interpretation."""
    time_s = np.asarray(time_s, dtype=float)
    flow_rate = np.asarray(flow_rate, dtype=float)
    if len(time_s) < 2:
        return np.zeros_like(time_s)

    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.maximum(dt, 0.0)
    positive_rate = np.maximum(flow_rate, 0.0)
    volume = np.cumsum(positive_rate * dt)
    return volume - volume[0]


def detect_reopening_pressure(df, start_idx, stop_idx):
    """Detect Pr from the stiffness drop on the pre-shut-in rising segment."""
    window = df.loc[start_idx:stop_idx].copy()
    if len(window) < 4:
        idx = int(window['P'].idxmax())
        return float(df.loc[idx, 'T']), float(df.loc[idx, 'P'])

    peak_idx = int(window['P'].idxmax())
    rising = df.loc[start_idx:peak_idx].copy()
    if len(rising) < 4:
        return float(df.loc[peak_idx, 'T']), float(df.loc[peak_idx, 'P'])

    volume = cumulative_pressurization_volume(rising['T'], rising['Q'])
    fit = detect_slope_change_pressure(volume, rising['P'])
    if not fit or np.nanmax(volume) == np.nanmin(volume):
        x = rising['T'] - float(rising['T'].iloc[0])
        fit = detect_slope_change_pressure(x, rising['P'])
    if not fit:
        return float(df.loc[peak_idx, 'T']), float(df.loc[peak_idx, 'P'])

    row = rising.iloc[int(fit['idx'])]
    return float(row['T']), float(row['P'])


def detect_sqrt_closure(x, y, min_points=6):
    """Detect closure pressure by linear-leakoff deviation on P-sqrt(dt)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) > 450:
        sample_idx = np.unique(np.linspace(0, len(x) - 1, 450).astype(int))
        x = x[sample_idx]
        y = y[sample_idx]

    if len(x) < min_points * 2 or np.nanmax(x) == np.nanmin(x):
        return piecewise_linear_closure(x, y, min_side=4)

    y_smooth = pd.Series(y).rolling(5, center=True, min_periods=1).median().to_numpy()
    n = len(x)
    start_min = max(1, int(n * 0.04))
    search_end = max(start_min + min_points, int(n * 0.65))
    best = None

    start_candidates = np.unique(np.linspace(start_min, max(start_min, int(n * 0.35)), min(35, max(2, n // 8))).astype(int))
    for start in start_candidates:
        end_min = start + min_points
        if end_min >= search_end:
            continue
        end_candidates = np.unique(np.linspace(end_min, search_end, min(45, max(2, (search_end - end_min) // 4))).astype(int))
        for end in end_candidates:
            xs = x[start:end]
            ys = y_smooth[start:end]
            if np.nanmax(xs) == np.nanmin(xs):
                continue
            coef = np.polyfit(xs, ys, 1)
            if coef[0] >= 0:
                continue
            fit = np.polyval(coef, xs)
            ss_res = float(np.sum((ys - fit) ** 2))
            ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            span = float(xs[-1] - xs[0])
            score = r2 + min(span / max(np.nanmax(x) - np.nanmin(x), 1e-6), 1.0) * 0.15
            if best is None or score > best['score']:
                best = {
                    'start': start,
                    'end': end,
                    'coef': coef,
                    'score': score,
                    'r2': r2,
                }

    if best is None:
        return piecewise_linear_closure(x, y, min_side=4)

    coef = best['coef']
    y_fit = np.polyval(coef, x)
    residual = y_smooth - y_fit
    base_residual = residual[best['start']:best['end']]
    threshold = max(float(np.nanstd(base_residual)) * 2.5, max(float(np.nanstd(y_smooth)) * 0.025, 0.05))
    closure_idx = None

    for idx in range(best['end'], n - 2):
        if np.all(residual[idx:idx + 3] < -threshold):
            closure_idx = idx
            break

    if closure_idx is None:
        fallback = piecewise_linear_closure(x, y, min_side=4)
        closure_idx = int(fallback['idx']) if fallback and 'idx' in fallback else best['end']

    right_end = min(n, closure_idx + max(min_points, n // 5))
    right_coef = None
    if right_end - closure_idx >= 3:
        right_coef = np.polyfit(x[closure_idx:right_end], y_smooth[closure_idx:right_end], 1)

    return {
        'idx': int(closure_idx),
        'x': float(x[closure_idx]),
        'p': float(y[closure_idx]),
        'coef': coef,
        'right_coef': right_coef,
        'line_start': int(best['start']),
        'line_end': int(best['end']),
        'threshold': threshold,
        'r2': float(best['r2']),
    }


def fit_segment_quality(x, y, start, end):
    if end - start < 3:
        return None
    xs = np.asarray(x[start:end], dtype=float)
    ys = np.asarray(y[start:end], dtype=float)
    if np.nanmax(xs) == np.nanmin(xs):
        return None
    coef = np.polyfit(xs, ys, 1)
    fit = np.polyval(coef, xs)
    ss_res = float(np.sum((ys - fit) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'coef': coef, 'r2': r2, 'start': int(start), 'end': int(end)}


def find_derivative_flow_segments(dt, pressure):
    """Identify transient-flow and post-closure windows from log pressure derivative."""
    dt = np.asarray(dt, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    valid = np.isfinite(dt) & np.isfinite(pressure) & (dt > 0)
    dt = dt[valid]
    pressure = pressure[valid]

    if len(dt) > 450:
        idx = np.unique(np.linspace(0, len(dt) - 1, 450).astype(int))
        dt = dt[idx]
        pressure = pressure[idx]

    if len(dt) < 18:
        return None

    log_t = np.log10(dt.clip(min=1e-6))
    pressure_drop = pressure[0] - pressure
    deriv = np.abs(np.gradient(pressure_drop, log_t))
    log_deriv = np.log10(np.maximum(deriv, 1e-6))
    log_deriv = pd.Series(log_deriv).rolling(5, center=True, min_periods=1).median().to_numpy()

    n = len(log_t)
    win = max(6, min(28, n // 8))
    candidates = []
    for start in range(1, max(2, n - win - 1), max(1, win // 3)):
        end = start + win
        quality = fit_segment_quality(log_t, log_deriv, start, end)
        if not quality:
            continue
        slope = float(quality['coef'][0])
        center = (start + end) / 2 / n
        candidates.append({**quality, 'slope': slope, 'center': center})

    if not candidates:
        return None

    early_targets = [0.5, 0.25]
    early_candidates = [
        item for item in candidates
        if item['center'] < 0.7 and item['slope'] > 0.05
    ]
    late_candidates = [
        item for item in candidates
        if item['center'] > 0.35
    ]

    if not early_candidates or not late_candidates:
        return None

    early = max(
        early_candidates,
        key=lambda item: item['r2'] - min(abs(item['slope'] - target) for target in early_targets),
    )
    late = max(
        late_candidates,
        key=lambda item: item['r2'] - abs(item['slope'] - 0.0),
    )

    if late['start'] <= early['end']:
        later = [item for item in late_candidates if item['start'] > early['end']]
        if later:
            late = max(later, key=lambda item: item['r2'] - abs(item['slope']))
        else:
            return None

    return {
        'early': early,
        'late': late,
        'log_t': log_t,
        'log_deriv': log_deriv,
    }


def line_intersection_x(coef1, coef2):
    denom = coef1[0] - coef2[0]
    if abs(denom) < 1e-9:
        return None
    return float((coef2[1] - coef1[1]) / denom)


def fit_quality_from_coef(x, y, coef):
    """Return R-squared for a fitted line over one diagnostic segment."""
    fit = np.polyval(coef, x)
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_g_function(dt, pump_duration, alpha=1.0):
    """
    Calculate Nolte G-function time for a shut-in pressure decline.

    The G-function approximates cumulative Carter leakoff after shut-in. The
    default alpha=1.0 is commonly used for low-leakoff or high-efficiency tests.
    """
    dt = np.asarray(dt, dtype=float)
    duration = max(float(pump_duration), 1.0)
    dt_d = np.maximum(dt, 0.0) / duration

    if abs(alpha - 0.5) < 1e-9:
        g_value = (
            (1.0 + dt_d) * np.arcsin(1.0 / np.sqrt(1.0 + dt_d))
            + np.sqrt(dt_d)
        )
        g0 = np.pi / 2.0
    else:
        g_value = (4.0 / 3.0) * ((1.0 + dt_d) ** 1.5 - dt_d ** 1.5)
        g0 = 4.0 / 3.0

    return (4.0 / np.pi) * (g_value - g0)


def detect_sqrt_closure_with_derivative(dt, sqrt_dt, pressure):
    """Use derivative flow windows, then intersect two lines on P-sqrt(dt)."""
    dt = np.asarray(dt, dtype=float)
    sqrt_dt = np.asarray(sqrt_dt, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    valid = np.isfinite(dt) & np.isfinite(sqrt_dt) & np.isfinite(pressure) & (dt > 0)
    dt = dt[valid]
    sqrt_dt = sqrt_dt[valid]
    pressure = pressure[valid]

    if len(dt) > 450:
        idx = np.unique(np.linspace(0, len(dt) - 1, 450).astype(int))
        dt = dt[idx]
        sqrt_dt = sqrt_dt[idx]
        pressure = pressure[idx]

    segments = find_derivative_flow_segments(dt, pressure)
    if not segments:
        fallback = detect_sqrt_closure(sqrt_dt, pressure)
        if fallback:
            fallback['source'] = 'sqrt_deviation'
        return fallback

    early = segments['early']
    late = segments['late']
    early_fit = fit_segment_quality(sqrt_dt, pressure, early['start'], early['end'])
    late_fit = fit_segment_quality(sqrt_dt, pressure, late['start'], late['end'])
    if not early_fit or not late_fit:
        fallback = detect_sqrt_closure(sqrt_dt, pressure)
        if fallback:
            fallback['source'] = 'sqrt_deviation'
        return fallback

    x_cross = line_intersection_x(early_fit['coef'], late_fit['coef'])
    if x_cross is None or x_cross < np.nanmin(sqrt_dt) or x_cross > np.nanmax(sqrt_dt):
        fallback = detect_sqrt_closure(sqrt_dt, pressure)
        if fallback:
            fallback['source'] = 'sqrt_deviation'
        return fallback

    p_cross = float(np.polyval(early_fit['coef'], x_cross))
    idx = int(np.nanargmin(np.abs(sqrt_dt - x_cross)))

    return {
        'idx': idx,
        'x': x_cross,
        'p': p_cross,
        'coef': early_fit['coef'],
        'right_coef': late_fit['coef'],
        'line_start': early_fit['start'],
        'line_end': early_fit['end'],
        'right_start': late_fit['start'],
        'right_end': late_fit['end'],
        'r2': float((early_fit['r2'] + late_fit['r2']) / 2.0),
        'early_r2': early_fit['r2'],
        'late_r2': late_fit['r2'],
        'source': 'derivative_sqrt_intersection',
        'early_slope': early['slope'],
        'late_slope': late['slope'],
        'early_derivative_r2': early['r2'],
        'late_derivative_r2': late['r2'],
        'derivative': segments,
    }


def detect_g_function_closure(g_value, pressure, min_points=6):
    """
    Detect closure pressure with G-function pressure and derivative diagnostics.

    The method follows the G-derivative concept: before closure, G*(-dP/dG)
    should approximately form a straight line through the origin. For flowback
    assisted tests, closure is picked at the sustained upward departure point.
    """
    g_value = np.asarray(g_value, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    valid = np.isfinite(g_value) & np.isfinite(pressure)
    g_value = g_value[valid]
    pressure = pressure[valid]

    if len(g_value) > 450:
        idx = np.unique(np.linspace(0, len(g_value) - 1, 450).astype(int))
        g_value = g_value[idx]
        pressure = pressure[idx]

    if len(g_value) < min_points * 2 or np.nanmax(g_value) == np.nanmin(g_value):
        fallback = piecewise_linear_closure(g_value, pressure, min_side=4)
        if fallback:
            fallback['source'] = 'piecewise_fallback'
        return fallback

    pressure_smooth = pd.Series(pressure).rolling(5, center=True, min_periods=1).median().to_numpy()
    dp_dg = np.gradient(pressure_smooth, g_value)
    g_derivative = g_value * np.maximum(-dp_dg, 0.0)
    g_derivative = pd.Series(g_derivative).rolling(5, center=True, min_periods=1).median().to_numpy()
    n = len(g_value)
    search_end = max(min_points * 2, int(n * 0.65))
    best = None

    for start in range(1, max(2, int(n * 0.20)) + 1):
        end_min = start + min_points
        if end_min >= search_end:
            continue
        end_candidates = np.unique(np.linspace(
            end_min,
            search_end,
            min(40, max(2, search_end - end_min)),
        ).astype(int))
        for end in end_candidates:
            x_seg = g_value[start:end]
            y_seg = g_derivative[start:end]
            if np.nanmax(x_seg) == np.nanmin(x_seg):
                continue
            slope = float(np.sum(x_seg * y_seg) / max(np.sum(x_seg ** 2), 1e-12))
            if slope <= 0:
                continue
            fit = slope * x_seg
            ss_res = float(np.sum((y_seg - fit) ** 2))
            ss_tot = float(np.sum((y_seg - np.mean(y_seg)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            span = float(x_seg[-1] - x_seg[0])
            span_score = span / max(float(np.nanmax(g_value) - np.nanmin(g_value)), 1e-6)
            score = r2 + min(span_score, 1.0) * 0.10
            if best is None or score > best['score']:
                best = {
                    'start': int(start),
                    'end': int(end),
                    'derivative_slope': slope,
                    'r2': float(r2),
                    'score': float(score),
                }

    if best is None:
        fallback = piecewise_linear_closure(g_value, pressure, min_side=4)
        if fallback:
            fallback['source'] = 'piecewise_fallback'
        return fallback

    pressure_coef = np.polyfit(
        g_value[best['start']:best['end']],
        pressure_smooth[best['start']:best['end']],
        1,
    )
    baseline = best['derivative_slope'] * g_value
    residual = g_derivative - baseline
    base = residual[best['start']:best['end']]
    base_std = float(np.nanstd(base))
    derivative_scale = max(float(np.nanmax(g_derivative) - np.nanmin(g_derivative)), 1e-6)
    threshold = max(base_std * 2.5, derivative_scale * 0.04)

    closure_idx = None
    for idx in range(best['end'], n - 2):
        if np.all(residual[idx:idx + 3] > threshold):
            closure_idx = idx
            break

    if closure_idx is None:
        fallback = piecewise_linear_closure(g_value, pressure, min_side=4)
        closure_idx = int(fallback['idx']) if fallback and 'idx' in fallback else best['end']

    right_end = min(n, closure_idx + max(min_points, n // 5))
    right_coef = None
    if right_end - closure_idx >= 3:
        right_coef = np.polyfit(g_value[closure_idx:right_end], pressure_smooth[closure_idx:right_end], 1)

    return {
        'idx': int(closure_idx),
        'x': float(g_value[closure_idx]),
        'p': float(pressure[closure_idx]),
        'coef': pressure_coef,
        'right_coef': right_coef,
        'line_start': best['start'],
        'line_end': best['end'],
        'derivative': g_derivative,
        'derivative_slope': best['derivative_slope'],
        'derivative_baseline': baseline,
        'threshold': threshold,
        'r2': best['r2'],
        'source': 'g_derivative_upturn',
    }


def detect_pressure_points(df):
    """Detect hydraulic fracturing key points from pressure and flow curves."""
    result = {}
    pb_idx = df['P'].idxmax()
    result['pb_t'] = float(df.loc[pb_idx, 'T'])
    result['pb_p'] = float(df.loc[pb_idx, 'P'])

    q_limit = max(0.05, float(df['Q'].max()) * 0.03)
    pumping = df['Q'] > q_limit
    shut_candidates = df.index[pumping.shift(fill_value=False) & ~pumping]
    after_pb = [idx for idx in shut_candidates if df.loc[idx, 'T'] >= result['pb_t']]
    shut_idx = after_pb[0] if after_pb else (shut_candidates[0] if len(shut_candidates) else pb_idx)
    result['shut_t'] = float(df.loc[shut_idx, 'T'])
    result['isip_p'] = float(df.loc[shut_idx, 'P'])

    result['pr_t'], result['pr_p'] = detect_reopening_pressure(df, int(df.index[0]), int(pb_idx))

    return result


def build_analysis_methods(analysis_df, pump_duration):
    """Calculate closure pressure using several pressure-decline diagnostics."""
    methods = {}
    work = analysis_df[analysis_df['dt'] > 0].copy()
    if len(work) < 9:
        return methods

    sqrt_fit = detect_sqrt_closure_with_derivative(work['dt'], work['sqrt_dt'], work['P'])
    if sqrt_fit:
        methods['sqrt'] = {
            'name': '平方根时间法',
            'pressure': sqrt_fit['p'],
            'x': sqrt_fit['x'],
            'series_x': work['sqrt_dt'],
            'series_y': work['P'],
            'xlabel': 'sqrt(t)',
            'fit': sqrt_fit,
        }

    duration = max(float(pump_duration), 1.0)
    work['G'] = compute_g_function(work['dt'], duration)
    g_fit = detect_g_function_closure(work['G'], work['P'])
    if g_fit:
        methods['g'] = {
            'name': 'G 函数法',
            'pressure': g_fit['p'],
            'x': g_fit['x'],
            'series_x': work['G'],
            'series_y': work['P'],
            'xlabel': 'G',
            'fit': g_fit,
        }

    if len(work) >= 9:
        pressure = work['P'].to_numpy(dtype=float)
        time = work['dt'].to_numpy(dtype=float)
        work['flowback_volume'] = cumulative_flowback_volume(time, work['Q'])
        stiffness_fit = detect_system_stiffness_closure(work['flowback_volume'], pressure)
        series_x = work['flowback_volume'].to_numpy(dtype=float)
        xlabel = 'Flowback volume'
        if not stiffness_fit:
            pressure_derivative = -np.gradient(pressure, time)
            series_x = pd.Series(pressure_derivative).rolling(5, center=True, min_periods=1).median().to_numpy()
            stiffness_fit = piecewise_linear_closure(series_x, pressure)
            if stiffness_fit:
                stiffness_fit = {**stiffness_fit, 'source': 'pressure_derivative_fallback'}
                xlabel = '-dP/dt'
        if stiffness_fit:
            methods['stiffness'] = {
                'name': '系统刚度法',
                'pressure': stiffness_fit['p'],
                'x': stiffness_fit['x'],
                'series_x': series_x,
                'series_y': pressure,
                'xlabel': xlabel,
                'fit': stiffness_fit,
            }

    return methods


def mean_std(values):
    """Return sample mean and sample standard deviation for report rows."""
    series = pd.Series(values, dtype='float64').dropna()
    if series.empty:
        return np.nan, np.nan
    std_value = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return float(series.mean()), std_value


def add_report_stat_rows(df, columns):
    """Append average and standard-deviation rows like the engineering report."""
    average_row = {'周期 #': '平均值'}
    std_row = {'周期 #': '标准方差'}

    for col in columns:
        avg, std = mean_std(df[col])
        average_row[col] = avg
        std_row[col] = std

    return pd.concat([df, pd.DataFrame([average_row, std_row])], ignore_index=True)


def add_report_stat_error_rows(df, columns, label_col):
    """Append average, standard deviation, and relative error rows."""
    average_row = {label_col: '平均值'}
    std_row = {label_col: '标准方差'}
    rel_row = {label_col: '相对误差'}

    for col in columns:
        avg, std = mean_std(df[col])
        average_row[col] = avg
        std_row[col] = std
        rel_row[col] = std / avg if np.isfinite(avg) and abs(avg) > 1e-9 else np.nan

    return pd.concat([df, pd.DataFrame([average_row, std_row, rel_row])], ignore_index=True)


def fmt_pressure(value):
    if pd.isna(value):
        return ''
    if isinstance(value, str):
        return value
    return f"{float(value):.3f}"


def fmt_percent(value):
    if pd.isna(value):
        return ''
    if isinstance(value, str):
        return value
    return f"{float(value) * 100:.2f}%"


def fmt_depth(value):
    return f"{float(value):.2f}"


def hydrostatic_pressure(fluid_density, depth_m, gravity=9.81):
    """
    Estimate pore pressure from a hydrostatic fluid column.

    Parameters
    ----------
    fluid_density : float
        Formation fluid density in g/cm3.
    depth_m : float
        Representative reservoir depth in m.
    gravity : float
        Gravitational acceleration in m/s2.

    Returns
    -------
    float
        Pore pressure in MPa.
    """
    return 1e-3 * float(fluid_density) * float(gravity) * float(depth_m)


def standpipe_pore_pressure(standpipe_pressure, mud_density, depth_m, gravity=9.81):
    """Estimate pore pressure from shut-in standpipe pressure plus mud column."""
    mud_column = hydrostatic_pressure(mud_density, depth_m, gravity)
    return float(standpipe_pressure) + mud_column


def regression_pore_pressure(depth_m, slope, intercept):
    """Estimate pore pressure from a local linear pressure-depth relation."""
    return float(slope) * float(depth_m) + float(intercept)


def apply_bottomhole_pressure_compensation(df, method, constant_loss=0.0, q2_coeff=0.0):
    """
    Deduct estimated tubing/friction loss from measured pressure.

    The corrected pressure is used for stress interpretation. The original
    pressure is preserved in P_raw for plotting and audit.
    """
    corrected = df.copy()
    corrected['P_raw'] = corrected['P']

    if method == "不校正":
        corrected['friction_loss'] = 0.0
    elif method == "常数摩阻扣除":
        corrected['friction_loss'] = float(constant_loss)
    else:
        q_abs = np.abs(corrected['Q'].to_numpy(dtype=float))
        corrected['friction_loss'] = float(q2_coeff) * q_abs ** 2

    corrected['friction_loss'] = np.maximum(corrected['friction_loss'], 0.0)
    corrected['P'] = corrected['P_raw'] - corrected['friction_loss']
    return corrected


def add_hydrostatic_column(df, fluid_density, depth_m, gravity=9.81):
    """Attach hydrostatic column and bottom-hole interpreted pressure columns."""
    out = df.copy()
    ph = hydrostatic_pressure(fluid_density, depth_m, gravity)
    out['PH'] = ph
    out['P_wh'] = out['P']  # wellhead pressure after friction compensation
    out['P_bh'] = out['P_wh'] + ph
    return out


def style_report_table(df):
    pressure_cols = [col for col in df.columns if col != '周期 #']
    return df.style.format({col: fmt_pressure for col in pressure_cols})


def analysis_signature(values):
    """Stable signature used to decide whether the user confirmed current choices."""
    return tuple(values)


def cumulative_injection_volume(time_s, flow_rate, unit='m3/min'):
    """Integrate flow rate to cumulative injection volume in liters."""
    time_s = np.asarray(time_s, dtype=float)
    flow_rate = np.asarray(flow_rate, dtype=float)
    if len(time_s) < 2:
        return np.zeros_like(time_s)

    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.maximum(dt, 0)
    volume = np.cumsum(flow_rate * dt / 60.0)
    if unit == 'm3/min':
        volume = volume * 1000.0
    return volume


def cumulative_flowback_volume(time_s, flow_rate):
    """
    Integrate post shut-in flow rate to relative flowback volume.

    Only the monotonic volume trend is required for stiffness interpretation, so
    the native rate scale is preserved and does not affect the picked pressure.
    """
    time_s = np.asarray(time_s, dtype=float)
    flow_rate = np.asarray(flow_rate, dtype=float)
    if len(time_s) < 2:
        return np.zeros_like(time_s)

    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.maximum(dt, 0.0)
    signed_volume = np.cumsum(flow_rate * dt)
    span = float(np.nanmax(signed_volume) - np.nanmin(signed_volume))
    abs_volume = np.cumsum(np.abs(flow_rate) * dt)

    if span > max(float(np.nanmax(abs_volume)) * 0.05, 1e-9):
        volume = signed_volume - signed_volume[0]
        if volume[-1] < 0:
            volume = -volume
        return volume

    return abs_volume - abs_volume[0]


def detect_system_stiffness_closure(flowback_volume, pressure, min_side=5):
    """
    Pick closure from the slope break on pressure versus flowback volume.

    Before closure, the system compliance includes wellbore plus open fracture.
    After closure, the fracture compliance is removed, so |dP/dV| generally
    increases. The closure is interpreted as the intersection of these trends.
    """
    volume = np.asarray(flowback_volume, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    valid = np.isfinite(volume) & np.isfinite(pressure)
    volume = volume[valid]
    pressure = pressure[valid]

    if len(volume) > 450:
        idx = np.unique(np.linspace(0, len(volume) - 1, 450).astype(int))
        volume = volume[idx]
        pressure = pressure[idx]

    if len(volume) < min_side * 2 + 1 or np.nanmax(volume) == np.nanmin(volume):
        return None

    pressure_smooth = pd.Series(pressure).rolling(5, center=True, min_periods=1).median().to_numpy()
    best = None
    fallback = None

    for split in range(min_side, len(volume) - min_side):
        left_coef = np.polyfit(volume[:split], pressure_smooth[:split], 1)
        right_coef = np.polyfit(volume[split:], pressure_smooth[split:], 1)
        left_fit = np.polyval(left_coef, volume[:split])
        right_fit = np.polyval(right_coef, volume[split:])
        sse = float(
            np.sum((pressure_smooth[:split] - left_fit) ** 2)
            + np.sum((pressure_smooth[split:] - right_fit) ** 2)
        )
        left_r2 = fit_quality_from_coef(volume[:split], pressure_smooth[:split], left_coef)
        right_r2 = fit_quality_from_coef(volume[split:], pressure_smooth[split:], right_coef)
        slope_ratio = abs(right_coef[0]) / max(abs(left_coef[0]), 1e-9)
        contrast = abs(right_coef[0] - left_coef[0])
        candidate = {
            'idx': split,
            'x': float(volume[split]),
            'p': float(pressure[split]),
            'coef': left_coef,
            'right_coef': right_coef,
            'line_start': 0,
            'line_end': split,
            'right_start': split,
            'right_end': len(volume),
            'r2': float((left_r2 + right_r2) / 2.0),
            'left_r2': float(left_r2),
            'right_r2': float(right_r2),
            'sse': sse,
            'source': 'flowback_volume_stiffness',
        }
        if fallback is None or sse < fallback['sse']:
            fallback = candidate
        if left_coef[0] < 0 and right_coef[0] < 0 and slope_ratio > 1.15:
            score = candidate['r2'] + min(contrast / max(abs(left_coef[0]), 1e-9), 3.0) * 0.05
            candidate['score'] = float(score)
            if best is None or score > best['score']:
                best = candidate

    fit = best or fallback
    if not fit:
        return None

    x_cross = line_intersection_x(fit['coef'], fit['right_coef'])
    if x_cross is not None and np.nanmin(volume) <= x_cross <= np.nanmax(volume):
        fit['x'] = float(x_cross)
        fit['p'] = float(np.polyval(fit['coef'], x_cross))
        fit['idx'] = int(np.nanargmin(np.abs(volume - x_cross)))

    return fit


def method_pressure(result, key):
    return result['methods'].get(key, {}).get('pressure', np.nan)


def correction_key(cycle_no, name):
    return f"cycle_{cycle_no}_{name}_override"


def quick_adjust_key(cycle_no, name):
    return f"cycle_{cycle_no}_{name}_quick_adjust"


def commit_quick_adjustment(cycle_no, name):
    """Commit a chart-side pressure calibration to the cycle override value."""
    quick_key = quick_adjust_key(cycle_no, name)
    value = float(st.session_state.get(quick_key, 0.0) or 0.0)
    committed_key = correction_key(cycle_no, name)
    st.session_state[committed_key] = value
    st.session_state[f"{committed_key}_pending"] = value


def corrected_value(cycle_no, name, default_value):
    value = st.session_state.get(correction_key(cycle_no, name), 0.0)
    if value and value > 0:
        return float(value)
    return float(default_value) if np.isfinite(default_value) else np.nan


def setup_report_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color='#b0b0b0', linewidth=0.7)


def draw_report_header(fig, well_name, depth, layer_name, cycle_label):
    header = f"客户：{well_name}\n深度：{depth:.1f}m\n井号：{layer_name}\n周期：{cycle_label}"
    fig.text(0.03, 0.97, header, ha='left', va='top', fontsize=9, fontweight='bold')
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.86, 0.86], color='black', linewidth=0.8, transform=fig.transFigure))


def pressure_at_volume(cycle_df, volume_l, pressure):
    idx = int(np.nanargmin(np.abs(cycle_df['V'] - volume_l)))
    return float(cycle_df.iloc[idx]['P']) if not np.isfinite(pressure) else pressure


def build_cycle_plot_data(df, cycle, flow_unit):
    cycle_df = df.loc[cycle['pump_start_idx']:cycle['end_idx']].copy()
    cycle_df['T_rel'] = cycle_df['T'] - float(cycle_df['T'].iloc[0])
    cycle_df['dt_shut'] = cycle_df['T'] - cycle['shut_t']
    cycle_df['V'] = cumulative_injection_volume(cycle_df['T'], cycle_df['Q'], flow_unit)
    cycle_df['V_rel'] = cycle_df['V'] - float(cycle_df['V'].min())

    decline_df = cycle_df[cycle_df['dt_shut'] > 0].copy()
    decline_df['sqrt_dt'] = np.sqrt(decline_df['dt_shut'])
    decline_df['log_dt'] = np.log10(decline_df['dt_shut'].clip(lower=1e-6))
    decline_df['flowback_volume'] = cumulative_flowback_volume(decline_df['dt_shut'], decline_df['Q'])
    duration = max(cycle['shut_t'] - float(df.loc[cycle['pump_start_idx'], 'T']), 1.0)
    decline_df['G'] = compute_g_function(decline_df['dt_shut'], duration)
    return cycle_df, decline_df


def plot_injection_rate(cycle_df, well_name, depth, layer_name, cycle_label):
    fig, ax1 = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    ax1.plot(cycle_df['T_rel'], cycle_df['P'], color='blue', linewidth=1.2)
    setup_report_axes(ax1, "井底压力和注入率", "相对时间(s)", "井底压力(MPa)")
    ax2 = ax1.twinx()
    ax2.plot(cycle_df['T_rel'], cycle_df['Q'], color='red', linestyle='--', linewidth=1.0)
    ax2.set_ylabel("注入率")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def plot_pv(cycle_df, pr_pressure, well_name, depth, layer_name, cycle_label):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    ax.plot(cycle_df['V_rel'], cycle_df['P'], color='blue', linewidth=1.2)
    idx = int(np.nanargmin(np.abs(cycle_df['P'] - pr_pressure)))
    peak_idx = int(cycle_df['P'].idxmax())
    rising = cycle_df.iloc[:peak_idx + 1].copy()
    if len(rising) >= 9:
        fit = detect_slope_change_pressure(rising['V_rel'], rising['P'])
        if fit and fit['left_coef'] is not None and fit['right_coef'] is not None:
            split = int(fit['idx'])
            left_x = np.linspace(float(rising['V_rel'].iloc[0]), float(rising['V_rel'].iloc[split]), 30)
            right_x = np.linspace(float(rising['V_rel'].iloc[split]), float(rising['V_rel'].iloc[-1]), 30)
            ax.plot(left_x, np.polyval(fit['left_coef'], left_x), color='green', linestyle='--', linewidth=1.0)
            ax.plot(right_x, np.polyval(fit['right_coef'], right_x), color='green', linestyle='--', linewidth=1.0)
    ax.scatter(cycle_df.iloc[idx]['V_rel'], pr_pressure, color='red', s=24, zorder=4)
    ax.annotate(f"重张压力: {pr_pressure:.3f} MPa", xy=(cycle_df.iloc[idx]['V_rel'], pr_pressure),
                xytext=(8, 8), textcoords='offset points', color='#1f2a5a', fontsize=9)
    setup_report_axes(ax, "P-V", "注入体积(L)", "井底压力(MPa)")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def plot_pdt(decline_df, isip_pressure, well_name, depth, layer_name, cycle_label):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    ax.plot(decline_df['dt_shut'], decline_df['P'], color='blue', linewidth=1.2)
    if len(decline_df) >= 4:
        fit_count = max(3, min(12, len(decline_df) // 5))
        coef = np.polyfit(decline_df['dt_shut'].iloc[:fit_count], decline_df['P'].iloc[:fit_count], 1)
        xfit = np.linspace(0, float(decline_df['dt_shut'].quantile(0.35)), 30)
        ax.plot(xfit, np.polyval(coef, xfit), color='green', linestyle='--', linewidth=1.0)
    ax.scatter(0, isip_pressure, color='red', s=24, zorder=4)
    ax.annotate(f"瞬时关井压力: {isip_pressure:.3f} MPa", xy=(0, isip_pressure),
                xytext=(8, 8), textcoords='offset points', color='#1f2a5a', fontsize=9)
    setup_report_axes(ax, "P-Δt", "相对时间(s)", "井底压力(MPa)")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def plot_loglog(decline_df, well_name, depth, layer_name, cycle_label):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    y = np.log10(np.maximum(decline_df['P'].iloc[0] - decline_df['P'], 1e-6))
    x = decline_df['log_dt']
    ax.plot(x, y, color='blue', linewidth=1.2)
    for i, frac in enumerate([0.45, 0.8], start=1):
        end = max(4, int(len(x) * frac))
        start = max(0, end - max(4, len(x) // 3))
        if end - start >= 3:
            coef = np.polyfit(x.iloc[start:end], y.iloc[start:end], 1)
            xfit = np.linspace(float(x.iloc[start]), float(x.iloc[end - 1]), 30)
            ax.plot(xfit, np.polyval(coef, xfit), color='green', linestyle='--', linewidth=1.0)
            ax.text(0.05, 0.92 - i * 0.08, f"{i}st 斜率: {coef[0]:.2f}", transform=ax.transAxes,
                    color='#1f2a5a', fontsize=9)
    setup_report_axes(ax, "lg(ΔP)-lg(Δt)", "lg(Δt)", "lg(井底压力)")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def plot_sqrt(decline_df, closure_pressure, well_name, depth, layer_name, cycle_label):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    x = decline_df['sqrt_dt']
    y = decline_df['P']
    ax.plot(x, y, color='blue', linewidth=1.2)
    fit = detect_sqrt_closure_with_derivative(decline_df['dt_shut'], x, y)
    if not np.isfinite(closure_pressure) and fit:
        closure_pressure = fit.get('p', np.nan)
    if not np.isfinite(closure_pressure):
        closure_pressure = float(y.median())
    if fit and fit.get('coef') is not None:
        line_start = fit.get('line_start', 0)
        line_end = min(fit.get('line_end', len(x) - 1), len(x) - 1)
        xfit = np.linspace(float(x.iloc[line_start]), float(x.iloc[min(len(x) - 1, max(line_end, line_start + 1))]), 40)
        ax.plot(xfit, np.polyval(fit['coef'], xfit), color='green', linestyle='--', linewidth=1.0)

        right_coef = fit.get('right_coef')
        split = int(np.nanargmin(np.abs(y - closure_pressure)))
        if right_coef is not None:
            right_end = min(len(x) - 1, split + max(4, len(x) // 5))
            xfit2 = np.linspace(float(x.iloc[split]), float(x.iloc[right_end]), 40)
            ax.plot(xfit2, np.polyval(right_coef, xfit2), color='green', linestyle='--', linewidth=1.0)

    idx = int(np.nanargmin(np.abs(y - closure_pressure)))
    ax.scatter(x.iloc[idx], closure_pressure, color='red', s=24, zorder=4)
    ax.annotate(f"闭合压力: {closure_pressure:.3f} MPa", xy=(x.iloc[idx], closure_pressure),
                xytext=(8, 8), textcoords='offset points', color='#1f2a5a', fontsize=9)
    setup_report_axes(ax, "平方根曲线", "Sqrt(Δt)", "井底压力(MPa)")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def plot_stiffness(decline_df, closure_pressure, well_name, depth, layer_name, cycle_label):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    if len(decline_df) < 3:
        closure_pressure = float(decline_df['P'].median()) if len(decline_df) else 0.0
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        draw_report_header(fig, well_name, depth, layer_name, cycle_label)
        setup_report_axes(ax, "系统刚度法", "回流体积", "井底压力(MPa)")
        return fig

    if 'flowback_volume' in decline_df:
        x = decline_df['flowback_volume'].to_numpy(dtype=float)
    else:
        x = cumulative_flowback_volume(decline_df['dt_shut'], decline_df['Q'])
    y = decline_df['P'].to_numpy(dtype=float)
    if not np.isfinite(closure_pressure):
        closure_pressure = float(np.nanmedian(y))
    ax.plot(x, y, color='blue', linewidth=1.2)
    fit = detect_system_stiffness_closure(x, y)
    if fit and fit.get('coef') is not None:
        split = fit.get('idx', int(len(x) / 2))
        left_end = max(1, min(split, len(x) - 1))
        xfit = np.linspace(float(x[0]), float(x[left_end]), 30)
        ax.plot(xfit, np.polyval(fit['coef'], xfit), color='green', linestyle='--', linewidth=1.0)
        right_coef = fit.get('right_coef')
        if right_coef is not None:
            xfit2 = np.linspace(float(x[left_end]), float(x[-1]), 30)
            ax.plot(xfit2, np.polyval(right_coef, xfit2), color='green', linestyle='--', linewidth=1.0)

    idx = int(np.nanargmin(np.abs(y - closure_pressure)))
    ax.scatter(x[idx], closure_pressure, color='red', s=24, zorder=4)
    ax.annotate(f"闭合压力: {closure_pressure:.3f} MPa", xy=(x[idx], closure_pressure),
                xytext=(8, -20), textcoords='offset points', color='#1f2a5a', fontsize=9)
    setup_report_axes(ax, "系统刚度法", "回流体积", "井底压力(MPa)")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def plot_derivative(decline_df, well_name, depth, layer_name, cycle_label):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    if len(decline_df) < 3:
        setup_report_axes(ax, "双对数压力导数", "lg(Δt)", "lg(Δt*dP/dΔt)")
        fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
        return fig

    x = decline_df['log_dt'].to_numpy(dtype=float)
    dt = decline_df['dt_shut'].to_numpy(dtype=float)
    pressure = decline_df['P'].to_numpy(dtype=float)
    pressure_drop = pressure[0] - pressure
    deriv = np.abs(np.gradient(pressure_drop, x))
    y = np.log10(np.maximum(deriv, 1e-6))
    ax.plot(x, y, color='blue', linewidth=1.0)
    segments = find_derivative_flow_segments(dt, pressure)
    if segments:
        segment_specs = [
            ('线性/双线性流', segments['early']),
            ('闭合后流态', segments['late']),
        ]
        for i, (label, segment) in enumerate(segment_specs, start=1):
            start = segment['start']
            end = segment['end']
            coef = segment['coef']
            xfit = np.linspace(float(x[start]), float(x[end - 1]), 30)
            ax.plot(xfit, np.polyval(coef, xfit), color='green', linestyle='--', linewidth=1.1)
            ax.text(
                0.05,
                0.92 - i * 0.08,
                f"{label}: 斜率 {segment['slope']:.2f}",
                transform=ax.transAxes,
                color='#1f2a5a',
                fontsize=9,
            )
    setup_report_axes(ax, "双对数压力导数", "lg(Δt)", "lg(Δt*dP/dΔt)")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def plot_g_function(decline_df, closure_pressure, well_name, depth, layer_name, cycle_label):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_report_header(fig, well_name, depth, layer_name, cycle_label)
    x = decline_df['G']
    y = decline_df['P']
    if not np.isfinite(closure_pressure):
        closure_pressure = float(y.median())
    ax.plot(x, y, color='blue', linewidth=1.0)
    fit = detect_g_function_closure(x, y)
    if fit and fit.get('coef') is not None:
        line_start = fit.get('line_start', 0)
        line_end = min(fit.get('line_end', len(x) - 1), len(x) - 1)
        end_idx = min(len(x) - 1, max(line_end, line_start + 1))
        xfit = np.linspace(float(x.iloc[line_start]), float(x.iloc[end_idx]), 30)
        ax.plot(xfit, np.polyval(fit['coef'], xfit), color='green', linestyle='--', linewidth=1.0)
        right_coef = fit.get('right_coef')
        split = int(np.nanargmin(np.abs(y - closure_pressure)))
        if right_coef is not None:
            right_end = min(len(x) - 1, split + max(4, len(x) // 5))
            xfit2 = np.linspace(float(x.iloc[split]), float(x.iloc[right_end]), 30)
            ax.plot(xfit2, np.polyval(right_coef, xfit2), color='green', linestyle='--', linewidth=1.0)

        derivative = fit.get('derivative')
        if derivative is not None and len(derivative) == len(x):
            ax2 = ax.twinx()
            ax2.plot(x, derivative, color='#9467bd', linewidth=0.8, alpha=0.65)
            baseline = fit.get('derivative_baseline')
            if baseline is not None and len(baseline) == len(x):
                ax2.plot(x, baseline, color='#9467bd', linestyle='--', linewidth=0.8, alpha=0.85)
            ax2.set_ylabel("G*(-dP/dG) (MPa)")
    idx = int(np.nanargmin(np.abs(y - closure_pressure)))
    ax.scatter(x.iloc[idx], closure_pressure, color='red', s=24, zorder=4)
    ax.annotate(f"闭合压力: {closure_pressure:.3f} MPa", xy=(x.iloc[idx], closure_pressure),
                xytext=(8, 8), textcoords='offset points', color='#1f2a5a', fontsize=9)
    setup_report_axes(ax, "G-函数", "G", "井底压力(MPa)")
    fig.tight_layout(rect=[0.12, 0.08, 0.92, 0.82])
    return fig


def show_independent_plot(title, plot_func, *args):
    try:
        fig = plot_func(*args)
        st.pyplot(fig)
    except Exception as exc:
        st.warning(f"{title} 暂未生成：{exc}。可先使用左侧手动修正参数。")


def detect_decline_cycles(df):
    """Split data from one pump start to the next pump start as one cycle."""
    q_limit = max(0.05, float(df['Q'].max()) * 0.03)
    pumping = df['Q'] > q_limit
    pump_start_indices = list(df.index[~pumping.shift(fill_value=False) & pumping])

    if not pump_start_indices:
        fallback = detect_pressure_points(df)
        return [{
            'cycle': 1,
            'start_idx': int((df['T'] - fallback['shut_t']).abs().idxmin()),
            'end_idx': int(df.index[-1]),
            'pump_start_idx': int(df.index[0]),
            'shut_t': fallback['shut_t'],
            'isip_p': fallback['isip_p'],
            'pb_t': fallback['pb_t'],
            'pb_p': fallback['pb_p'],
            'pr_t': fallback['pr_t'],
            'pr_p': fallback['pr_p'],
            'label': f"第1次压降：{fallback['shut_t']:.1f}s"
        }]

    cycles = []
    for start_pos, pump_start_idx in enumerate(pump_start_indices):
        next_start_idx = pump_start_indices[start_pos + 1] if start_pos + 1 < len(pump_start_indices) else int(df.index[-1] + 1)
        end_idx = int(min(next_start_idx - 1, df.index[-1]))

        if end_idx - pump_start_idx < 6:
            continue

        cycle_window = df.loc[pump_start_idx:end_idx].copy()
        local_pumping = pumping.loc[pump_start_idx:end_idx]
        shut_candidates = list(cycle_window.index[local_pumping.shift(fill_value=True) & ~local_pumping])
        start_idx = int(shut_candidates[0]) if shut_candidates else int(cycle_window['P'].idxmax())

        if end_idx - start_idx < 5:
            start_idx = int(max(pump_start_idx, end_idx - 5))

        decline_window = df.loc[start_idx:end_idx].copy()
        pb_idx = cycle_window['P'].idxmax()
        pr_t, pr_p = detect_reopening_pressure(df, int(pump_start_idx), int(start_idx))

        shut_t = float(df.loc[start_idx, 'T'])
        cycles.append({
            'cycle': len(cycles) + 1,
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'pump_start_idx': int(pump_start_idx),
            'shut_t': shut_t,
            'isip_p': float(df.loc[start_idx, 'P']),
            'pb_t': float(df.loc[pb_idx, 'T']),
            'pb_p': float(df.loc[pb_idx, 'P']),
            'pr_t': pr_t,
            'pr_p': pr_p,
            'label': f"第{len(cycles) + 1}周期：{float(cycle_window['T'].iloc[0]):.1f}s - {float(cycle_window['T'].iloc[-1]):.1f}s"
        })

    if not cycles:
        fallback = detect_pressure_points(df)
        return [{
            'cycle': 1,
            'start_idx': int((df['T'] - fallback['shut_t']).abs().idxmin()),
            'end_idx': int(df.index[-1]),
            'pump_start_idx': int(df.index[0]),
            'shut_t': fallback['shut_t'],
            'isip_p': fallback['isip_p'],
            'pb_t': fallback['pb_t'],
            'pb_p': fallback['pb_p'],
            'pr_t': fallback['pr_t'],
            'pr_p': fallback['pr_p'],
            'label': f"第1周期：{float(df['T'].iloc[0]):.1f}s - {float(df['T'].iloc[-1]):.1f}s"
        }]

    return cycles


def analyze_decline_cycle(df, cycle):
    """Run three-method closure analysis for one shut-in decline cycle."""
    analysis_df = df.loc[cycle['start_idx']:cycle['end_idx']].copy()
    analysis_df['dt'] = analysis_df['T'] - cycle['shut_t']
    analysis_df['sqrt_dt'] = np.sqrt(analysis_df['dt'])
    pump_duration = max(cycle['shut_t'] - float(df.loc[cycle['pump_start_idx'], 'T']), 1.0)
    try:
        methods = build_analysis_methods(analysis_df, pump_duration)
        error = ''
    except Exception as exc:
        methods = {}
        error = str(exc)
    values = [item['pressure'] for item in methods.values()]
    closure_avg = float(np.mean(values)) if values else np.nan

    return {
        'cycle': cycle,
        'analysis_df': analysis_df,
        'methods': methods,
        'closure_avg': closure_avg,
        'method_count': len(values),
        'error': error,
    }


# --- 环境设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="工程级地应力综合分析系统", layout="wide")

st.title("🏗️ 地应力数据智能分析专家系统 V3.5")

# --- 1. 增强版数据导入模块 ---
st.sidebar.header("📂 数据源配置")
uploaded_file = st.sidebar.file_uploader("上传施工文件 (csv, xlsx, txt)", type=["csv", "xlsx", "txt"])

if uploaded_file:
    try:
        # 自动识别分隔符读取
        if uploaded_file.name.endswith('csv'):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('xlsx'):
            if importlib.util.find_spec('openpyxl') is None:
                st.error("读取 Excel 文件需要安装 openpyxl。请在命令行运行：python -m pip install openpyxl")
                st.stop()
            df_raw = pd.read_excel(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # 修复空列名问题（针对你CSV最后的那个逗号）
        new_cols = []
        for i, col in enumerate(df_raw.columns):
            if "Unnamed" in str(col):
                new_cols.append(f"列_{i}(数据)")
            else:
                new_cols.append(col)
        df_raw.columns = new_cols

        # 列映射设置
        st.sidebar.subheader("📍 字段校准")
        all_cols = df_raw.columns.tolist()
        
        sel_time = st.sidebar.selectbox("时间轴 (Time)", all_cols, index=0)
        sel_pres = st.sidebar.selectbox("压力轴 (Pressure)", all_cols, index=1 if len(all_cols)>1 else 0)
        sel_flow = st.sidebar.selectbox("排量轴 (Flow)", all_cols, index=2 if len(all_cols)>2 else 0)

        # 核心清洗：强制转数字
        df = pd.DataFrame({
            'T': pd.to_numeric(df_raw[sel_time], errors='coerce'),
            'P': pd.to_numeric(df_raw[sel_pres], errors='coerce'),
            'Q': pd.to_numeric(df_raw[sel_flow], errors='coerce')
        }).dropna().sort_values('T').drop_duplicates('T').reset_index(drop=True)

    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.stop()
else:
    st.info("👋 欢迎！请在左侧上传你的数据文件。系统将自动提取压力和排量信息。")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🧯 井底压力补偿")
friction_method = st.sidebar.selectbox(
    "摩阻/管柱压降扣除",
    ["不校正", "常数摩阻扣除", "排量平方摩阻扣除"],
    index=0,
)
constant_friction_loss = st.sidebar.number_input("常数摩阻 ΔPf (MPa)", min_value=0.0, value=0.0, step=0.1)
q2_friction_coeff = st.sidebar.number_input(
    "排量平方系数 k (MPa/(排量单位)^2)",
    min_value=0.0,
    value=0.0,
    step=0.01,
    format="%.4f",
)
df = apply_bottomhole_pressure_compensation(
    df,
    friction_method,
    constant_loss=constant_friction_loss,
    q2_coeff=q2_friction_coeff,
)

# --- 2. 井深与多次压降设置 ---
cycles = detect_decline_cycles(df)
cycle_map = {item['label']: item for item in cycles}
data_signature = analysis_signature([
    uploaded_file.name,
    sel_time,
    sel_pres,
    sel_flow,
    friction_method,
    constant_friction_loss,
    q2_friction_coeff,
])

if st.session_state.get('active_data_signature') != data_signature:
    st.session_state['active_data_signature'] = data_signature
    st.session_state['analysis_started'] = False

st.sidebar.markdown("---")
st.sidebar.header("🧭 井深 / 层位参数")
well_name = st.sidebar.text_input("井号", value="未命名井")
layer_name = st.sidebar.text_input("测试层位", value="第1层")
test_depth = st.sidebar.number_input("测试井深 (m)", min_value=0.0, value=0.0, step=0.1)
test_depth_bottom = st.sidebar.number_input("测试段底深 (m, 0=同井深)", min_value=0.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("🔁 多次压降选择")
selected_labels = st.sidebar.multiselect(
    "参与本层平均的压降段",
    options=list(cycle_map.keys()),
    default=list(cycle_map.keys()),
)
detail_options = selected_labels if selected_labels else list(cycle_map.keys())
detail_label = st.sidebar.selectbox("查看单次压降细节", detail_options, index=0)

st.sidebar.markdown("---")
st.sidebar.header("🧮 计算参数")
flow_unit = st.sidebar.selectbox("排量单位", ["m3/min", "L/min"], index=0)
pore_method = st.sidebar.selectbox(
    "孔隙压力 Pp 估算方法",
    ["静水压力法", "关井立管压力法", "区域经验公式法", "手动/DFIT解释值"],
    index=0,
)
fluid_density = st.sidebar.number_input("地层流体密度 ρf (g/cm³)", min_value=0.0, value=1.03, step=0.01)
mud_density = st.sidebar.number_input("钻井液密度 ρm (g/cm³)", min_value=0.0, value=1.20, step=0.01)
standpipe_pressure = st.sidebar.number_input("关井立管压力 Ps (MPa)", min_value=0.0, value=0.0, step=0.1)
regression_slope = st.sidebar.number_input("经验公式斜率 a (MPa/m)", value=0.00981, step=0.0001, format="%.5f")
regression_intercept = st.sidebar.number_input("经验公式截距 b (MPa)", value=0.0, step=0.1)
manual_pore_pressure = st.sidebar.number_input("手动/DFIT孔隙压力 (MPa)", min_value=0.0, value=25.0, step=0.1)
pore_pressure_override = st.sidebar.number_input("最终 Pp 修正值 (0=采用上方估算)", min_value=0.0, value=0.0, step=0.1)
t_str = st.sidebar.number_input("岩石抗张强度 T (MPa)", value=1.5)
vertical_gradient = st.sidebar.number_input("垂向地应力梯度 (kPa/m)", min_value=0.0, value=23.0, step=0.1)
vertical_stress_override = st.sidebar.number_input("垂向地应力修正值 (0=按梯度计算)", min_value=0.0, value=0.0, step=0.1)
sigma_h_method = st.sidebar.selectbox(
    "最大水平主应力 σH 计算方法",
    ["重张压力 Pr 法（推荐）", "破裂压力 Pb 法"],
    index=0,
)
final_sh_override = st.sidebar.number_input("本层最终 σh 修正值 (0=采用自动平均)", min_value=0.0, value=0.0, step=0.1)

submitted = st.sidebar.button("确定并开始分析", use_container_width=True)

if submitted:
    st.session_state['analysis_started'] = True
    st.rerun()

if not st.session_state.get('analysis_started', False):
    st.info("请在左侧完成井深、压降段和计算参数选择，然后点击“确定并开始分析”。")
    st.stop()

st.sidebar.success("已开始分析")

if not selected_labels:
    st.warning("请在左侧至少选择一次压降段参与计算。")
    st.stop()

selected_cycles = [cycle_map[label] for label in selected_labels]
cycle_results = [analyze_decline_cycle(df, cycle) for cycle in selected_cycles]
detail_result = next(item for item in cycle_results if item['cycle']['label'] == detail_label)
detail_cycle = detail_result['cycle']
failed_cycles = [item for item in cycle_results if item.get('error')]
if failed_cycles:
    st.warning("部分周期自动算法失败，已跳过自动值，可使用左侧手动修正：" + "；".join(
        f"C{item['cycle']['cycle']}: {item['error']}" for item in failed_cycles
    ))

st.sidebar.markdown("---")
with st.sidebar.form(f"correction_form_{detail_cycle['cycle']}"):
    st.header("✍️ 当前周期手动修正")
    st.caption("填 0 表示采用自动识别值；点击确定后才更新统计表和图件。")

    pending_map = {}
    labels = {
        'pr': "重张压力 Pr 修正 (MPa)",
        'isip': "瞬时关井压力 ISIP 修正 (MPa)",
        'sqrt': "平方根法闭合压力修正 (MPa)",
        'stiffness': "系统刚度法闭合压力修正 (MPa)",
        'g': "G 函数法闭合压力修正 (MPa)",
    }
    for name, label in labels.items():
        committed_key = correction_key(detail_cycle['cycle'], name)
        pending_key = f"{committed_key}_pending"
        if committed_key not in st.session_state:
            st.session_state[committed_key] = 0.0
        if pending_key not in st.session_state:
            st.session_state[pending_key] = float(st.session_state[committed_key])
        pending_map[name] = pending_key
        st.number_input(label, min_value=0.0, step=0.1, key=pending_key)

    correction_submitted = st.form_submit_button("确定并修正", use_container_width=True)

if correction_submitted:
    for name, pending_key in pending_map.items():
        st.session_state[correction_key(detail_cycle['cycle'], name)] = float(st.session_state[pending_key])
    st.rerun()

corrected_cycle_values = []
for result in cycle_results:
    cycle_no = result['cycle']['cycle']
    sqrt_value = corrected_value(cycle_no, 'sqrt', method_pressure(result, 'sqrt'))
    stiffness_value = corrected_value(cycle_no, 'stiffness', method_pressure(result, 'stiffness'))
    g_value = corrected_value(cycle_no, 'g', method_pressure(result, 'g'))
    cycle_avg, _ = mean_std([sqrt_value, stiffness_value, g_value])
    corrected_cycle_values.append(cycle_avg)

valid_results = [value for value in corrected_cycle_values if np.isfinite(value)]

if valid_results:
    layer_auto_sh = float(np.mean(valid_results))
else:
    layer_auto_sh = float(np.mean([
        corrected_value(item['cycle']['cycle'], 'isip', item['cycle']['isip_p'])
        for item in cycle_results
    ]))

manual_sh = final_sh_override if final_sh_override > 0 else layer_auto_sh
depth_bottom = test_depth_bottom if test_depth_bottom > 0 else test_depth
depth_mid = (test_depth + depth_bottom) / 2.0 if depth_bottom > 0 else test_depth
depth_label = f"{test_depth:.0f}-{depth_bottom:.0f}" if depth_bottom > test_depth else fmt_depth(test_depth)

if pore_method == "静水压力法":
    pore_pressure_auto = hydrostatic_pressure(fluid_density, depth_mid)
elif pore_method == "关井立管压力法":
    pore_pressure_auto = standpipe_pore_pressure(standpipe_pressure, mud_density, depth_mid)
elif pore_method == "区域经验公式法":
    pore_pressure_auto = regression_pore_pressure(depth_mid, regression_slope, regression_intercept)
else:
    pore_pressure_auto = manual_pore_pressure

p_pore = pore_pressure_override if pore_pressure_override > 0 else pore_pressure_auto

# --- 3. 界面第一位：原始施工监控图 ---
st.subheader("📊 施工全过程监控 (压力-排量-时间)")
fig_main, ax1 = plt.subplots(figsize=(14, 5))

if 'P_raw' in df and np.nanmax(np.abs(df['P_raw'] - df['P'])) > 1e-9:
    ax1.plot(df['T'], df['P_raw'], color='#8fb9dd', linewidth=1.0, alpha=0.65, label='原始压力 (MPa)')
line_p, = ax1.plot(df['T'], df['P'], color='#1f77b4', linewidth=1.5, label='井底修正压力 (MPa)')
ax1.set_xlabel("时间 (s)", fontsize=10)
ax1.set_ylabel("压力 (MPa)", color='#1f77b4', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.grid(True, linestyle=':', alpha=0.6)

for result in cycle_results:
    cycle = result['cycle']
    isip_plot = corrected_value(cycle['cycle'], 'isip', cycle['isip_p'])
    ax1.axvspan(cycle['shut_t'], float(df.loc[cycle['end_idx'], 'T']), color='#2ca02c', alpha=0.08)
    key_points = [
        (f"Pb{cycle['cycle']}", cycle['pb_t'], cycle['pb_p'], '#d62728'),
        (f"ISIP{cycle['cycle']}", cycle['shut_t'], isip_plot, '#ff7f0e'),
    ]
    for label, t_value, p_value, color in key_points:
        ax1.scatter(t_value, p_value, color=color, s=35, zorder=5)
        ax1.annotate(
            f"{label}: {p_value:.2f}",
            xy=(t_value, p_value),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=8,
            color=color,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.72),
        )

detail_cycle = detail_result['cycle']
if np.isfinite(manual_sh):
    ax1.axhline(manual_sh, color='#2ca02c', linestyle='--', linewidth=1.1, alpha=0.8)
    ax1.annotate(
        f"本层σh: {manual_sh:.2f} MPa",
        xy=(detail_cycle['shut_t'], manual_sh),
        xytext=(8, 8),
        textcoords='offset points',
        fontsize=9,
        color='#2ca02c',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2ca02c', alpha=0.75),
    )

ax2 = ax1.twinx()
ax2.fill_between(df['T'], df['Q'], color='red', alpha=0.15, label='排量')
ax2.set_ylabel("排量 (m³/min)", color='red', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red')

ax1.legend(loc='upper left')
st.pyplot(fig_main)

# --- 4. 多次压降汇总计算 ---
p_break = float(max(item['cycle']['pb_p'] for item in cycle_results))
avg_pr = float(np.mean([
    corrected_value(item['cycle']['cycle'], 'pr', item['cycle']['pr_p'])
    for item in cycle_results
]))
avg_isip = float(np.mean([
    corrected_value(item['cycle']['cycle'], 'isip', item['cycle']['isip_p'])
    for item in cycle_results
]))
avg_sh = manual_sh
sigma_H_from_pr = 3 * avg_sh - avg_pr - p_pore
sigma_H_from_pb = 3 * avg_sh - p_break + t_str - p_pore
sigma_H = sigma_H_from_pr if sigma_h_method.startswith("重张压力") else sigma_H_from_pb
sigma_H_is_suspect = np.isfinite(sigma_H) and np.isfinite(avg_sh) and sigma_H < avg_sh
sigma_v = vertical_stress_override if vertical_stress_override > 0 else vertical_gradient * depth_mid / 1000.0
sv_gradient = sigma_v / depth_mid * 1000.0 if depth_mid > 0 else np.nan
sh_gradient = avg_sh / depth_mid * 1000.0 if depth_mid > 0 else np.nan
sH_gradient = sigma_H / depth_mid * 1000.0 if depth_mid > 0 else np.nan
mises_stress = np.sqrt(
    0.5 * (
        (sigma_v - sigma_H) ** 2
        + (sigma_H - avg_sh) ** 2
        + (avg_sh - sigma_v) ** 2
    )
)

summary_rows = []
for result in cycle_results:
    cycle = result['cycle']
    cycle_no = cycle['cycle']
    sqrt_value = corrected_value(cycle_no, 'sqrt', method_pressure(result, 'sqrt'))
    stiffness_value = corrected_value(cycle_no, 'stiffness', method_pressure(result, 'stiffness'))
    g_value = corrected_value(cycle_no, 'g', method_pressure(result, 'g'))
    cycle_avg, _ = mean_std([sqrt_value, stiffness_value, g_value])
    summary_rows.append({
        '周期 #': cycle['cycle'],
        '重张压力 (MPa)': corrected_value(cycle_no, 'pr', cycle['pr_p']),
        '瞬时关井压力 (MPa)': corrected_value(cycle_no, 'isip', cycle['isip_p']),
        '裂缝闭合压力，平方根 (MPa)': sqrt_value,
        '裂缝闭合压力，系统刚度 (MPa)': stiffness_value,
        '裂缝闭合压力，G 函数 (MPa)': g_value,
        '单周期平均闭合压力 (MPa)': cycle_avg,
    })

st.subheader("📋 特征压力统计表")
st.write(f"**井号：** {well_name}　**深度：** {fmt_depth(test_depth)} m　**地层：** {layer_name}　**周期：** 1 to {len(cycle_results)}")
report_df = pd.DataFrame(summary_rows)
pressure_columns = [
    '重张压力 (MPa)',
    '瞬时关井压力 (MPa)',
    '裂缝闭合压力，平方根 (MPa)',
    '裂缝闭合压力，系统刚度 (MPa)',
    '裂缝闭合压力，G 函数 (MPa)',
    '单周期平均闭合压力 (MPa)',
]
report_df = add_report_stat_rows(report_df, pressure_columns)
st.dataframe(style_report_table(report_df), use_container_width=True)

quality_rows = []
for result in cycle_results:
    cycle_no = result['cycle']['cycle']
    for key, method in result['methods'].items():
        fit = method.get('fit', {})
        quality_rows.append({
            '周期 #': cycle_no,
            '方法': method.get('name', key),
            '自动值 MPa': method.get('pressure', np.nan),
            '线性段 R²': fit.get('r2', np.nan),
            '识别来源': fit.get('source', 'piecewise'),
        })

if quality_rows:
    with st.expander("🔎 自动判读质量记录"):
        st.caption("R² 越接近 1，说明闭合前线性诊断段越稳定；识别来源为 fallback 或 piecewise 时，建议结合图件人工复核。")
        quality_df = pd.DataFrame(quality_rows)
        st.dataframe(
            quality_df.style.format({'自动值 MPa': fmt_pressure, '线性段 R²': fmt_pressure}),
            use_container_width=True,
        )

for row in summary_rows:
    method_values = [
        row['裂缝闭合压力，平方根 (MPa)'],
        row['裂缝闭合压力，系统刚度 (MPa)'],
        row['裂缝闭合压力，G 函数 (MPa)'],
    ]
    method_avg, method_std = mean_std(method_values)
    if np.isfinite(method_avg) and method_avg > 0 and method_std / method_avg > 0.10:
        st.warning(
            f"周期 {row['周期 #']} 三种闭合压力离散度较大"
            f"（标准差/均值={method_std / method_avg:.1%}），建议以图件人工判读值为准。"
        )

if sigma_H_is_suspect:
    st.error(
        f"当前计算得到 σH={sigma_H:.2f} MPa，小于 σh={avg_sh:.2f} MPa。"
        "这不符合最大/最小水平主应力命名关系，建议优先复核重张压力 Pr、孔隙压力 Pp、"
        "井底压力校正以及 σH 计算方法。"
    )

sqrt_avg, sqrt_std = mean_std(pd.DataFrame(summary_rows)['裂缝闭合压力，平方根 (MPa)'])
stiffness_avg, stiffness_std = mean_std(pd.DataFrame(summary_rows)['裂缝闭合压力，系统刚度 (MPa)'])
g_avg, g_std = mean_std(pd.DataFrame(summary_rows)['裂缝闭合压力，G 函数 (MPa)'])
cycle_avg, cycle_std = mean_std(pd.DataFrame(summary_rows)['单周期平均闭合压力 (MPa)'])

with st.expander("🧮 本表采用公式"):
    st.markdown(
        """
        - 单周期平均闭合压力：`Pc_i = (Pc_平方根,i + Pc_系统刚度,i + Pc_G函数,i) / 3`
        - 本层平均闭合压力：`Pc_avg = (Pc_1 + Pc_2 + ... + Pc_n) / n`
        - 标准方差：`s = sqrt(Σ(x_i - x_avg)^2 / (n - 1))`
        - 最大水平主应力（推荐）：`σH = 3σh - Pr - Pp`
        - 最大水平主应力（破裂压力对照）：`σH = 3σh - Pb + T - Pp`
        """
    )
    formula_df = pd.DataFrame([
        {'项目': '平方根法平均', '平均值 MPa': sqrt_avg, '标准方差 MPa': sqrt_std},
        {'项目': '系统刚度法平均', '平均值 MPa': stiffness_avg, '标准方差 MPa': stiffness_std},
        {'项目': 'G 函数法平均', '平均值 MPa': g_avg, '标准方差 MPa': g_std},
        {'项目': '单周期平均闭合压力', '平均值 MPa': cycle_avg, '标准方差 MPa': cycle_std},
    ])
    st.dataframe(formula_df.style.format({'平均值 MPa': '{:.3f}', '标准方差 MPa': '{:.3f}'}), use_container_width=True)

if valid_results:
    st.info(f"本层自动平均 σh = {layer_auto_sh:.2f} MPa；侧边栏可对最终采用值进行人工校准。")
else:
    st.warning("选中的压降段数据不足，三种闭合压力算法未能形成有效结果；当前最终值暂以 ISIP 平均值作为校准初值。")

st.subheader(f"📈 单次压降细节：{detail_label}")
detail_cycle_no = detail_cycle['cycle']
cycle_df, decline_df = build_cycle_plot_data(df, detail_cycle, flow_unit)
detail_pr = corrected_value(detail_cycle_no, 'pr', detail_cycle['pr_p'])
detail_isip = corrected_value(detail_cycle_no, 'isip', detail_cycle['isip_p'])
detail_sqrt = corrected_value(detail_cycle_no, 'sqrt', method_pressure(detail_result, 'sqrt'))
detail_stiffness = corrected_value(detail_cycle_no, 'stiffness', method_pressure(detail_result, 'stiffness'))
detail_g = corrected_value(detail_cycle_no, 'g', method_pressure(detail_result, 'g'))

if len(decline_df) >= 6:
    st.markdown("**图上快速校准**")
    st.caption("绿色线为自动拟合诊断线；下方数值控制红色闭合点和统计表采用值。填 0 表示恢复自动识别值。")
    quick_specs = [
        ('sqrt', '平方根法 Pc', detail_sqrt, method_pressure(detail_result, 'sqrt')),
        ('stiffness', '系统刚度 Pc', detail_stiffness, method_pressure(detail_result, 'stiffness')),
        ('g', 'G 函数 Pc', detail_g, method_pressure(detail_result, 'g')),
    ]
    quick_cols = st.columns(3)
    for col, (name, label, current_value, auto_value) in zip(quick_cols, quick_specs):
        quick_key = quick_adjust_key(detail_cycle_no, name)
        if quick_key not in st.session_state:
            committed = st.session_state.get(correction_key(detail_cycle_no, name), 0.0)
            st.session_state[quick_key] = float(committed or 0.0)
        with col:
            st.number_input(
                label,
                min_value=0.0,
                value=float(st.session_state[quick_key]),
                step=0.01,
                format="%.3f",
                key=quick_key,
                help=f"自动识别值：{auto_value:.3f} MPa；输入 0 恢复自动值。",
                on_change=commit_quick_adjustment,
                args=(detail_cycle_no, name),
            )

    detail_sqrt = corrected_value(detail_cycle_no, 'sqrt', method_pressure(detail_result, 'sqrt'))
    detail_stiffness = corrected_value(detail_cycle_no, 'stiffness', method_pressure(detail_result, 'stiffness'))
    detail_g = corrected_value(detail_cycle_no, 'g', method_pressure(detail_result, 'g'))

    st.markdown(
        """
        **本周期图件公式**
        `V=Σ(Q·Δt)`；`ISIP=P(Δt=0)`；`重张压力Pr` 取启泵升压段 P-V/P-t 曲线由线性增压转为斜率下降的突变点；
        平方根法使用 `P=a√Δt+b` 的闭合前线性段，曲线持续偏离该线性段的位置判为闭合压力；
        系统刚度法使用回流压力-回流体积关系，取闭合前后两段刚度变化直线的交点；
        G 函数法使用 Nolte G 时间和 `G*(-dP/dG)` 导数上翘点。
        """
    )
    chart_rows = [
        [
            ("井底压力和注入率", plot_injection_rate, (cycle_df, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
            ("P-V", plot_pv, (cycle_df, detail_pr, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
        ],
        [
            ("P-Δt", plot_pdt, (decline_df, detail_isip, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
            ("lg(ΔP)-lg(Δt)", plot_loglog, (decline_df, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
        ],
        [
            ("平方根曲线", plot_sqrt, (decline_df, detail_sqrt, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
            ("系统刚度法", plot_stiffness, (decline_df, detail_stiffness, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
        ],
        [
            ("双对数压力导数", plot_derivative, (decline_df, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
            ("G-函数", plot_g_function, (decline_df, detail_g, well_name, test_depth, layer_name, f"C{detail_cycle_no}")),
        ],
    ]

    for left_item, right_item in chart_rows:
        left, right = st.columns(2)
        with left:
            show_independent_plot(left_item[0], left_item[1], *left_item[2])
        with right:
            show_independent_plot(right_item[0], right_item[1], *right_item[2])
else:
    st.warning("该压降段数据点不足，无法生成完整判读图谱。")

with st.expander("📌 本层参数与关键点明细"):
    layer_info = pd.DataFrame([{
        '井号': well_name,
        '层位': layer_name,
        '测试段深度 m': depth_label,
        '代表深度 m': depth_mid,
        '垂向地应力 MPa': sigma_v,
        '本层最终 σh MPa': avg_sh,
        '最大 σH MPa': sigma_H,
        'σH校核': '需复核：σH<σh' if sigma_H_is_suspect else '通过',
        'σH_Pr法 MPa': sigma_H_from_pr,
        'σH_Pb法 MPa': sigma_H_from_pb,
        'Mises剪应力 MPa': mises_stress,
        '孔隙压力方法': pore_method,
        '孔隙压力 MPa': p_pore,
        '抗张强度 MPa': t_str,
        '压力补偿方法': friction_method,
        '常数摩阻 MPa': constant_friction_loss,
        '排量平方系数': q2_friction_coeff,
    }])
    st.dataframe(layer_info, use_container_width=True)

st.divider()
st.subheader("📋 分周期地应力解释结果表")
cycle_pressure_rows = []
for row in summary_rows:
    cycle_label = f"C{int(row['周期 #'])}" if isinstance(row['周期 #'], (int, np.integer)) else row['周期 #']
    closure_pressure = row['单周期平均闭合压力 (MPa)']
    cycle_pressure_rows.append({
        '测试次数': cycle_label,
        '裂缝重启压力 (MPa)': row['重张压力 (MPa)'],
        '瞬时关井压力 (MPa)': row['瞬时关井压力 (MPa)'],
        '裂缝闭合压力 (MPa)': closure_pressure,
        '最小水平主应力(MPa)': closure_pressure,
    })

cycle_pressure_df = pd.DataFrame(cycle_pressure_rows)
cycle_pressure_cols = [
    '裂缝重启压力 (MPa)',
    '瞬时关井压力 (MPa)',
    '裂缝闭合压力 (MPa)',
    '最小水平主应力(MPa)',
]
cycle_pressure_df = add_report_stat_error_rows(cycle_pressure_df, cycle_pressure_cols, '测试次数')
cycle_pressure_display = cycle_pressure_df.copy()
for col in cycle_pressure_cols:
    cycle_pressure_display[col] = [
        fmt_percent(value) if label == '相对误差' else fmt_pressure(value)
        for label, value in zip(cycle_pressure_display['测试次数'], cycle_pressure_display[col])
    ]
st.dataframe(cycle_pressure_display, use_container_width=True)

st.subheader("📋 本层地应力解释成果表")
stress_result_df = pd.DataFrame([{
    '测试段深度 m': depth_label,
    '孔隙压力 MPa': p_pore,
    '垂向地应力 MPa': sigma_v,
    '最小水平主应力 MPa': avg_sh,
    '最大水平主应力 MPa': sigma_H,
    '竖向地应力梯度 kPa/m': sv_gradient,
    '最小水平主应力梯度 kPa/m': sh_gradient,
    '最大水平主应力梯度 kPa/m': sH_gradient,
    'Mises剪应力 MPa': mises_stress,
    'σH计算方法': sigma_h_method,
    'σH校核': '需复核：σH<σh' if sigma_H_is_suspect else '通过',
}])
st.dataframe(style_report_table(stress_result_df), use_container_width=True)

st.caption(
    f"σH_Pr法 = 3σh - Pr_avg - Pp = {sigma_H_from_pr:.3f} MPa；"
    f"σH_Pb法 = 3σh - Pb + T - Pp = {sigma_H_from_pb:.3f} MPa。"
)

with st.expander("🧮 孔隙压力 Pp 计算过程"):
    pore_rows = [{
        '估算方法': pore_method,
        '代表深度 m': depth_mid,
        '自动估算 Pp MPa': pore_pressure_auto,
        '最终采用 Pp MPa': p_pore,
        '地层流体密度 g/cm³': fluid_density,
        '钻井液密度 g/cm³': mud_density,
        '关井立管压力 MPa': standpipe_pressure,
        '经验公式 a MPa/m': regression_slope,
        '经验公式 b MPa': regression_intercept,
    }]
    st.dataframe(pd.DataFrame(pore_rows).style.format({
        '代表深度 m': '{:.2f}',
        '自动估算 Pp MPa': '{:.3f}',
        '最终采用 Pp MPa': '{:.3f}',
        '地层流体密度 g/cm³': '{:.3f}',
        '钻井液密度 g/cm³': '{:.3f}',
        '关井立管压力 MPa': '{:.3f}',
        '经验公式 a MPa/m': '{:.5f}',
        '经验公式 b MPa': '{:.3f}',
    }), use_container_width=True)
    st.markdown(
        """
        - 静水压力法：`Pp = 10^-3 * ρf * g * H`
        - 关井立管压力法：`Pp = Ps + 10^-3 * ρm * g * H`
        - 区域经验公式法：`Pp = aH + b`
        - 手动/DFIT解释值：直接采用输入值；最终修正值大于 0 时优先采用修正值。
        """
    )

with st.expander("🧯 井底压力补偿说明"):
    pressure_comp_df = pd.DataFrame([{
        '补偿方法': friction_method,
        '常数摩阻 MPa': constant_friction_loss,
        '排量平方系数': q2_friction_coeff,
        '最大扣除 MPa': float(np.nanmax(df['friction_loss'])) if 'friction_loss' in df else 0.0,
        '平均扣除 MPa': float(np.nanmean(df['friction_loss'])) if 'friction_loss' in df else 0.0,
    }])
    st.dataframe(pressure_comp_df.style.format({
        '常数摩阻 MPa': '{:.3f}',
        '排量平方系数': '{:.4f}',
        '最大扣除 MPa': '{:.3f}',
        '平均扣除 MPa': '{:.3f}',
    }), use_container_width=True)
    st.markdown(
        """
        - 不校正：`P = P_raw`
        - 常数摩阻扣除：`P = P_raw - ΔPf`
        - 排量平方摩阻扣除：`P = P_raw - kQ²`
        - 修正后的 `P` 用于 `Pb`、`Pr`、`ISIP`、`Pc` 和 `σH` 计算；原始压力仅作为图上对照。
        """
    )

if sigma_H_is_suspect:
    st.warning(
        "当前采用的 σH 结果小于 σh，成果表保留公式计算值用于追溯，"
        "但不建议直接作为最终最大水平主应力。可尝试：降低偏高的 Pr 人工校准值、"
        "核对 Pp，或切换到 Pb 法做对照。"
    )

st.divider()
st.subheader("🔍 本层地应力计算结论")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("测试段", depth_label)
k2.metric("本层 σh", f"{avg_sh:.2f} MPa")
k3.metric("最大 σH", f"{sigma_H:.2f} MPa", delta="需复核" if sigma_H_is_suspect else None)
k4.metric("垂向 σv", f"{sigma_v:.2f} MPa")
k5.metric("Mises剪应力", f"{mises_stress:.2f} MPa")

st.caption(
    f"{well_name} - {layer_name}：参与压降 {len(valid_results)}/{len(cycle_results)} 次；"
    f"平均 ISIP={avg_isip:.2f} MPa；平均 Pr={avg_pr:.2f} MPa；"
    f"σH 当前采用：{sigma_h_method}。"
)
