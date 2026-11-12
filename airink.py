"""airink.py - shared library for the TDS 2026 Sep P1 Q9 "Invisible Ink" decoder (v15).

Pipeline for one recording (orientation stream; the hidden test files have no pen state):

  trace_recording()   Python port of renderer.js trace(): orientation -> smoothed cursor x, y
  extract_features()  ~250 kinematic features per sample (unchanged from v14)
  ink model           Stage-1 LightGBM "is the pen down?" + Stage-2 LightGBM that re-reads the
                      Stage-1 probabilities in context. Target = 4th sample column (pen state).
  postprocess_ink()   drop ink before the last "clear", pop strokes removed by "undo"
  decoders            split the ink into exactly `length` letters
                        v14 / rules-lite : x-slot DP (+/- the v14 hand-written rules)
                        joint-*          : over-segment the ink, then a DP picks the grouping whose
                                           letters the OCR reads best (segmentation by recognition)
                        auto             : run v14 and joint-time, keep the more confident reading
  OCR                 5-model ensemble on a 253-number letter descriptor -> 17 letters
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import shutil
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.ndimage import binary_closing, gaussian_filter, uniform_filter1d
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

VERSION = "v15"
ALPHABET = "ACEFHJKMNPRTUVWXY"
RAW_DIR = "data/raw"
MASTER_JSONL = "master_training_pooled.jsonl"
DEFAULT_MODEL = "models/airink_v15.joblib"
BEST_DECODER_FILE = "models/best_decoder.json"
DEFAULT_DECODER = "v14"
FEATURE_EXCLUDE = {"t", "segment", "pen_down", "rec_id", "target"}


def log(msg=""):
    print(msg, flush=True)


def clamp(n, lo, hi):
    return max(lo, min(hi, n))


def _runs(mask):
    """Inclusive (start, end) pairs of consecutive True values in a 1-D mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


# --------------------------------------------------------------------------------------
# Data I/O
# --------------------------------------------------------------------------------------
def read_records(path):
    """Read a JSON object, JSON array, {"records": [...]} or JSONL file (same rules as renderer.js)."""
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data["records"] if isinstance(data.get("records"), list) else [data]
    return [r for r in data if isinstance(r, dict)]


def check_recording(r):
    """Raise ValueError if `r` cannot be decoded (labeled or hidden)."""
    rid = r.get("id")
    if not isinstance(rid, str):
        raise ValueError("recording without an id")
    if r.get("length") not in (3, 4, 5):
        raise ValueError(f"{rid}: length must be 3, 4 or 5 (got {r.get('length')!r})")
    s = r.get("samples")
    if not isinstance(s, list) or len(s) < 2:
        raise ValueError(f"{rid}: needs at least 2 samples")
    if not all(isinstance(x, (list, tuple)) and len(x) >= 3 for x in s):
        raise ValueError(f"{rid}: every sample must be [t, beta, gamma, ...]")
    return r


def is_labeled(r):
    code = r.get("code")
    return (isinstance(code, str) and len(code) == r.get("length")
            and all(c in ALPHABET for c in code)
            and isinstance(r.get("strokes"), list)
            and any(len(x) >= 4 for x in r.get("samples", [])))


def _record_hash(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def data_fingerprint(records):
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x["id"]):
        h.update(_record_hash(r).encode())
    return h.hexdigest()[:16]


def sync_training_data(raw_dir=RAW_DIR, master=MASTER_JSONL, verbose=True):
    """Move new *training*.jsonl files from the project root into data/raw/, merge every file in
    data/raw/ (plus the existing master), de-duplicate by recording id, rewrite the master file."""
    os.makedirs(raw_dir, exist_ok=True)
    for f in glob.glob("*training*.jsonl"):
        if os.path.basename(f) == os.path.basename(master):
            continue
        dest = os.path.join(raw_dir, os.path.basename(f))
        if os.path.exists(dest):
            with open(f, "rb") as fh:
                suffix = hashlib.sha256(fh.read()).hexdigest()[:8]
            stem, ext = os.path.splitext(dest)
            dest = f"{stem}-{suffix}{ext}"
        if not os.path.exists(dest):
            shutil.move(f, dest)

    sources = sorted(glob.glob(os.path.join(raw_dir, "*.jsonl")) + glob.glob(os.path.join(raw_dir, "*.json")))
    if os.path.exists(master):
        sources.append(master)
    by_id, skipped, conflicts = {}, 0, 0
    for path in sources:
        try:
            recs = read_records(path)
        except Exception as e:  # a broken download must not stop training
            log(f"WARNING: could not read {path}: {e}")
            continue
        for r in recs:
            try:
                check_recording(r)
            except ValueError:
                skipped += 1
                continue
            if not is_labeled(r):
                skipped += 1
                continue
            old = by_id.get(r["id"])
            if old is None:
                by_id[r["id"]] = r
            elif _record_hash(old) != _record_hash(r):
                conflicts += 1
    records = [by_id[k] for k in sorted(by_id)]
    tmp = f"{master}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(tmp, master)
    if verbose:
        log(f"Data: {len(records)} unique labeled recordings from {len(sources)} files -> {master}"
            + (f" | skipped {skipped} unlabeled/invalid" if skipped else "")
            + (f" | {conflicts} duplicate ids with different content (kept first)" if conflicts else ""))
    return records


def strip_labels(r):
    """Return a copy that looks like a live test recording (no code, strokes or pen column)."""
    out = {k: v for k, v in r.items() if k not in ("code", "strokes")}
    out["samples"] = [list(s[:3]) for s in r["samples"]]
    return out


# --------------------------------------------------------------------------------------
# Trace (port of renderer.js trace) and undo/clear bookkeeping
# --------------------------------------------------------------------------------------
def _shortest(a, b):
    return ((a - b + 540.0) % 360.0) - 180.0


def control_times(r):
    """(time of last clear or -inf, sorted undo times after that clear)."""
    controls = sorted(r.get("controls") or [], key=lambda c: c["t"])
    clears = [c["t"] for c in controls if c.get("type") == "clear"]
    last_clear = max(clears) if clears else -math.inf
    undos = [float(c["t"]) for c in controls if c.get("type") == "undo" and c["t"] > last_clear]
    return last_clear, undos


def trace_recording(r):
    s = r["samples"]
    controls = sorted(r.get("controls") or [], key=lambda c: c["t"])
    canvas = r.get("canvas") or {}
    w = float(canvas.get("width") or 360.0)
    h = float(canvas.get("height") or 300.0)
    screen = str(canvas.get("screen") or "portrait-primary")
    if "landscape" in screen:
        angle = -math.pi / 2.0 if "secondary" in screen else math.pi / 2.0
    elif "portrait-secondary" in screen:
        angle = math.pi
    else:
        angle = 0.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    pad = min(22.0, w * 0.065, h * 0.065)
    last_clear, undos = control_times(r)

    c_idx, origin = 0, None
    x, y = w / 2.0, h / 2.0
    prior = float(s[0][0])
    segment = 0
    rows = []
    for sample in s:
        t, beta, gamma = float(sample[0]), float(sample[1]), float(sample[2])
        pen = int(sample[3]) if len(sample) > 3 and sample[3] is not None else -1
        reset = 0
        while c_idx < len(controls) and controls[c_idx]["t"] <= t:
            ev = controls[c_idx]
            c_idx += 1
            if ev.get("type") == "recenter":
                o = ev.get("origin")
                if isinstance(o, dict) and "beta" in o and "gamma" in o:
                    origin = {"beta": float(o["beta"]), "gamma": float(o["gamma"])}
                else:
                    origin = {"beta": beta, "gamma": gamma}
                x, y = w / 2.0, h / 2.0
                segment += 1
                reset = 1
        if origin is None:
            origin = {"beta": beta, "gamma": gamma}
        db = _shortest(beta, origin["beta"])
        dg = _shortest(gamma, origin["gamma"])
        gx = dg * cos_a + db * sin_a
        gy = db * cos_a - dg * sin_a
        tx = clamp(w / 2.0 + (gx / 29.0) * w * 0.46, pad, w - pad)
        ty = clamp(h / 2.0 + (gy / 34.0) * h * 0.46, pad, h - pad)
        a = 1.0 - math.exp(-clamp(t - prior, 0.0, 250.0) / 1000.0 * 19.0)
        x += a * (tx - x)
        y += a * (ty - y)
        rows.append((t, x / w, y / h, beta, gamma, segment, reset, 1 if t > last_clear else 0, pen))
        prior = t
    df = pd.DataFrame(rows, columns=["t", "x", "y", "beta", "gamma", "segment", "reset", "after_clear", "pen_down"])
    tt = df["t"].values
    if undos:
        u = np.asarray(undos)
        idx = np.searchsorted(u, tt, side="left")
        nxt = np.where(idx < len(u), u[np.minimum(idx, len(u) - 1)] - tt, 15000.0)
    else:
        nxt = np.full(len(df), 15000.0)
    df.insert(8, "time_to_undo", np.minimum(nxt, 15000.0))
    return df


def final_ink_mask(df, r):
    """Samples whose ink is still on the final canvas: pen down, after the last clear and not
    removed by an undo (each undo pops the most recent surviving stroke). Training data only."""
    t = df["t"].values
    pen = (df["pen_down"].values == 1) & (df["after_clear"].values == 1)
    runs = _runs(pen)
    alive = [True] * len(runs)
    for ut in control_times(r)[1]:
        for i in range(len(runs) - 1, -1, -1):
            if alive[i] and t[runs[i][0]] < ut:
                alive[i] = False
                break
    mask = np.zeros(len(df), dtype=bool)
    for (a, b), ok in zip(runs, alive):
        if ok:
            mask[a:b + 1] = True
    return mask, int(sum(alive))


# --------------------------------------------------------------------------------------
# Per-sample features (unchanged from v14 apart from float32 storage)
# --------------------------------------------------------------------------------------
def extract_features(df, rec_length):
    cols = {}
    t = df["t"].values
    x = df["x"].values
    y = df["y"].values
    beta = df["beta"].values
    gamma = df["gamma"].values
    reset = df["reset"].values
    after_clear = df["after_clear"].values

    t_min, t_max = t[0], t[-1]
    duration = max(1.0, t_max - t_min)
    cols["rel_t"] = (t - t_min) / duration
    cols["code_length"] = np.full(len(df), rec_length, dtype=np.float32)
    if np.any(after_clear == 1):
        t_c = t[np.argmax(after_clear == 1)]
        cols["rel_t_after_clear"] = np.clip((t - t_c) / max(1.0, t_max - t_c), 0.0, 1.0)
    else:
        cols["rel_t_after_clear"] = cols["rel_t"]

    active_x = x[after_clear == 1] if np.any(after_clear == 1) else x
    active_y = y[after_clear == 1] if np.any(after_clear == 1) else y
    x_min, x_max = np.min(active_x), np.max(active_x)
    y_min, y_max = np.min(active_y), np.max(active_y)
    cols["rel_box_x"] = (x - x_min) / max(0.05, x_max - x_min)
    cols["rel_box_y"] = (y - y_min) / max(0.05, y_max - y_min)
    cols["x_minus_rel_t"] = cols["rel_box_x"] - cols["rel_t_after_clear"]

    dt = np.clip(np.diff(t, prepend=t[0] - 50.0), 1.0, None)
    cols["dt"] = dt
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dx[reset == 1] = 0.0
    dy[reset == 1] = 0.0
    step_dist = np.hypot(dx, dy)
    vx = dx / (dt / 1000.0)
    vy = dy / (dt / 1000.0)
    speed = np.hypot(vx, vy)
    cols["vx"] = vx
    cols["vy"] = vy
    cols["speed"] = speed
    cols["vy_ratio"] = vy / (speed + 1e-4)
    cols["vx_ratio"] = vx / (speed + 1e-4)
    cols["vert_dominance"] = (np.abs(vy) - np.abs(vx)) / (speed + 1e-4)
    cols["up_left_jump"] = np.where((vy < 0) & (vx < 0), speed, 0.0)
    cols["down_stroke_energy"] = np.where(vy > 0, speed, 0.0)

    ax = np.diff(vx, prepend=vx[0]) / (dt / 1000.0)
    ay = np.diff(vy, prepend=vy[0]) / (dt / 1000.0)
    acc = np.hypot(ax, ay)
    cols["ax"] = ax
    cols["ay"] = ay
    cols["acc"] = acc
    cols["d_speed"] = np.diff(speed, prepend=speed[0])
    cols["jerk"] = np.diff(acc, prepend=acc[0])

    angles = np.arctan2(dy, dx)
    d_angle = ((np.diff(angles, prepend=angles[0]) + np.pi) % (2 * np.pi)) - np.pi
    abs_turn = np.abs(d_angle)
    cols["abs_turn"] = abs_turn
    cols["curvature"] = abs_turn / (step_dist + 1e-4)
    cols["cos_angle"] = np.cos(angles)
    cols["sin_angle"] = np.sin(angles)

    d_beta = np.diff(beta, prepend=beta[0])
    d_gamma = np.diff(gamma, prepend=gamma[0])
    cols["d_beta"] = d_beta
    cols["d_gamma"] = d_gamma
    cols["rot_speed"] = np.hypot(d_beta, d_gamma)
    cols["dist_center"] = np.hypot(x - 0.5, y - 0.5)
    cols["dist_edge"] = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))

    for name, arr in [("speed", speed), ("acc", acc), ("rot", cols["rot_speed"]), ("vy", vy), ("vx", vx)]:
        mean_v, std_v = np.mean(arr), np.std(arr) + 1e-5
        cols[f"{name}_zscore"] = (arr - mean_v) / std_v
        cols[f"{name}_rank"] = pd.Series(arr).rank(pct=True).values

    s_x, s_y, s_step, s_speed = pd.Series(x), pd.Series(y), pd.Series(step_dist), pd.Series(speed)
    for w_eff in [5, 11, 21, 35]:
        half = w_eff // 2
        net_dx = s_x.shift(-half).ffill() - s_x.shift(half).bfill()
        net_dy = s_y.shift(-half).ffill() - s_y.shift(half).bfill()
        net_disp = np.hypot(net_dx, net_dy)
        path_len = s_step.rolling(w_eff, center=True, min_periods=1).sum()
        cols[f"straightness_{w_eff}"] = (net_disp / (path_len + 1e-5)).values
        loc_ymin = s_y.rolling(w_eff, center=True, min_periods=1).min()
        loc_ymax = s_y.rolling(w_eff, center=True, min_periods=1).max()
        cols[f"local_y_span_{w_eff}"] = (loc_ymax - loc_ymin).values
        cols[f"local_y_pos_{w_eff}"] = ((s_y - loc_ymin) / (loc_ymax - loc_ymin + 0.02)).values

    for k_win in [4, 8, 14]:
        bwd_speed = s_speed.rolling(k_win, min_periods=1).mean()
        fwd_speed = s_speed.iloc[::-1].rolling(k_win, min_periods=1).mean().iloc[::-1]
        cols[f"fwd_bwd_speed_diff_{k_win}"] = (fwd_speed - bwd_speed).values
        cols[f"fwd_bwd_speed_ratio_{k_win}"] = ((fwd_speed + 1e-3) / (bwd_speed + 1e-3)).values

    is_cusp = ((speed[1:-1] <= speed[:-2]) & (speed[1:-1] <= speed[2:])) | (abs_turn[1:-1] > 1.15)
    cusp_flags = np.zeros(len(df), dtype=int)
    cusp_flags[1:-1] = is_cusp.astype(int)
    micro_id = np.cumsum(cusp_flags)
    s_df = pd.DataFrame({"m_id": micro_id, "step": step_dist, "vx": vx, "vy": vy, "sp": speed, "y": y, "x": x})
    m_grp = s_df.groupby("m_id")
    cols["micro_path_len"] = m_grp["step"].transform("sum").values
    cols["micro_duration"] = m_grp["step"].transform("count").values
    cols["micro_mean_vy"] = m_grp["vy"].transform("mean").values
    cols["micro_mean_vx"] = m_grp["vx"].transform("mean").values
    cols["micro_max_sp"] = m_grp["sp"].transform("max").values
    cols["micro_y_span"] = (m_grp["y"].transform("max") - m_grp["y"].transform("min")).values
    cols["micro_x_span"] = (m_grp["x"].transform("max") - m_grp["x"].transform("min")).values

    base = {"x": s_x, "y": s_y, "vx": pd.Series(vx), "vy": pd.Series(vy), "speed": s_speed,
            "acc": pd.Series(acc), "abs_turn": pd.Series(abs_turn)}
    for shift in [-7, -5, -3, -2, -1, 1, 2, 3, 5, 7]:
        for name in ["x", "y", "vx", "vy", "speed", "abs_turn"]:
            cols[f"{name}_shift_{shift}"] = base[name].shift(shift).bfill().ffill().values
    for win in [3, 5, 9, 15, 25, 41]:
        for name in ["speed", "acc", "abs_turn", "vx", "vy", "y"]:
            rolled = base[name].rolling(window=win, center=True, min_periods=1)
            cols[f"{name}_mean_{win}"] = rolled.mean().values
            cols[f"{name}_std_{win}"] = rolled.std().fillna(0.0).values

    feat_df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
    feature_cols = [c for c in feat_df.columns if c not in FEATURE_EXCLUDE]
    feat_df[feature_cols] = feat_df[feature_cols].astype(np.float32)
    return feat_df, feature_cols


def build_features(r):
    """Trace + features for one recording; adds rec_id and (if labeled) the pen-down target."""
    df_feat, _ = extract_features(trace_recording(r), r["length"])
    df_feat["rec_id"] = r["id"]
    if is_labeled(r):
        df_feat["target"] = (df_feat["pen_down"].values == 1).astype(np.int8)
    return df_feat


def feature_columns(df_feat):
    return [c for c in df_feat.columns if c not in FEATURE_EXCLUDE]


# --------------------------------------------------------------------------------------
# Ink model: Stage 1 (pen down per sample) + Stage 2 (context refiner)
# --------------------------------------------------------------------------------------
S1_PARAMS = [
    dict(objective="binary", metric="auc", learning_rate=0.035, num_leaves=31, min_child_samples=25,
         subsample=0.8, colsample_bytree=0.72, reg_alpha=0.5, reg_lambda=1.0, n_estimators=900,
         random_state=42, n_jobs=-1, verbose=-1),
    dict(objective="binary", metric="auc", learning_rate=0.035, num_leaves=47, min_child_samples=35,
         subsample=0.85, colsample_bytree=0.65, reg_alpha=1.0, reg_lambda=2.0, n_estimators=900,
         random_state=108, n_jobs=-1, verbose=-1),
]
S2_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=15, min_child_samples=50,
                 subsample=0.8, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=2.0, n_estimators=500,
                 random_state=100, n_jobs=-1, verbose=-1)
S2_ANCHORS = ["after_clear", "time_to_undo", "rel_t_after_clear", "rel_box_x", "rel_box_y", "speed_zscore",
              "vy_ratio", "vert_dominance", "micro_path_len", "micro_y_span", "code_length"]


def _grp_roll(s, g, win, how):
    r = getattr(s.groupby(g).rolling(win, center=True, min_periods=1), how)()
    return r.reset_index(level=0, drop=True).reindex(s.index).values


def build_stage2_features(df_feat, s1, rec_ids=None):
    """Stage-2 inputs. v15 fix: every window/shift stays inside its own recording and results are
    re-aligned by index (v14 relied on records being sorted by id and filled edges across records)."""
    s = pd.Series(np.asarray(s1, dtype=float), index=df_feat.index)
    g = pd.Series(np.asarray(rec_ids) if rec_ids is not None else np.zeros(len(s), dtype=int), index=df_feat.index)
    out = {"s1_prob": s.values, "s1_prob_rank": s.groupby(g).rank(pct=True).values}
    for k in [-14, -10, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 10, 14]:
        sh = s.groupby(g).shift(k)
        out[f"s1_prob_shift_{k}"] = sh.groupby(g).transform(lambda v: v.bfill().ffill()).values
    for win in [3, 5, 9, 15, 25, 41]:
        out[f"s1_prob_mean_{win}"] = _grp_roll(s, g, win, "mean")
        out[f"s1_prob_max_{win}"] = _grp_roll(s, g, win, "max")
        out[f"s1_prob_min_{win}"] = _grp_roll(s, g, win, "min")
        out[f"s1_prob_std_{win}"] = np.nan_to_num(_grp_roll(s, g, win, "std"))
    for span in [4, 10, 22]:
        out[f"s1_ema_fwd_{span}"] = s.groupby(g).transform(lambda v: v.ewm(span=span).mean()).values
        out[f"s1_ema_bwd_{span}"] = s.groupby(g).transform(
            lambda v: v.iloc[::-1].ewm(span=span).mean().iloc[::-1]).values
    out["s1_diff_mean_9"] = s.values - out["s1_prob_mean_9"]
    out["s1_diff_mean_25"] = s.values - out["s1_prob_mean_25"]
    out["s1_Range_9"] = out["s1_prob_max_9"] - out["s1_prob_min_9"]
    for c in S2_ANCHORS:
        out[c] = df_feat[c].values
    return pd.DataFrame(out, index=df_feat.index)


def fit_ink_model(frame, feat_cols, row_groups, n_folds=5, fast=False, verbose=True):
    """Returns (ink_model_dict, {rec_id: (oof_blend, oof_stage1)}). OOF = predicted by models that
    did not see that recording's group; used to train Stage 2 and the OCR on realistic ink."""
    X1 = frame[feat_cols]
    y = frame["target"].values.astype(int)
    n_folds = max(2, min(n_folds, len(np.unique(row_groups))))
    folds = list(GroupKFold(n_splits=n_folds).split(X1, y, row_groups))
    params = S1_PARAMS[:1] if fast else S1_PARAMS

    oof1 = np.zeros(len(frame))
    s1_models = []
    for tr, va in folds:
        preds = []
        for p in params:
            m = lgb.LGBMClassifier(**p)
            m.fit(X1.iloc[tr], y[tr], eval_set=[(X1.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            preds.append(m.predict_proba(X1.iloc[va])[:, 1])
            s1_models.append(m)
        oof1[va] = np.mean(preds, axis=0)
    auc1 = roc_auc_score(y, oof1)

    X2 = build_stage2_features(frame, oof1, frame["rec_id"].values)
    s2_cols = list(X2.columns)
    oof2 = np.zeros(len(frame))
    s2_models = []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**S2_PARAMS)
        m.fit(X2.iloc[tr], y[tr], eval_set=[(X2.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        oof2[va] = m.predict_proba(X2.iloc[va])[:, 1]
        s2_models.append(m)

    best_alpha, auc2 = 0.5, 0.0
    for alpha in np.linspace(0.0, 1.0, 21):
        sc = roc_auc_score(y, (1 - alpha) * oof1 + alpha * oof2)
        if sc > auc2:
            auc2, best_alpha = sc, float(alpha)
    blend = (1 - best_alpha) * oof1 + best_alpha * oof2
    best_th, best_f1 = 0.5, 0.0
    for th in np.linspace(0.30, 0.70, 41):
        sc = f1_score(y, (blend >= th).astype(int))
        if sc > best_f1:
            best_f1, best_th = sc, float(th)
    if verbose:
        log(f"  ink model: Stage-1 AUC {auc1:.4f} -> blended AUC {auc2:.4f} (alpha={best_alpha:.2f}) | "
            f"F1 {best_f1:.4f} at th={best_th:.2f}  [{n_folds}-fold grouped OOF]")
    oof = {rid: (blend[idx], oof1[idx]) for rid, idx in frame.groupby("rec_id").indices.items()}
    ink = dict(s1_models=s1_models, s2_models=s2_models, s1_cols=list(feat_cols), s2_cols=s2_cols,
               alpha=best_alpha, th=best_th, auc_s1=float(auc1), auc_s2=float(auc2), f1=float(best_f1))
    return ink, oof


def ink_probabilities(df_feat, ink):
    """(blended probability, stage-1 probability) for every sample of an unseen recording."""
    s1 = np.mean([m.predict_proba(df_feat[ink["s1_cols"]])[:, 1] for m in ink["s1_models"]], axis=0)
    df2 = build_stage2_features(df_feat, s1, None)
    s2 = np.mean([m.predict_proba(df2[ink["s2_cols"]])[:, 1] for m in ink["s2_models"]], axis=0)
    a = ink["alpha"]
    return (1.0 - a) * s1 + a * s2, s1


def postprocess_ink(df_feat, r, raw, s1, th, rescue=True, undo="v14"):
    """Mask ink before the last clear, pop strokes removed by undo, and (optionally) rescue x-slots
    that look empty. Returns per-sample ink probabilities.
    undo="v14": walk back from the undone stroke while the smoothed probability stays >= 0.18 (can
                erase earlier letters written without a clear pause - seen in testing).
    undo="dip": same, but also stop at the first sample whose raw probability drops below 0.6*th,
                i.e. at the pen lift before the undone stroke."""
    s1 = np.array(s1, dtype=float, copy=True)
    raw = np.asarray(raw, dtype=float)
    after_clear = df_feat["after_clear"].values
    probs = uniform_filter1d(raw * after_clear, size=3)
    t = df_feat["t"].values
    for ut in control_times(r)[1]:
        pre = (t <= ut) & (t >= ut - 6500.0) & (probs >= 0.25)
        if np.any(pre):
            end_i = np.where(pre)[0][-1]
            start_i = end_i
            while start_i > 0 and probs[start_i - 1] >= 0.18 and (t[start_i] - t[start_i - 1]) < 350:
                if undo == "dip" and raw[start_i - 1] < 0.6 * th and start_i < end_i:
                    break
                start_i -= 1
            probs[start_i:end_i + 1] = 0.02
            s1[start_i:end_i + 1] = 0.02
            raw = raw.copy()
            raw[start_i:end_i + 1] = 0.02
    if not rescue:
        return uniform_filter1d(probs, size=2)

    k = r["length"]
    x = df_feat["x"].values
    active_x = x[probs >= 0.28] if np.any(probs >= 0.28) else x
    x_min, x_max = np.percentile(active_x, 2), np.percentile(active_x, 98)
    slot_edges = np.linspace(x_min, x_max, k + 1)
    slot_w = (x_max - x_min) / k
    boosted = probs.copy()
    vy = df_feat["vy"].values
    speed = df_feat["speed"].values
    for s_idx in range(k):
        core_left = slot_edges[s_idx] + 0.14 * slot_w
        core_right = slot_edges[s_idx + 1] - 0.14 * slot_w
        in_core = (after_clear == 1) & (x >= core_left) & (x <= core_right)
        in_slot = (after_clear == 1) & (x >= slot_edges[s_idx]) & (x <= slot_edges[s_idx + 1])
        if np.sum(in_core & (boosted >= th)) < 6 and np.sum(in_slot) >= 5:
            kin = np.where(vy > -0.05, np.clip(speed / (np.percentile(speed, 85) + 1e-4), 0, 1), 0.15)
            local = 0.65 * s1 + 0.35 * boosted + 0.20 * kin
            p88 = np.percentile(local[in_core] if np.sum(in_core) >= 4 else local[in_slot], 88)
            if p88 > 0.04:
                rescued = np.clip((local[in_slot] / p88) * (th * 1.10), 0.0, 0.92)
                med = np.percentile(local[in_slot], 35)
                rescued = np.where(local[in_slot] >= med, rescued, boosted[in_slot])
                boosted[in_slot] = np.maximum(boosted[in_slot], rescued)
    return uniform_filter1d(boosted, size=2)


# --------------------------------------------------------------------------------------
# Decoder 1: v14 x-slot DP (+ optional hand-written geometric rules)
# --------------------------------------------------------------------------------------
def _strict_runs(mask_arr, same_seg):
    runs, idx, n = [], 0, len(mask_arr)
    while idx < n:
        if mask_arr[idx]:
            end_i = idx
            while end_i + 1 < n and mask_arr[end_i + 1] and same_seg[end_i + 1]:
                end_i += 1
            runs.append((idx, end_i))
            idx = end_i + 1
        else:
            idx += 1
    return runs


def _v14_rules(letter_assignments, pts, seg_probs, same_seg, k, high_th):
    """The v14 rules (calibration-circle filter, below-baseline tail clipper, early tick filter,
    up-arm walker, vertical stem rescue). Tuned by eye on the v14 benchmark rows. Mutates seg_probs."""
    out_ids = letter_assignments.copy()
    n_seg = len(seg_probs)
    seg_x = (pts[:-1, 0] + pts[1:, 0]) / 2.0
    seg_y = (pts[:-1, 1] + pts[1:, 1]) / 2.0
    dx = pts[1:, 0] - pts[:-1, 0]
    dy = pts[1:, 1] - pts[:-1, 1]
    step_len = np.hypot(dx, dy)
    active_mask = out_ids != -1
    if np.sum(active_mask) < 4:
        return out_ids
    y_word_top = float(np.percentile(seg_y[active_mask], 10))
    y_word_bot = float(np.percentile(seg_y[active_mask], 90))
    word_h = max(0.18, y_word_bot - y_word_top)
    x_min_all = float(np.min(seg_x[active_mask]))
    x_max_all = float(np.max(seg_x[active_mask]))
    slot_w = max(0.12, (x_max_all - x_min_all) / k)
    slot_edges = np.linspace(x_min_all, x_max_all, k + 1)

    for l_idx in range(k):
        c_mask = out_ids == l_idx
        if not np.any(c_mask):
            continue
        other_mask = (out_ids != -1) & (out_ids != l_idx)
        other_bot = float(np.percentile(seg_y[other_mask], 90)) if np.sum(other_mask) >= 6 else y_word_bot
        other_top = float(np.percentile(seg_y[other_mask], 10)) if np.sum(other_mask) >= 6 else y_word_top
        other_h = max(0.16, other_bot - other_top)
        runs = _strict_runs(c_mask, same_seg)
        if len(runs) >= 2:
            run_stats = []
            for a, b in runs:
                sl = slice(a, b + 1)
                r_x, r_y = seg_x[sl], seg_y[sl]
                run_stats.append({"a": a, "b": b, "path": float(np.sum(step_len[sl])),
                                  "xspan": float(np.ptp(r_x)), "yspan": float(np.ptp(r_y)),
                                  "ymin": float(np.min(r_y)), "ymax": float(np.max(r_y))})
            for st in run_stats:  # Rule 1: enveloping calibration circle
                other_path = sum(o["path"] for o in run_stats if o is not st)
                if other_path >= 0.25 and ((st["yspan"] > 0.48 and st["xspan"] > 0.22)
                                           or (st["yspan"] > 1.38 * word_h and st["xspan"] > 0.85 * slot_w)):
                    out_ids[st["a"]:st["b"] + 1] = -1
                    seg_probs[st["a"]:st["b"] + 1] *= 0.15
                    st["path"] = 0.0
            surviving = [s for s in run_stats if s["path"] > 0]
            if l_idx == 0 and len(surviving) >= 2:  # Rule 3: early false-start tick in letter 1
                main_run = max(surviving, key=lambda s: s["path"])
                if main_run["path"] >= 0.35 and main_run["yspan"] >= 0.30:
                    for st in surviving:
                        if st is not main_run:
                            time_gap = main_run["a"] - st["b"]
                            if (time_gap >= 8 and st["path"] < 0.42 * main_run["path"]
                                    and st["ymax"] < main_run["ymax"] - 0.12):
                                out_ids[st["a"]:st["b"] + 1] = -1
                                seg_probs[st["a"]:st["b"] + 1] *= 0.15
                                st["path"] = 0.0
        # Rule 2: below-baseline tail on letters that already have >= 2 vertical reversals
        c_mask = out_ids == l_idx
        in_band = c_mask & (seg_y <= other_bot + 0.08 * other_h)
        low_tail = c_mask & (seg_y > other_bot + 0.08 * other_h)
        if np.any(low_tail) and np.sum(in_band) >= 10:
            dy_in = dy[in_band]
            if float(np.sum(step_len[in_band])) >= 0.48 and int(np.sum(dy_in[:-1] * dy_in[1:] < -1e-5)) >= 2:
                out_ids[low_tail] = -1
                seg_probs[low_tail] *= 0.15
        c_mask = out_ids == l_idx
        if not np.any(c_mask):
            continue
        idxs = np.where(c_mask)[0]
        l_ymin, l_ymax = float(np.min(seg_y[idxs])), float(np.max(seg_y[idxs]))
        l_h = max(0.05, l_ymax - l_ymin)
        # Rule 4: up-arm walker for a single-leg V without a top bar
        top_zone = idxs[seg_y[idxs] <= l_ymin + 0.28 * l_h]
        has_top_bar = (float(np.ptp(seg_x[top_zone])) if len(top_zone) >= 2 else 0.0) >= 0.055
        bot_i = int(idxs[int(np.argmax(seg_y[idxs]))])
        bot_x, bot_y = float(seg_x[bot_i]), float(seg_y[bot_i])
        after_bot = idxs[idxs > bot_i]
        upward_after_bot = float(np.sum(np.maximum(0.0, -dy[after_bot]))) if len(after_bot) else 0.0
        if (not has_top_bar) and (bot_y - l_ymin) >= 0.45 * word_h and upward_after_bot < 0.40 * word_h:
            f_i, cand_up = bot_i + 1, []
            max_allowed_x = min(slot_edges[l_idx + 1] + 0.035, bot_x + 0.70 * slot_w)
            while f_i < n_seg and len(cand_up) < 42 and same_seg[f_i] and out_ids[f_i] in (-1, l_idx):
                cx_f, cy_f = float(seg_x[f_i]), float(seg_y[f_i])
                if cx_f < bot_x - 0.04 or cx_f > max_allowed_x or cy_f > bot_y + 0.035:
                    break
                if (bot_y - cy_f) >= 0.40 * word_h and dy[f_i] > 0.005:
                    break
                cand_up.append(f_i)
                if cy_f <= min(l_ymin + 0.03, y_word_top + 0.06 * word_h):
                    break
                f_i += 1
            if len(cand_up) >= 4:
                if bot_y - float(np.min(seg_y[cand_up])) >= 0.45 * word_h and float(seg_x[cand_up[-1]]) >= bot_x + 0.02:
                    out_ids[cand_up] = l_idx
                    seg_probs[cand_up] = np.maximum(seg_probs[cand_up], high_th * 0.96)
        # Rule 5: detached vertical stem inside the slot
        idxs = np.where(out_ids == l_idx)[0]
        l_xmin, l_xmax = float(np.min(seg_x[idxs])), float(np.max(seg_x[idxs]))
        if float(np.sum(step_len[idxs])) < 0.48:
            cand = (same_seg & (out_ids == -1) & (seg_x >= l_xmin - 0.38 * slot_w) & (seg_x <= l_xmax + 0.25 * slot_w)
                    & (seg_y >= y_word_top - 0.06 * word_h) & (seg_y <= y_word_bot + 0.06 * word_h)
                    & (dy > 0.001) & (np.abs(dy) > np.abs(dx) * 1.15) & (seg_probs >= 0.08))
            for a, b in _strict_runs(cand, same_seg):
                if (b - a + 1) >= 4 and float(np.ptp(seg_y[a:b + 1])) >= 0.14 and float(np.ptp(seg_x[a:b + 1])) < 0.09:
                    out_ids[a:b + 1] = l_idx
                    seg_probs[a:b + 1] = np.maximum(seg_probs[a:b + 1], high_th * 0.95)
    return out_ids


def _stroke_obj(a, b, seg_x, seg_probs, step_len, n_seg, w_floor=None):
    st_x = seg_x[a:b + 1]
    w_arr = seg_probs[a:b + 1] if w_floor is None else np.maximum(seg_probs[a:b + 1], w_floor)
    w_arr = np.maximum(w_arr, 1e-6)
    return {"start": int(a), "end": int(b), "x_mean": float(np.average(st_x, weights=w_arr)),
            "x_min": float(np.min(st_x)), "x_max": float(np.max(st_x)),
            "t_mean": (a + b) / (2.0 * max(1, n_seg)), "w": float(np.sum(w_arr)),
            "path": float(np.sum(step_len[a:b + 1]))}


def _v14_segment(pts, seg_probs, same_seg, k, high_th):
    """v14 DP stroke partitioning into k letters ordered by x. Returns per-segment letter ids."""
    n_seg = len(seg_probs)
    seg_x = (pts[:-1, 0] + pts[1:, 0]) / 2.0
    step_len = np.hypot(pts[1:, 0] - pts[:-1, 0], pts[1:, 1] - pts[:-1, 1])
    low_th = max(0.21, high_th - 0.16)
    active = binary_closing(same_seg & (seg_probs >= low_th), structure=np.ones(3)) & same_seg
    raw = []
    for i, j in _runs(active):
        max_p = float(np.max(seg_probs[i:j + 1]))
        path_d = float(np.sum(step_len[i:j + 1]))
        is_speck = (j - i + 1 < 3) or (path_d < 0.050 and max_p < high_th + 0.16)
        is_edge = (i < n_seg * 0.08 or j > n_seg * 0.92) and (max_p < high_th + 0.10) and (path_d < 0.22)
        if not is_speck and not is_edge:
            raw.append(_stroke_obj(i, j, seg_x, seg_probs, step_len, n_seg))
    ids = np.full(n_seg, -1, dtype=int)
    if not raw:
        return ids
    x_min_all = min(s["x_min"] for s in raw)
    x_max_all = max(s["x_max"] for s in raw)
    edges = np.linspace(x_min_all, x_max_all, k + 1)
    for s_idx in range(k):
        left, right = edges[s_idx], edges[s_idx + 1]
        if not any(left <= st["x_mean"] <= right for st in raw):
            in_slot = np.where(same_seg & (seg_x >= left + 0.01) & (seg_x <= right - 0.01))[0]
            if len(in_slot) >= 4:
                best_i = in_slot[int(np.argmax(seg_probs[in_slot]))]
                a, b = max(in_slot[0], best_i - 6), min(in_slot[-1], best_i + 6)
                st = _stroke_obj(a, b, seg_x, seg_probs, step_len, n_seg, w_floor=0.32)
                st["path"] = max(0.14, st["path"])
                raw.append(st)
    raw.sort(key=lambda s: (round(s["x_mean"], 2), s["t_mean"]))
    ideal_width = max(0.15, x_max_all - x_min_all) / k

    strokes = []
    for st in raw:
        if (st["x_max"] - st["x_min"]) > ideal_width * 1.25 and (st["end"] - st["start"]) >= 7:
            a, b = st["start"], st["end"]
            margin = max(2, (b - a) // 5)
            split = a + margin + int(np.argmin(seg_probs[a + margin:b - margin + 1]))
            strokes.append(_stroke_obj(a, split, seg_x, seg_probs, step_len, n_seg))
            strokes.append(_stroke_obj(split + 1, b, seg_x, seg_probs, step_len, n_seg))
        else:
            strokes.append(st)
    strokes.sort(key=lambda s: (round(s["x_mean"], 2), s["t_mean"]))
    while len(strokes) < k:
        splittable = [i for i, s in enumerate(strokes) if (s["end"] - s["start"]) >= 2]
        if not splittable:
            break
        wi = max(splittable, key=lambda i: (strokes[i]["x_max"] - strokes[i]["x_min"]) + 0.1 * strokes[i]["path"])
        st = strokes.pop(wi)
        a, b = st["start"], st["end"]
        mid = a + 1 + int(np.argmin(seg_probs[a + 1:b])) if (b - a) >= 4 else (a + b) // 2
        strokes.insert(wi, _stroke_obj(mid + 1, b, seg_x, seg_probs, step_len, n_seg))
        strokes.insert(wi, _stroke_obj(a, mid, seg_x, seg_probs, step_len, n_seg))
        strokes.sort(key=lambda s: (round(s["x_mean"], 2), s["t_mean"]))

    M = len(strokes)
    if M <= k:
        for i, st in enumerate(strokes):
            ids[st["start"]:st["end"] + 1] = min(k - 1, i)
        return ids
    centers = x_min_all + (np.arange(k) + 0.5) * ideal_width
    ideal_path = max(0.10, sum(s["path"] for s in strokes) / k)

    def cost(a, b, li):
        g = strokes[a:b + 1]
        g_xmin = min(s["x_min"] for s in g)
        g_xmax = max(s["x_max"] for s in g)
        g_path = sum(s["path"] for s in g)
        g_xmean = np.average([s["x_mean"] for s in g], weights=[max(s["w"], 1e-6) for s in g])
        c = 30.0 * (max(0.0, (g_xmax - g_xmin) - ideal_width * 1.22) / ideal_width) ** 2
        if g_path < ideal_path * 0.38:
            c += 30.0 * ((ideal_path * 0.38 - g_path) / ideal_path) ** 2
        gap, cur = 0.0, g[0]["x_max"]
        for s in g[1:]:
            gap = max(gap, s["x_min"] - cur)
            cur = max(cur, s["x_max"])
        if gap > 0.015:
            c += 35.0 * (gap / ideal_width) ** 2
        c += 3.2 * ((g_xmean - centers[li]) / ideal_width) ** 2
        if b + 1 < M:
            nxt = strokes[b + 1]
            overlap = min(g_xmax, nxt["x_max"]) - max(g_xmin, nxt["x_min"])
            if overlap > 0.014:
                c += 16.0 * (overlap / ideal_width) + 2.0
            if nxt["x_mean"] - strokes[b]["x_mean"] < 0.028:
                c += 4.0
        return float(c)

    dp = np.full((k, M), np.inf)
    parent = np.zeros((k, M), dtype=int)
    for b in range(M - k + 1):
        dp[0, b] = cost(0, b, 0)
    for li in range(1, k):
        for b in range(li, M - (k - 1 - li)):
            for a in range(li, b + 1):
                c = dp[li - 1, a - 1] + cost(a, b, li)
                if c < dp[li, b]:
                    dp[li, b], parent[li, b] = c, a
    end = M - 1
    for li in range(k - 1, -1, -1):
        start = 0 if li == 0 else int(parent[li, end])
        start = max(li, min(start, end))
        for m in range(start, end + 1):
            ids[strokes[m]["start"]:strokes[m]["end"] + 1] = li
        end = start - 1
    return ids


def decode_rules(pts, seg_probs, same_seg, k, high_th, rules=True):
    """Returns (letter id per segment or -1, segment weights after the rules)."""
    sp = np.array(seg_probs, dtype=float, copy=True)
    ids = _v14_segment(pts, sp, same_seg, k, high_th)
    if rules:
        ids = _v14_rules(ids, pts, sp, same_seg, k, high_th)
    return ids, sp


# --------------------------------------------------------------------------------------
# Letter OCR
# --------------------------------------------------------------------------------------
DESC_DIM = 253  # 12x12 raster + 3x3x6 HOG + 3x8 vertical strips + 31 geometry numbers
OCR_WEIGHTS = {"et": 0.25, "rf": 0.15, "lgb": 0.25, "lr": 0.20, "mlp": 0.15}


def extract_letter_descriptor(segments_list, weights_list=None):
    """Same descriptor as v14 (vectorised). v15 fix: the empty case returns DESC_DIM zeros (was 252)."""
    if len(segments_list) == 0:
        return np.zeros(DESC_DIM, dtype=np.float32)
    segs = np.asarray(segments_list, dtype=np.float32).reshape(-1, 2, 2)
    weights = (np.ones(len(segs), dtype=np.float32) if weights_list is None
               else np.asarray(weights_list, dtype=np.float32).reshape(-1))
    if len(segs) >= 12 and np.max(weights) > 0.42:
        keep = weights >= 0.30
        if np.sum(keep) >= 8:
            segs, weights = segs[keep], weights[keep]
    pts = segs.reshape(-1, 2)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    w = max(0.02, float(xmax - xmin))
    h = max(0.02, float(ymax - ymin))
    scale = max(w, h)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    norm = np.empty_like(segs)
    norm[:, :, 0] = 0.5 + (segs[:, :, 0] - cx) / (scale * 1.15)
    norm[:, :, 1] = 0.5 + (segs[:, :, 1] - cy) / (scale * 1.15)
    norm = np.clip(norm, 0.0, 0.999)

    G = 12
    raw_dx = segs[:, 1, 0] - segs[:, 0, 0]
    raw_dy = segs[:, 1, 1] - segs[:, 0, 1]
    raw_len = np.hypot(raw_dx, raw_dy)
    dx = norm[:, 1, 0] - norm[:, 0, 0]
    dy = norm[:, 1, 1] - norm[:, 0, 1]
    lengths = np.hypot(dx, dy)
    orient = (np.arctan2(dy, dx) % np.pi) / np.pi * 6.0
    box_x0 = np.clip((segs[:, 0, 0] - xmin) / w, 0.0, 0.999)
    box_y0 = np.clip((segs[:, 0, 1] - ymin) / h, 0.0, 0.999)
    wi = weights * (lengths + 0.005)

    raster = np.zeros((G, G), dtype=np.float32)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = np.clip(((norm[:, 0, 0] + a * dx) * G).astype(int), 0, G - 1)
        gy = np.clip(((norm[:, 0, 1] + a * dy) * G).astype(int), 0, G - 1)
        np.add.at(raster, (gy, gx), wi)
    mx = 0.5 * (norm[:, 0, 0] + norm[:, 1, 0])
    my = 0.5 * (norm[:, 0, 1] + norm[:, 1, 1])
    hog = np.zeros((3, 3, 6), dtype=np.float32)
    np.add.at(hog, (np.clip((my * 3).astype(int), 0, 2), np.clip((mx * 3).astype(int), 0, 2),
                    np.clip(orient.astype(int), 0, 5)), wi)
    v_strips = np.zeros((3, 8), dtype=np.float32)
    np.add.at(v_strips, (np.minimum(2, (box_y0 * 3).astype(int)), np.minimum(7, (box_x0 * 8).astype(int))),
              weights * np.abs(raw_dy))

    raster = gaussian_filter(raster, sigma=0.65)
    if np.max(raster) > 0:
        raster /= np.max(raster)
    if np.linalg.norm(hog) > 0:
        hog /= (np.linalg.norm(hog) + 1e-5)
    if np.max(v_strips) > 0:
        v_strips /= np.max(v_strips)

    aspect = w / (h + 0.02)
    hw_ratio = h / (w + 0.02)
    start_x, start_y = norm[0, 0, 0], norm[0, 0, 1]
    end_x, end_y = norm[-1, 1, 0], norm[-1, 1, 1]
    r_sum = np.sum(raster) + 1e-5
    top_mass = np.sum(raster[:4, :]) / r_sum
    mid_mass = np.sum(raster[4:8, :]) / r_sum
    bot_mass = np.sum(raster[8:, :]) / r_sum
    left_mass = np.sum(raster[:, :4]) / r_sum
    center_mass = np.sum(raster[:, 4:8]) / r_sum
    right_mass = np.sum(raster[:, 8:]) / r_sum
    npts = norm.reshape(-1, 2)
    bot_pts = npts[npts[:, 1] >= 0.62, 0]
    top_pts = npts[npts[:, 1] <= 0.38, 0]
    bot_span = float(np.ptp(bot_pts)) if len(bot_pts) >= 2 else 0.0
    top_span = float(np.ptp(top_pts)) if len(top_pts) >= 2 else 0.0
    ux = dx / (lengths + 1e-5)
    uy = dy / (lengths + 1e-5)
    vert_energy = float(np.sum(np.abs(uy) * weights) / (np.sum(weights) + 1e-5))
    horz_energy = float(np.sum(np.abs(ux) * weights) / (np.sum(weights) + 1e-5))
    reversals = float(np.sum(dy[:-1] * dy[1:] < -1e-4)) / max(1.0, len(dy))
    total_dy_over_h = float(np.sum(np.abs(raw_dy)) / h)
    total_dx_over_w = float(np.sum(np.abs(raw_dx)) / w)
    total_len_over_h = float(np.sum(raw_len) / h)
    top_center_dip = float(np.mean(v_strips[0, 2:6]) - 0.5 * (np.mean(v_strips[0, :2]) + np.mean(v_strips[0, 6:])))
    bot_center_dip = float(np.mean(v_strips[2, 2:6]) - 0.5 * (np.mean(v_strips[2, :2]) + np.mean(v_strips[2, 6:])))
    mid_profile = np.mean(raster[4:8, :], axis=0)
    mid_peaks = float(np.sum((mid_profile[1:-1] > mid_profile[:-2]) & (mid_profile[1:-1] > mid_profile[2:])
                             & (mid_profile[1:-1] > 0.22)))
    bot_profile = np.mean(raster[8:, :], axis=0)
    bot_peaks = float(np.sum((bot_profile[1:-1] > bot_profile[:-2]) & (bot_profile[1:-1] > bot_profile[2:])
                             & (bot_profile[1:-1] > 0.22)))
    lr_mirror_diff = float(np.mean(np.abs(raster[:, :6] - raster[:, 6:][:, ::-1])))
    y_grid = np.arange(12, dtype=np.float32)[:, None]
    left_y_cog = float(np.sum(raster[:, 2:6] * y_grid) / (np.sum(raster[:, 2:6]) + 1e-5)) / 12.0
    right_y_cog = float(np.sum(raster[:, 6:10] * y_grid) / (np.sum(raster[:, 6:10]) + 1e-5)) / 12.0
    geom = np.array([
        aspect, hw_ratio, start_x, start_y, end_x, end_y,
        top_mass, mid_mass, bot_mass, left_mass, center_mass, right_mass,
        bot_span, top_span, bot_span - top_span, vert_energy, horz_energy, reversals,
        np.sum(raster[:6, :6]) / r_sum, np.sum(raster[:6, 6:]) / r_sum,
        mid_peaks, bot_peaks, lr_mirror_diff, right_y_cog - left_y_cog, left_y_cog, right_y_cog,
        total_dy_over_h, total_dx_over_w, total_len_over_h, top_center_dip, bot_center_dip,
    ], dtype=np.float32)
    return np.concatenate([raster.ravel(), hog.ravel(), v_strips.ravel(), geom]).astype(np.float32)


def segment_true_strokes_into_k(strokes_list, k):
    """Group the ground-truth canvas strokes into k letters (x-ordered DP, as in v14)."""
    objs = []
    for idx, st in enumerate(strokes_list):
        if not isinstance(st, list) or len(st) < 2:
            continue
        arr = np.array([p[:2] for p in st], dtype=np.float32)
        path = float(np.sum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))))
        if path < 0.02:
            continue
        objs.append({"segs": np.stack([arr[:-1], arr[1:]], axis=1), "x_mean": float(np.mean(arr[:, 0])),
                     "x_min": float(np.min(arr[:, 0])), "x_max": float(np.max(arr[:, 0])), "order": idx, "path": path})
    if not objs:
        return [[] for _ in range(k)]
    objs.sort(key=lambda s: (round(s["x_mean"], 2), s["order"]))
    M = len(objs)
    if M < k:
        return [[] for _ in range(k)]
    x_min_all = min(s["x_min"] for s in objs)
    x_max_all = max(s["x_max"] for s in objs)
    ideal_w = max(0.15, x_max_all - x_min_all) / k
    centers = x_min_all + (np.arange(k) + 0.5) * ideal_w
    ideal_path = sum(s["path"] for s in objs) / k

    def cost(a, b, li):
        g = objs[a:b + 1]
        g_w = max(s["x_max"] for s in g) - min(s["x_min"] for s in g)
        g_path = sum(s["path"] for s in g)
        g_xm = np.average([s["x_mean"] for s in g], weights=[s["path"] for s in g])
        c = 25.0 * (max(0.0, g_w - ideal_w * 1.25) / ideal_w) ** 2
        if g_path < ideal_path * 0.35:
            c += 25.0 * ((ideal_path * 0.35 - g_path) / ideal_path) ** 2
        c += 3.0 * ((g_xm - centers[li]) / ideal_w) ** 2
        if b + 1 < M:
            g_xmax, g_xmin = max(s["x_max"] for s in g), min(s["x_min"] for s in g)
            ov = min(g_xmax, objs[b + 1]["x_max"]) - max(g_xmin, objs[b + 1]["x_min"])
            if ov > 0.01:
                c += 18.0 * (ov / ideal_w) + 2.5
        return float(c)

    dp = np.full((k, M), np.inf)
    parent = np.zeros((k, M), dtype=int)
    for b in range(M - k + 1):
        dp[0, b] = cost(0, b, 0)
    for li in range(1, k):
        for b in range(li, M - (k - 1 - li)):
            for a in range(li, b + 1):
                v = dp[li - 1, a - 1] + cost(a, b, li)
                if v < dp[li, b]:
                    dp[li, b], parent[li, b] = v, a
    groups = [[] for _ in range(k)]
    end = M - 1
    for li in range(k - 1, -1, -1):
        start = 0 if li == 0 else int(parent[li, end])
        start = max(li, min(start, end))
        for m in range(start, end + 1):
            groups[li].append(objs[m]["segs"])
        end = start - 1
    return [np.concatenate(g, axis=0) if g else [] for g in groups]


def _augment(segs, rng):
    out = np.array(segs, dtype=np.float32, copy=True)
    sx, sy = rng.uniform(0.88, 1.12, size=2)
    shear = rng.uniform(-0.12, 0.12)
    out[:, :, 0] = out[:, :, 0] * sx + shear * out[:, :, 1]
    out[:, :, 1] *= sy
    return out


def train_ocr(records, feats, ink=None, oof=None, fast=False, seed=42, verbose=True):
    """Letter OCR trained on (a) ground-truth canvas ink (+2 augmentations), (b) the traced cursor
    path under the true, undo-aware pen state, (c) out-of-fold *predicted* ink clusters (what the
    decoder really sees at test time)."""
    X, y = [], []
    rng = np.random.RandomState(seed)
    n_gt = n_trace = n_oof = 0
    for r in records:
        k, code = r["length"], r["code"]
        for li, segs in enumerate(segment_true_strokes_into_k(r["strokes"], k)):
            if len(segs) < 2:
                continue
            lab = ALPHABET.index(code[li])
            X.append(extract_letter_descriptor(segs))
            y.append(lab)
            n_gt += 1
            for _ in range(2):
                X.append(extract_letter_descriptor(_augment(segs, rng)))
                y.append(lab)
        df = feats[r["id"]]
        pts = df[["x", "y"]].values.astype(float)
        segs_all = np.stack([pts[:-1], pts[1:]], axis=1)
        seg = df["segment"].values
        same = seg[:-1] == seg[1:]
        ink_mask, _ = final_ink_mask(df, r)  # v15 fix: undone strokes no longer leak into letters
        pen_seg = ink_mask[:-1] & same
        if pen_seg.sum() >= 3 * k:
            ids, _ = decode_rules(pts, pen_seg.astype(float), same, k, 0.5, rules=True)
            for li in range(k):
                sel = ids == li
                if sel.sum() >= 2:
                    X.append(extract_letter_descriptor(segs_all[sel]))
                    y.append(ALPHABET.index(code[li]))
                    n_trace += 1
        if ink is not None and oof is not None and r["id"] in oof:
            blend, s1 = oof[r["id"]]
            probs = postprocess_ink(df, r, blend, s1, ink["th"], rescue=True)
            sp = (probs[:-1] + probs[1:]) / 2.0
            ids, sp2 = decode_rules(pts, sp, same, k, ink["th"], rules=True)
            for li in range(k):
                sel = ids == li
                if sel.sum() >= 4:
                    X.append(extract_letter_descriptor(segs_all[sel], sp2[sel]))
                    y.append(ALPHABET.index(code[li]))
                    n_oof += 1
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    n_et, n_rf, n_lgb = (200, 150, 120) if fast else (400, 300, 220)
    models = {
        # min_samples_leaf=2 keeps the forests (and the saved model file) a few times smaller
        "et": ExtraTreesClassifier(n_estimators=n_et, max_features="sqrt", min_samples_leaf=2,
                                   class_weight="balanced", random_state=seed, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=n_rf, max_features="sqrt", min_samples_leaf=2,
                                     class_weight="balanced", random_state=seed + 58, n_jobs=-1),
        "lgb": lgb.LGBMClassifier(objective="multiclass", learning_rate=0.04, num_leaves=15, min_child_samples=15,
                                  n_estimators=n_lgb, subsample=0.8, colsample_bytree=0.75, class_weight="balanced",
                                  random_state=seed, n_jobs=-1, verbose=-1),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(C=0.60, class_weight="balanced", max_iter=600,
                                                                 random_state=seed)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(144, 72), alpha=0.45,
                                                             max_iter=380, random_state=seed)),
    }
    for m in models.values():
        m.fit(X, y)
    if verbose:
        log(f"  OCR: {n_gt} true-ink + {n_trace} traced + {n_oof} predicted-ink letters "
            f"({len(y)} samples incl. augmentation), {len(set(y.tolist()))}/17 classes")
    return {"models": models, "weights": dict(OCR_WEIGHTS), "n_samples": int(len(y))}


def ocr_proba(ocr, X):
    """(n, 17) letter probabilities in ALPHABET order, aligned per model by classes_."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    total = np.zeros((len(X), len(ALPHABET)))
    for name, w in ocr["weights"].items():
        m = ocr["models"][name]
        total[:, np.asarray(m.classes_, dtype=int)] += w * m.predict_proba(X)
    total = np.maximum(total, 1e-9)
    return total / total.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Decoder 2: joint segmentation + recognition
# --------------------------------------------------------------------------------------
def _split_at_dips(sp, a, b, th, min_len=3, depth=0.12):
    """Cut a candidate stroke wherever the ink probability dips clearly (a likely pen lift)."""
    if b - a + 1 < 2 * min_len + 1:
        return [(a, b)]
    s = uniform_filter1d(sp[a:b + 1], 3)
    pieces, start = [], 0
    for i in range(min_len, len(s) - min_len):
        if s[i] < th and s[i] <= s[i - 1] and s[i] <= s[i + 1]:
            if min(s[:i].max(), s[i + 1:].max()) - s[i] >= depth and i - start >= min_len:
                pieces.append((a + start, a + i))
                start = i + 1
    pieces.append((a + start, b))
    return pieces


def decode_joint(pts, sp, valid, k, th, ocr, order="time", skip_w=4.0, max_span=8):
    """Over-segment the predicted ink into small primitives, then choose (by dynamic programming)
    how to group consecutive primitives into exactly k letters - or skip a primitive as noise - so
    that the OCR is most confident about the k letters, with light geometric priors.
    order="time": letters are consecutive in writing order (handles overlapping letters);
    order="x"   : letters are consecutive left to right.
    Returns (ids, sp) like decode_rules, or None if there is too little ink."""
    n = len(sp)
    seg_x = (pts[:-1, 0] + pts[1:, 0]) / 2.0
    step = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    segs_all = np.stack([pts[:-1], pts[1:]], axis=1)
    low = max(0.15, th - 0.20)
    m = valid & (sp >= low)
    m = binary_closing(m, structure=np.ones(3)) & valid
    prims = []
    for a, b in _runs(m):
        if b - a + 1 < 3 or sp[a:b + 1].max() < 0.85 * th or step[a:b + 1].sum() < 0.015:
            continue
        prims.extend(_split_at_dips(sp, a, b, th))
    if not prims:
        return None
    hi = valid & (sp >= th)
    xs = seg_x[hi] if hi.sum() >= 5 else seg_x[valid]
    if len(xs) < 2:
        return None
    x0, x1 = np.percentile(xs, 5), np.percentile(xs, 95)
    est_w = max(0.15, x1 - x0) / k
    # split primitives much wider than one letter at their weakest point
    changed = True
    while changed:
        changed = False
        for i, (a, b) in enumerate(prims):
            if b - a + 1 >= 8 and np.ptp(seg_x[a:b + 1]) > 1.4 * est_w:
                lo, hi_ = a + (b - a) // 5, b - (b - a) // 5
                c = lo + int(np.argmin(sp[lo:hi_ + 1]))
                prims[i:i + 1] = [(a, c), (c + 1, b)]
                changed = True
                break
    while len(prims) < k:
        cand = [i for i, (a, b) in enumerate(prims) if b - a + 1 >= 4]
        if not cand:
            return None
        i = max(cand, key=lambda j: prims[j][1] - prims[j][0])
        a, b = prims[i]
        c = (a + b) // 2
        prims[i:i + 1] = [(a, c), (c + 1, b)]

    mass = np.array([float(np.sum(sp[a:b + 1] * step[a:b + 1])) for a, b in prims])
    cxs = np.array([float(np.average(seg_x[a:b + 1], weights=np.maximum(sp[a:b + 1], 1e-6))) for a, b in prims])
    if order == "x":
        perm = np.argsort(cxs, kind="stable")
    else:
        perm = np.argsort([a for a, _ in prims], kind="stable")
    prims = [prims[i] for i in perm]
    mass, cxs = mass[perm], cxs[perm]
    M = len(prims)
    avg_mass = max(1e-6, mass.sum() / k)
    skip = skip_w * mass / avg_mass
    xmins = np.array([seg_x[a:b + 1].min() for a, b in prims])
    xmaxs = np.array([seg_x[a:b + 1].max() for a, b in prims])

    spans = [(i, e) for i in range(M) for e in range(i, min(M, i + max_span))]
    idx_of = {}
    descs = []
    for i, e in spans:
        sel = np.concatenate([np.arange(prims[q][0], prims[q][1] + 1) for q in range(i, e + 1)])
        idx_of[(i, e)] = sel
        descs.append(extract_letter_descriptor(segs_all[sel], sp[sel]))
    P = ocr_proba(ocr, np.array(descs))
    gcost = {}
    for row, (i, e) in enumerate(spans):
        c = -math.log(float(P[row].max()) + 1e-4)
        width = xmaxs[i:e + 1].max() - xmins[i:e + 1].min()
        c += 3.0 * (max(0.0, width - 1.6 * est_w) / est_w) ** 2
        mf = mass[i:e + 1].sum() / avg_mass
        if mf < 0.3:
            c += 4.0 * ((0.3 - mf) / 0.3) ** 2
        gcost[(i, e)] = (c, float(np.average(cxs[i:e + 1], weights=np.maximum(mass[i:e + 1], 1e-9))))

    INF = float("inf")
    dp = np.full((M + 1, k + 1), INF)
    back = {}
    dp[0, 0] = 0.0
    for i in range(M):
        for j in range(k + 1):
            base = dp[i, j]
            if base == INF:
                continue
            if base + skip[i] < dp[i + 1, j]:
                dp[i + 1, j] = base + skip[i]
                back[(i + 1, j)] = ("skip", i, j, None)
            if j < k:
                ideal = x0 + (j + 0.5) * est_w
                for e in range(i, min(M, i + max_span)):
                    c, cx = gcost[(i, e)]
                    v = base + c + 0.35 * ((cx - ideal) / est_w) ** 2
                    if v < dp[e + 1, j + 1]:
                        dp[e + 1, j + 1] = v
                        back[(e + 1, j + 1)] = ("grp", i, j, e)
    if dp[M, k] == INF:
        return None
    ids = np.full(n, -1, dtype=int)
    state = (M, k)
    while state != (0, 0):
        kind, i, j, e = back[state]
        if kind == "grp":
            ids[idx_of[(i, e)]] = j
        state = (i, j)
    return ids, np.array(sp, dtype=float, copy=True)


# --------------------------------------------------------------------------------------
# Decoding a recording end to end
# --------------------------------------------------------------------------------------
DECODERS = {
    "v14":              dict(kind="rules", rules=True, rescue=True, undo="v14"),
    "v14-dip":          dict(kind="rules", rules=True, rescue=True, undo="dip"),
    "rules-lite":       dict(kind="rules", rules=False, rescue=True, undo="dip"),
    "joint-time":       dict(kind="joint", order="time", rescue=False, skip_w=4.0, undo="dip"),
    "joint-time-loose": dict(kind="joint", order="time", rescue=False, skip_w=2.0, undo="dip"),
    "joint-x":          dict(kind="joint", order="x", rescue=False, skip_w=4.0, undo="dip"),
    "auto":             dict(kind="auto", members=("v14-dip", "joint-time")),
}
_JOINT_ERRORS_SHOWN = [0]


def make_context(r, bundle, df_feat=None):
    """Shared per-recording state: features, ink probabilities and geometry."""
    if df_feat is None:
        df_feat = build_features(r)
    raw, s1 = ink_probabilities(df_feat, bundle["ink"])
    pts = df_feat[["x", "y"]].values.astype(float)
    seg = df_feat["segment"].values
    return {"r": r, "df": df_feat, "raw": raw, "s1": s1, "pts": pts,
            "segs": np.stack([pts[:-1], pts[1:]], axis=1), "same": seg[:-1] == seg[1:],
            "after": df_feat["after_clear"].values[:-1] == 1, "probs": {}, "results": {}}


def _ink(ctx, bundle, rescue, undo):
    key = (rescue, undo)
    if key not in ctx["probs"]:
        ctx["probs"][key] = postprocess_ink(ctx["df"], ctx["r"], ctx["raw"], ctx["s1"],
                                            bundle["ink"]["th"], rescue=rescue, undo=undo)
    return ctx["probs"][key]


def read_letters(ctx, bundle, ids, sp, k):
    """OCR every letter cluster. Empty clusters fall back to an equal time-slice of the ink, so the
    answer is always k valid letters (v14 could print '?', which the game input rejects)."""
    valid = ctx["same"] & ctx["after"]
    descs, fallback = [], []
    for li in range(k):
        sel = np.where(ids == li)[0]
        fb = len(sel) < 2
        if fb:
            pool = np.where(valid & (sp >= 0.2))[0]
            if len(pool) < 2 * k:
                pool = np.where(valid)[0] if valid.sum() >= 2 * k else np.arange(len(sp))
            sel = np.array_split(pool, k)[li]
            w = np.full(len(sel), 0.5)
        else:
            w = sp[sel]
        descs.append(extract_letter_descriptor(ctx["segs"][sel], w))
        fallback.append(fb)
    P = ocr_proba(bundle["ocr"], np.array(descs))
    letters = []
    for li in range(k):
        top = np.argsort(P[li])[::-1][:3]
        letters.append({"top": [(ALPHABET[i], float(P[li, i])) for i in top], "fallback": fallback[li]})
    p1 = np.array([l["top"][0][1] for l in letters])
    return {"code": "".join(l["top"][0][0] for l in letters), "letters": letters,
            "conf": float(np.mean(p1)), "score": float(np.mean(np.log(p1))), "ids": ids, "sp": sp}


def run_decoder(ctx, bundle, name):
    if name in ctx["results"]:
        return ctx["results"][name]
    cfg = DECODERS[name]
    k = ctx["r"]["length"]
    if cfg["kind"] == "auto":
        cands = [run_decoder(ctx, bundle, m) for m in cfg["members"]]
        best = max(cands, key=lambda c: c["score"])
        res = dict(best, config=name, picked=best["config"])
    else:
        th = bundle["ink"]["th"]
        probs = _ink(ctx, bundle, cfg["rescue"], cfg.get("undo", "v14"))
        sp = (probs[:-1] + probs[1:]) / 2.0
        out, note = None, ""
        if cfg["kind"] == "joint":
            try:
                out = decode_joint(ctx["pts"], sp, ctx["same"] & ctx["after"], k, th, bundle["ocr"],
                                   order=cfg["order"], skip_w=cfg["skip_w"])
                if out is None:
                    note = "joint: too little ink, used v14 segmentation"
            except Exception:
                note = "joint decoder error, used v14 segmentation"
                if _JOINT_ERRORS_SHOWN[0] < 1:
                    _JOINT_ERRORS_SHOWN[0] += 1
                    log("WARNING (shown once): joint decoder failed:\n" + traceback.format_exc())
        if out is None:
            out = decode_rules(ctx["pts"], sp, ctx["same"], k, th, rules=cfg.get("rules", True))
        res = read_letters(ctx, bundle, out[0], out[1], k)
        res.update(config=name, note=note)
    ctx["results"][name] = res
    return res


# --------------------------------------------------------------------------------------
# Training a full bundle, saving and loading
# --------------------------------------------------------------------------------------
def _canvas_key(r):
    c = r.get("canvas") if isinstance(r.get("canvas"), dict) else {}
    return f"{c.get('width')}x{c.get('height')}:{c.get('screen')}"


def make_groups(records, mode="canvas", min_groups=6, verbose=False):
    """Cross-validation groups. 'canvas' groups recordings by canvas size + orientation, a rough
    proxy for "same phone / same writer" (each student drew several codes), so held-out scores are
    not inflated by having seen the same writer. Falls back to one group per recording."""
    if mode == "canvas":
        keys = {r["id"]: _canvas_key(r) for r in records}
        n = len(set(keys.values()))
        if n >= min_groups:
            if verbose:
                log(f"  CV groups: {n} distinct canvases (device proxy) over {len(records)} recordings")
            return keys
        if verbose:
            log(f"  CV groups: only {n} distinct canvases, grouping by recording instead")
    return {r["id"]: r["id"] for r in records}


def package_versions():
    import sklearn
    import scipy
    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "lightgbm": lgb.__version__}


def train_bundle(records, fast=False, group_mode="canvas", feats=None, verbose=True):
    t0 = time.time()
    feats = {} if feats is None else feats
    for r in records:
        if r["id"] not in feats:
            feats[r["id"]] = build_features(r)
    frame = pd.concat([feats[r["id"]] for r in records], ignore_index=True)
    feat_cols = feature_columns(frame)
    gmap = make_groups(records, group_mode, verbose=verbose)
    row_groups = frame["rec_id"].map(gmap).values
    ink, oof = fit_ink_model(frame, feat_cols, row_groups, n_folds=3 if fast else 5, fast=fast, verbose=verbose)
    del frame
    ocr = train_ocr(records, feats, ink, oof, fast=fast, verbose=verbose)
    return {"version": VERSION, "alphabet": ALPHABET, "ink": ink, "ocr": ocr,
            "train_ids": sorted(r["id"] for r in records), "n_records": len(records),
            "data_sha": data_fingerprint(records), "fast": bool(fast), "group_mode": group_mode,
            "packages": package_versions(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_seconds": round(time.time() - t0, 1)}


def save_bundle(bundle, path=DEFAULT_MODEL):
    import joblib
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    meta = {k: v for k, v in bundle.items() if k not in ("ink", "ocr", "train_ids")}
    meta["ink_metrics"] = {k: bundle["ink"][k] for k in ("auc_s1", "auc_s2", "f1", "th", "alpha")}
    with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_bundle(path=DEFAULT_MODEL, verbose=True):
    """Only load model files you made yourself or got from someone you trust: joblib/pickle files can
    run code when loaded. Data files (.jsonl) are always safe to share."""
    import joblib
    bundle = joblib.load(path)
    mismatch = {k: (v, bundle.get("packages", {}).get(k)) for k, v in package_versions().items()
                if bundle.get("packages", {}).get(k) not in (None, v)}
    if mismatch and verbose:
        log("WARNING: model was saved with different package versions "
            + ", ".join(f"{k} {old}->{new}" for k, (new, old) in mismatch.items())
            + ". If it misbehaves, retrain with train.py.")
    return bundle


def default_decoder():
    if os.path.exists(BEST_DECODER_FILE):
        try:
            with open(BEST_DECODER_FILE, encoding="utf-8") as f:
                name = json.load(f).get("decoder")
            if name in DECODERS:
                return name
        except Exception:
            pass
    return DEFAULT_DECODER


# --------------------------------------------------------------------------------------
# Picture: trajectory | true ink (if known) | predicted letters | one crop per letter
# --------------------------------------------------------------------------------------
PALETTE = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed"]


def render_sheet(items, out_path, title=None, dpi=120):
    """items: list of (record, ctx, result, extra_title)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not items:
        return
    has_gt = any("strokes" in r for r, *_ in items)
    off = 1 if has_gt else 0
    n_cols = 2 + off + 5
    fig, axes = plt.subplots(len(items), n_cols, figsize=(2.6 * n_cols, 2.5 * len(items)), squeeze=False)
    for row, (r, ctx, res, extra) in enumerate(items):
        segs, same = ctx["segs"], ctx["same"]
        ids, sp = res["ids"], res["sp"]
        k = r["length"]
        ax = axes[row, 0]
        lc = LineCollection(segs[same], cmap="Spectral", linewidths=1.2, alpha=0.75)
        lc.set_array(np.linspace(0, 1, int(same.sum())))
        ax.add_collection(lc)
        ax.set_title(f"#{row + 1} {r['id']} ({k}L){extra or ''}", fontsize=8, fontweight="bold")
        if has_gt:
            ax = axes[row, 1]
            for st in r.get("strokes", []):
                if isinstance(st, list) and len(st) > 1:
                    a = np.array([p[:2] for p in st])
                    ax.plot(a[:, 0], a[:, 1], color="#111827", linewidth=2.4, solid_capstyle="round")
            ax.set_title(f"True: {r.get('code', '')}", fontsize=8, fontweight="bold", color="#047857")
        ax = axes[row, 1 + off]
        ax.add_collection(LineCollection(segs[same & (ids == -1)], colors="#cbd5e1", linewidths=0.6, alpha=0.4))
        for li in range(k):
            ax.add_collection(LineCollection(segs[same & (ids == li)], colors=PALETTE[li], linewidths=2.6, alpha=0.9))
        ax.set_title(f"Pred: {res['code']} ({res['config']}, conf {res['conf']:.2f})", fontsize=8, fontweight="bold")
        for li in range(5):
            ax = axes[row, 2 + off + li]
            if li >= k:
                ax.axis("off")
                continue
            mine = same & (ids == li)
            ax.add_collection(LineCollection(segs[same & (ids != -1) & ~mine], colors="#e5e7eb", linewidths=0.7, alpha=0.5))
            ax.add_collection(LineCollection(segs[same & (ids == -1)], colors="#94a3b8", linewidths=0.8, alpha=0.35))
            ax.add_collection(LineCollection(segs[mine], colors="#111827", linewidths=2.8))
            let = res["letters"][li]
            txt = " ".join(f"{c}{int(round(p * 100))}" for c, p in let["top"])
            true = f" [{r['code'][li]}]" if "code" in r else ""
            flag = " FALLBACK" if let["fallback"] else ""
            ax.set_title(f"L{li + 1}: {txt}{true}{flag}", fontsize=8, color=PALETTE[li], fontweight="bold")
        for c in range(n_cols):
            a = axes[row, c]
            a.set_xlim(0, 1)
            a.set_ylim(1, 0)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
