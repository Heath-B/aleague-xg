"""
xG-lite: a pre-shot expected goals model for the A-League Men.

Trains a logistic regression on every shot in shots_all.csv except penalties
and shootouts, using only information available BEFORE the ball is struck.

Outputs:
  xg_model_summary.txt   coefficients, fit statistics, calibration table
  shots_with_xg.csv      every shot with its xG attached
"""

import csv, math, json
import numpy as np

HERE = __file__.rsplit("/", 1)[0]
PITCH_LENGTH = 105.0      # metres
PITCH_WIDTH  = 68.0       # metres
GOAL_WIDTH   = 7.32       # metres
HALF_GOAL    = GOAL_WIDTH / 2

EXCLUDE_SITUATIONS = {"penalty", "shootout"}


# ---------------------------------------------------------------- geometry
def to_metres(x, y):
    """Sofascore x,y are 0-100. x is distance from the goal being attacked,
    y is across the pitch with 50 at the centre."""
    dist_out = x / 100.0 * PITCH_LENGTH      # metres from the goal line
    lateral  = (y - 50.0) / 100.0 * PITCH_WIDTH   # metres from centre, signed
    return dist_out, lateral


def shot_angle(dist_out, lateral):
    """Angle in radians subtended by the goal mouth from the shot location."""
    num = GOAL_WIDTH * dist_out
    den = dist_out**2 + lateral**2 - HALF_GOAL**2
    return math.atan2(num, den)


# ---------------------------------------------------------------- load
def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["situation"] in EXCLUDE_SITUATIONS:
                continue
            if r["round_name"].strip() != "":
                continue        # finals, kept out so the sample matches Part 1
            try:
                x = float(r["x"]); y = float(r["y"])
            except ValueError:
                continue
            dist_out, lateral = to_metres(x, y)
            distance = math.hypot(dist_out, lateral)
            if distance < 0.3:          # guard against degenerate coordinates
                distance = 0.3
            r["_distance"] = distance
            r["_abs_lateral"] = abs(lateral)
            r["_angle"] = shot_angle(dist_out, lateral)
            r["_is_goal"] = 1 if r["shot_type"] == "goal" else 0
            r["_is_header"] = 1 if r["body_part"] == "head" else 0
            rows.append(r)
    return rows


# ---------------------------------------------------------------- features
SITUATIONS = ["assisted", "corner", "fast-break", "free-kick",
              "set-piece", "throw-in-set-piece"]      # 'regular' is the baseline


def design_matrix(rows):
    """Specification chosen by comparing cross-validated log loss and
    Hosmer-Lemeshow calibration across nine candidate models. Distance enters
    both logged and linear, and angle interacts with logged distance, because
    a single log-distance term left the middle of the range miscalibrated."""
    names = ["intercept", "log_distance", "distance", "angle",
             "angle_x_logdist", "is_header"]
    names += [f"sit_{s}" for s in SITUATIONS]
    X = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        ld = math.log(r["_distance"])
        col = [1.0,
               ld,
               r["_distance"],
               r["_angle"],
               r["_angle"] * ld,
               float(r["_is_header"])]
        col += [1.0 if r["situation"] == s else 0.0 for s in SITUATIONS]
        X[i] = col
    y = np.array([r["_is_goal"] for r in rows], dtype=float)
    return X, y, names


# ---------------------------------------------------------------- fitting
def fit_logistic(X, y, max_iter=100, tol=1e-9):
    """Iteratively reweighted least squares. Returns coefficients and their
    standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        z = eta + (y - p) / W
        XtW = X.T * W
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv((X.T * W) @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se


def predict(X, beta):
    return 1.0 / (1.0 + np.exp(-(X @ beta)))


def norm_sf(z):
    """Two-sided p-value from a z statistic, without scipy."""
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------------------------------------------------------- metrics
def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def cross_validate(X, y, folds=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    chunks = np.array_split(idx, folds)
    oof = np.zeros(len(y))
    for k in range(folds):
        test = chunks[k]
        train = np.concatenate([chunks[j] for j in range(folds) if j != k])
        b, _ = fit_logistic(X[train], y[train])
        oof[test] = predict(X[test], b)
    return oof


def calibration_table(y, p, bins=10):
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    out = []
    for i, g in enumerate(groups, 1):
        out.append((i, len(g), float(p[g].mean()), float(y[g].mean()),
                    float(p[g].sum()), int(y[g].sum())))
    return out


# ---------------------------------------------------------------- main
def main():
    rows = load(f"{HERE}/shots_all.csv")
    X, y, names = design_matrix(rows)
    print(f"shots used: {len(rows)}, goals: {int(y.sum())} "
          f"({100*y.mean():.2f}%)")

    beta, se = fit_logistic(X, y)
    insample = predict(X, beta)
    oof = cross_validate(X, y)

    lines = []
    lines.append("xG-lite, pre-shot expected goals, A-League Men 2023/24 to 2025/26")
    lines.append("=" * 68)
    lines.append(f"Shots: {len(rows)}   Goals: {int(y.sum())} ({100*y.mean():.2f}%)")
    lines.append("Regular seasons only. Penalties and shootouts excluded. 'regular' play is the baseline situation.")
    lines.append("")
    lines.append(f"{'term':<26}{'coef':>9}{'se':>8}{'z':>8}{'p':>9}{'odds':>8}")
    lines.append("-" * 68)
    for nm, b, s in zip(names, beta, se):
        z = b / s
        pv = norm_sf(z)
        ps = "<.001" if pv < .001 else f"{pv:.3f}".lstrip("0")
        lines.append(f"{nm:<26}{b:>9.3f}{s:>8.3f}{z:>8.2f}{ps:>9}{math.exp(b):>8.2f}")
    lines.append("")
    null = log_loss(y, np.full_like(y, y.mean()))
    lines.append("Fit")
    lines.append(f"  log loss, null model      {null:.5f}")
    lines.append(f"  log loss, in sample       {log_loss(y, insample):.5f}")
    lines.append(f"  log loss, cross-validated {log_loss(y, oof):.5f}")
    lines.append(f"  pseudo R2 (McFadden, CV)  {1 - log_loss(y, oof)/null:.4f}")
    lines.append(f"  Brier, cross-validated    {brier(y, oof):.5f}")
    lines.append(f"  total xG {oof.sum():.1f} against {int(y.sum())} actual goals")
    lines.append("")
    lines.append("Calibration, cross-validated predictions in deciles")
    lines.append(f"{'bin':>4}{'n':>7}{'mean xG':>10}{'actual':>9}{'xG sum':>9}{'goals':>7}")
    lines.append("-" * 46)
    for i, n, mp, my, sp, sy in calibration_table(y, oof):
        lines.append(f"{i:>4}{n:>7}{mp:>10.4f}{my:>9.4f}{sp:>9.1f}{sy:>7}")

    report = "\n".join(lines)
    print(report)
    with open(f"{HERE}/xg_model_summary.txt", "w") as fh:
        fh.write(report + "\n")

    # attach xG to every shot and save
    with open(f"{HERE}/shots_with_xg.csv", "w", newline="") as fh:
        cols = [c for c in rows[0] if not c.startswith("_")]
        w = csv.DictWriter(fh, fieldnames=cols + ["distance_m", "angle_rad", "xg_lite"])
        w.writeheader()
        for r, p in zip(rows, oof):
            out = {c: r[c] for c in cols}
            out["distance_m"] = round(r["_distance"], 2)
            out["angle_rad"] = round(r["_angle"], 4)
            out["xg_lite"] = round(float(p), 5)
            w.writerow(out)

    with open(f"{HERE}/xg_model_coefficients.json", "w") as fh:
        json.dump({"terms": names, "coef": list(map(float, beta)),
                   "se": list(map(float, se))}, fh, indent=2)


if __name__ == "__main__":
    main()
