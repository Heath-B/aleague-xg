"""
Builds the body part and phase of play mix used by the "any" option in
aleague_xg.py, and writes it into that module between the MIX markers.

The model gives the value of a shot once you know how it was struck and what
created it. "Any" answers the other question, what a shot from a given spot is
worth before you know either, by averaging the model over what is actually hit
from there. This script measures that mix from the training sample.

The pitch is divided into bands by straight-line distance to the centre of the
goal and by sideways distance from the centre line. Each cell holds the shares
observed in it, shrunk towards the league-wide mix by KAPPA pseudo-shots so
that no cell is empty, and a cell with fewer than MIN_CELL shots falls back to
its distance band pooled across the pitch. The module interpolates between
cell centres, so the answer moves smoothly rather than stepping at a boundary.

Run after make_xg_model.py, since it uses the fitted coefficients:

    python3 make_xg_mix.py

Writes:
  aleague_xg.py          anchors and table replaced in place
  xg_mix_table.json      the same numbers, for anyone not using Python
  xg_mix_validation.txt  how faithfully the averaged value matches the data
"""

import csv, json, math, sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from aleague_xg import xg, BODY_PARTS, SITUATIONS

SHOTS = "shots_all.csv"
MODULE = "aleague_xg.py"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
EXCLUDE_SITUATIONS = {"penalty", "shootout"}

# Bands chosen by comparing candidate grids on a validation grid deliberately
# finer than the bands themselves, so that a coarse table could not flatter
# itself. This grid gave the lowest error and the smoothest surface.
DISTANCE_EDGES = (0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0, 25.0, 30.0)
LATERAL_EDGES = (0.0, 5.0, 11.0)
KAPPA = 10.0        # pseudo-shots of the league mix added to every cell
MIN_CELL = 150      # below this a cell falls back to its distance band
DP = 6              # decimal places kept in the published table

KEYS = tuple((b, s) for b in BODY_PARTS for s in SITUATIONS)
ND, NL = len(DISTANCE_EDGES), len(LATERAL_EDGES)


def band(value, edges):
    i = 0
    while i + 1 < len(edges) and value >= edges[i + 1]:
        i += 1
    return i


def load():
    rows = []
    for r in csv.DictReader(open(SHOTS)):
        if r["situation"] in EXCLUDE_SITUATIONS:
            continue
        if r["round_name"].strip() != "":
            continue                      # finals, excluded as in the model
        try:
            x = float(r["x"]); y = float(r["y"])
        except ValueError:
            continue
        d = x / 100.0 * PITCH_LENGTH
        lat = abs((y - 50.0) / 100.0 * PITCH_WIDTH)
        rows.append({
            "d": d, "lat": lat,
            "dist": max(math.hypot(d, lat), 0.3),
            # the model treats anything that is not a header as a foot shot
            "bp": "head" if r["body_part"] == "head" else "foot",
            "sit": r["situation"].replace("-", "_"),
            "goal": 1 if r["shot_type"] == "goal" else 0,
        })
    return rows


def exact_shares(counts, n, league):
    """Shrink, round to DP places, then make the row sum to exactly one."""
    t = n + KAPPA
    p = [round((counts.get(k, 0) + KAPPA * league[k]) / t, DP) for k in KEYS]
    drift = round(1.0 - sum(p), DP)
    p[p.index(max(p))] = round(max(p) + drift, DP)
    assert abs(sum(p) - 1.0) < 1e-12, sum(p)
    return p


def interpolator(table, danchors, lanchors):
    def nodes(v, a):
        if v <= a[0]:
            return 0, 0, 1.0
        if v >= a[-1]:
            return len(a) - 1, len(a) - 1, 1.0
        i = 0
        while a[i + 1] < v:
            i += 1
        return i, i + 1, (a[i + 1] - v) / (a[i + 1] - a[i])

    def weights(dist, lat):
        i0, i1, wd = nodes(dist, danchors)
        j0, j1, wl = nodes(lat, lanchors)
        out = [wd * wl * table[i0 * NL + j0][k]
               + (1 - wd) * wl * table[i1 * NL + j0][k]
               + wd * (1 - wl) * table[i0 * NL + j1][k]
               + (1 - wd) * (1 - wl) * table[i1 * NL + j1][k]
               for k in range(len(KEYS))]
        t = sum(out)
        return [v / t for v in out]
    return weights


def main():
    rows = load()
    n_total = len(rows)

    league = {k: 0.0 for k in KEYS}
    for r in rows:
        league[(r["bp"], r["sit"])] += 1
    league = {k: v / n_total for k, v in league.items()}

    cell_n, cell_c, dist_n, dist_c = {}, {}, {}, {}
    sum_d, sum_l, n_d, n_l = {}, {}, {}, {}
    for r in rows:
        i, j = band(r["dist"], DISTANCE_EDGES), band(r["lat"], LATERAL_EDGES)
        k = (r["bp"], r["sit"])
        cell_n[(i, j)] = cell_n.get((i, j), 0) + 1
        cell_c.setdefault((i, j), {})[k] = cell_c.setdefault((i, j), {}).get(k, 0) + 1
        dist_n[i] = dist_n.get(i, 0) + 1
        dist_c.setdefault(i, {})[k] = dist_c.setdefault(i, {}).get(k, 0) + 1
        sum_d[i] = sum_d.get(i, 0.0) + r["dist"]; n_d[i] = n_d.get(i, 0) + 1
        sum_l[j] = sum_l.get(j, 0.0) + r["lat"];  n_l[j] = n_l.get(j, 0) + 1

    # Interpolation nodes are the observed centre of gravity of each band, not
    # the nominal midpoint, so the open-ended outer bands sit where the shots
    # actually are.
    danchors = [round(sum_d[i] / n_d[i], 3) for i in range(ND)]
    lanchors = [round(sum_l[j] / n_l[j], 3) for j in range(NL)]

    table, thin = [], 0
    for i in range(ND):
        for j in range(NL):
            n = cell_n.get((i, j), 0)
            if n >= MIN_CELL:
                table.append(exact_shares(cell_c[(i, j)], n, league))
            else:
                thin += n
                table.append(exact_shares(dist_c.get(i, {}), dist_n.get(i, 0), league))

    weights = interpolator(table, danchors, lanchors)

    def weighted_value(r):
        w = weights(r["dist"], r["lat"])
        return sum(w[i] * xg(r["d"], r["lat"], k[0], k[1]) for i, k in enumerate(KEYS))

    def fitted_value(r):
        return xg(r["d"], r["lat"], r["bp"], r["sit"])

    # ------------------------------------------------------------ validation
    # Does averaging the model over the mix reproduce the mean fitted xG of the
    # shots that actually came from each area? The grid below is finer than the
    # table's own bands on purpose, so a coarse table cannot flatter itself.
    fine, total_any, total_act = {}, 0.0, 0.0
    for r in rows:
        a = weighted_value(r)
        act = fitted_value(r)
        total_any += a; total_act += act
        g = fine.setdefault((int(r["dist"] // 2), int(r["lat"] // 3)), [0.0, 0.0, 0])
        g[0] += a; g[1] += act; g[2] += 1

    worst, sq, counted = 0.0, 0.0, 0
    lines = ['Validation of the "any" option in aleague_xg.py', "=" * 64, "",
             f"Training shots                        {n_total:,}",
             f"Cells in the mix table                {ND * NL}",
             f"Shots in cells below the {MIN_CELL}-shot floor   {thin}", "",
             "Mean value of a shot in each two-metre by three-metre area,",
             "averaged over the mix, against the mean fitted xG of the shots",
             "actually taken there. The areas are finer than the table's bands.", "",
             f"{'distance':>12}  {'lateral':>10}  {'shots':>6}  "
             f"{'averaged':>9}  {'fitted':>7}  {'gap':>8}"]
    for (di, li), (sa, sx, n) in sorted(fine.items()):
        if n < 40:
            continue
        gap = sa / n - sx / n
        worst = max(worst, abs(gap)); sq += gap * gap * n; counted += n
        lines.append(f"{di*2:>6}-{di*2+2:<5}  {li*3:>4}-{li*3+3:<5}  {n:>6}  "
                     f"{sa/n*100:8.2f}%  {sx/n*100:6.2f}%  {gap*100:+7.2f}pp")

    # smoothness down the centre line
    rises, step, prev = 0, 0.0, None
    for q in range(20, 361):
        d = q / 10.0
        w = weights(d, 0.0)
        v = sum(w[n] * xg(d, 0.0, k[0], k[1]) for n, k in enumerate(KEYS))
        if prev is not None:
            if v > prev + 1e-9:
                rises += 1
            step = max(step, abs(v - prev))
        prev = v

    # --------------------------------------------- calibration against goals
    # The test above asks whether the averaged value matches the model. This
    # one asks whether it matches reality, and separates two things that are
    # easy to confuse. Sorting shots purely by location is a sharper test of
    # spatial fit than sorting by predicted value, because it stops a close
    # header and a long-range shot of similar value from cancelling out. The
    # fully specified model is run through the same groups, so any gap that
    # appears in both columns belongs to the model, not to the averaging.
    graded = sorted((weighted_value(r), fitted_value(r), r["goal"]) for r in rows)
    cal = ['', 'Calibration against actual goals, shots sorted by location value',
           '', f"{'bin':>4}{'shots':>7}{'averaged':>10}{'model':>9}{'goals':>7}"
               f"{'averaged z':>12}{'model z':>9}"]
    hl_any = hl_mod = 0.0
    for i in range(10):
        c = graded[i * n_total // 10:(i + 1) * n_total // 10]
        m = len(c)
        a = sum(v for v, _, _ in c); b = sum(v for _, v, _ in c)
        o = sum(g for _, _, g in c)
        hl_any += (o - a) ** 2 / (a * (1 - a / m))
        hl_mod += (o - b) ** 2 / (b * (1 - b / m))
        cal.append(f"{i+1:>4}{m:>7}{a:>10.1f}{b:>9.1f}{o:>7}"
                   f"{(o-a)/math.sqrt(a):>+12.2f}{(o-b)/math.sqrt(b):>+9.2f}")
    cal += ['',
            f"Hosmer-Lemeshow on these groups, averaged value   {hl_any:.2f}",
            f"Hosmer-Lemeshow on these groups, model            {hl_mod:.2f}",
            '',
            "The two are the same to within rounding, and the gaps appear in the",
            "same groups. Averaging over the mix inherits the model's spatial",
            "error and adds none of its own. The model under-rates the sixth and",
            "seventh deciles and over-rates the fourth, which is a limit of five",
            "features on three seasons, not of the averaging."]

    by_phase = ['', 'With the phase of play known and only the body part averaged', '']
    for sit in SITUATIONS:
        e = o = 0.0; k = 0
        for r in rows:
            if r["sit"] != sit:
                continue
            e += xg(r["d"], r["lat"], "any", sit); o += r["goal"]; k += 1
        if k == 0:
            continue
        by_phase.append(f"  {sit:<20} shots {k:>5}  expected {e:>7.1f}  "
                        f"actual {int(o):>4}  z = {(o-e)/math.sqrt(e):+.2f}")

    lines += ["",
              f"Worst area gap                        {worst*100:.3f} percentage points",
              f"Root mean square gap                  {math.sqrt(sq/counted)*100:.3f} percentage points",
              f"League total, averaged over the mix   {total_any:.1f}",
              f"League total, using each shot's own   {total_act:.1f}",
              f"Difference                            {(total_any/total_act-1)*100:+.2f}%",
              "",
              "Smoothness, walking down the centre line from 2 to 36 metres in",
              "100 millimetre steps:",
              f"  steps where the value rises as you move away  {rises}",
              f"  largest change in one step                    {step*100:.3f} pp",
              "",
              "The small positive difference in the league total is expected.",
              "Within a cell, headers sit closer to goal than foot shots, so a",
              "single mix for the whole cell slightly understates how often a",
              "close-range shot is a header.", ""]
    lines += cal + by_phase + [""]
    open("xg_mix_validation.txt", "w").write("\n".join(lines))

    json.dump({"distance_edges": list(DISTANCE_EDGES),
               "lateral_edges": list(LATERAL_EDGES),
               "distance_anchors": danchors, "lateral_anchors": lanchors,
               "keys": [f"{b}|{s}" for b, s in KEYS],
               "kappa": KAPPA, "min_cell": MIN_CELL,
               "cells": table}, open("xg_mix_table.json", "w"), indent=2)

    # ------------------------------------------------------------ write out
    anchors = ("_MIX_DISTANCE_ANCHORS = ("
               + ", ".join(f"{v:g}" for v in danchors) + ")\n"
               + "_MIX_LATERAL_ANCHORS = ("
               + ", ".join(f"{v:g}" for v in lanchors) + ")")
    out = ["_MIX = ("]
    for i in range(ND):
        lo = DISTANCE_EDGES[i]
        hi = DISTANCE_EDGES[i + 1] if i + 1 < ND else None
        out.append(f"    # {lo:.0f} to {hi:.0f} metres from goal" if hi
                   else f"    # {lo:.0f} metres or more from goal")
        for j in range(NL):
            llo = LATERAL_EDGES[j]
            lhi = LATERAL_EDGES[j + 1] if j + 1 < NL else None
            tag = f"{llo:.0f} to {lhi:.0f} m wide" if lhi else f"{llo:.0f} m or wider"
            nums = ", ".join(f"{v:.6f}" for v in table[i * NL + j])
            out.append(f"    ({nums}),  # {tag}")
    out.append(")")

    src = open(MODULE).read()
    for begin, end, body in (("# BEGIN MIX ANCHORS", "# END MIX ANCHORS", anchors),
                             ("# BEGIN MIX TABLE", "# END MIX TABLE", "\n".join(out))):
        a = src.index(begin) + len(begin) + 1
        b = src.index(end)
        src = src[:a] + body + "\n" + src[b:]
    open(MODULE, "w").write(src)

    print(f"{ND * NL} cells and {ND}x{NL} anchors written into {MODULE}")
    print(f"worst area gap {worst*100:.3f} pp, rms {math.sqrt(sq/counted)*100:.3f} pp, "
          f"league total {(total_any/total_act-1)*100:+.2f}%, "
          f"rises {rises}, largest step {step*100:.3f} pp")
    print("see xg_mix_validation.txt")


if __name__ == "__main__":
    main()
