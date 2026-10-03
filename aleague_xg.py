"""
aleague-xg v1.0
A free expected goals model for the A-League Men.

No dependencies. Python 3.8 or later.

    from aleague_xg import xg

    xg(from_goal_line_m=8.0, from_centre_m=3.0)                  -> 0.286
    xg(8.0, 3.0, body_part="head", situation="corner")           -> 0.073
    xg(8.0, 3.0, body_part="any", situation="any")               -> 0.184

Pass "any" for the body part, the phase of play, or both, and you get the
league-average shot from that spot, weighted by what actually happens there.

Trained on 14,065 shots from the 2023/24, 2024/25 and 2025/26 A-League Men
regular seasons, of which 1,463 were goals. Penalties, shootouts and finals
are excluded. See README.md for how it was built, how well it performs, and
what it cannot see.

Licence: MIT. Use it for anything, credit appreciated.
"""

from math import atan2, exp, hypot, log

__version__ = "1.0.0"
MODEL_SEASONS = "A-League Men 2023/24 to 2025/26"
TRAINING_SHOTS = 14065
TRAINING_GOALS = 1463

GOAL_WIDTH_M = 7.32
_HALF_GOAL = GOAL_WIDTH_M / 2

# Fitted coefficients. Do not edit by hand, they come from make_xg_model.py.
_B = {
    "intercept":        -2.7357013388210047,
    "log_distance":      0.07238694092396257,
    "distance":         -0.06311071862659293,
    "angle":             1.7218888931335286,
    "angle_x_logdist":   0.5208011785010307,
    "is_header":        -0.8877564681599229,
}

# Effect of the phase of play, relative to a regular open-play shot.
_SITUATION = {
    "regular":             0.0,
    "assisted":            0.05518385548497039,
    "corner":             -0.7417340238559987,
    "fast_break":          0.6030755805660073,
    "free_kick":           0.5116518081041089,
    "set_piece":          -0.33174497269662423,
    "throw_in_set_piece": -0.6904611161615137,
}

BODY_PARTS = ("foot", "head")
SITUATIONS = tuple(_SITUATION)
ANY = "any"

# Observed mix of body part and phase of play by area of the pitch, used only
# when body_part or situation is "any". The pitch is divided into bands by
# straight-line distance to the centre of the goal and by sideways distance
# from the centre line, and each cell holds the shares observed in the
# training sample, lightly shrunk towards the league-wide mix so that no cell
# is empty. A cell with fewer than 150 shots falls back to its distance band
# pooled across the pitch. Values are interpolated between the centres of
# neighbouring cells, so the answer moves smoothly rather than stepping at a
# boundary. Built by make_xg_mix.py. Do not edit by hand.
# BEGIN MIX ANCHORS
_MIX_DISTANCE_ANCHORS = (4.581, 7.12, 9.018, 10.96, 13.06, 15.033, 16.913, 19.476, 23.072, 27.312, 34.886)
_MIX_LATERAL_ANCHORS = (2.354, 7.967, 13.826)
# END MIX ANCHORS
_MIX_KEYS = tuple((b, s) for b in BODY_PARTS for s in SITUATIONS)
# BEGIN MIX TABLE
_MIX = (
    # 0 to 6 metres from goal
    (0.134537, 0.197435, 0.136659, 0.030818, 0.000435, 0.033170, 0.016606, 0.026875, 0.099063, 0.262986, 0.003001, 0.000000, 0.049425, 0.008990),  # 0 to 5 m wide
    (0.139521, 0.196348, 0.135796, 0.029970, 0.000423, 0.032258, 0.017597, 0.026136, 0.100680, 0.261544, 0.002918, 0.000000, 0.048066, 0.008743),  # 5 to 11 m wide
    (0.139521, 0.196348, 0.135796, 0.029970, 0.000423, 0.032258, 0.017597, 0.026136, 0.100680, 0.261544, 0.002918, 0.000000, 0.048066, 0.008743),  # 11 m or wider
    # 6 to 8 metres from goal
    (0.094156, 0.182557, 0.072957, 0.034267, 0.000535, 0.028004, 0.011281, 0.033076, 0.208004, 0.265070, 0.001861, 0.000000, 0.058999, 0.009233),  # 0 to 5 m wide
    (0.128424, 0.212560, 0.082248, 0.024933, 0.001276, 0.036201, 0.005062, 0.013361, 0.116027, 0.326323, 0.004438, 0.000000, 0.040233, 0.008914),  # 5 to 11 m wide
    (0.103803, 0.187812, 0.075601, 0.030993, 0.000382, 0.030445, 0.009359, 0.027529, 0.182445, 0.285917, 0.002636, 0.000000, 0.053874, 0.009204),  # 11 m or wider
    # 8 to 10 metres from goal
    (0.072289, 0.237861, 0.067592, 0.036663, 0.000300, 0.019805, 0.005297, 0.013408, 0.219271, 0.233807, 0.009257, 0.000000, 0.069007, 0.015443),  # 0 to 5 m wide
    (0.092380, 0.278459, 0.087749, 0.049146, 0.000859, 0.030265, 0.009292, 0.006058, 0.110501, 0.290377, 0.002989, 0.000000, 0.038863, 0.003062),  # 5 to 11 m wide
    (0.077001, 0.246684, 0.072726, 0.039655, 0.000224, 0.022462, 0.006257, 0.011549, 0.192155, 0.249791, 0.007681, 0.000000, 0.061513, 0.012302),  # 11 m or wider
    # 10 to 12 metres from goal
    (0.073907, 0.292635, 0.069977, 0.040778, 0.000446, 0.017237, 0.009403, 0.015358, 0.216138, 0.176684, 0.007659, 0.000000, 0.067501, 0.012277),  # 0 to 5 m wide
    (0.130638, 0.425728, 0.065571, 0.063391, 0.000566, 0.014128, 0.006123, 0.003992, 0.094128, 0.137070, 0.000032, 0.000000, 0.043049, 0.015584),  # 5 to 11 m wide
    (0.098236, 0.354157, 0.066527, 0.052076, 0.000247, 0.019654, 0.007729, 0.010177, 0.159975, 0.157576, 0.004233, 0.000000, 0.055876, 0.013537),  # 11 m or wider
    # 12 to 14 metres from goal
    (0.108433, 0.404452, 0.066213, 0.075753, 0.000572, 0.024051, 0.027709, 0.011859, 0.142016, 0.077746, 0.003946, 0.000000, 0.053255, 0.003995),  # 0 to 5 m wide
    (0.136782, 0.488376, 0.062196, 0.113437, 0.000537, 0.028107, 0.020513, 0.003786, 0.041489, 0.065677, 0.001868, 0.000000, 0.033480, 0.003752),  # 5 to 11 m wide
    (0.121575, 0.644384, 0.047056, 0.089519, 0.001398, 0.054020, 0.010331, 0.000286, 0.017082, 0.013053, 0.000078, 0.000000, 0.001021, 0.000197),  # 11 m or wider
    # 14 to 16 metres from goal
    (0.167377, 0.476868, 0.093463, 0.082829, 0.000507, 0.042170, 0.019374, 0.005312, 0.047865, 0.032514, 0.001765, 0.000000, 0.024676, 0.005280),  # 0 to 5 m wide
    (0.129116, 0.567612, 0.086431, 0.120293, 0.000452, 0.017477, 0.014178, 0.001640, 0.022555, 0.021251, 0.000025, 0.000000, 0.011166, 0.007804),  # 5 to 11 m wide
    (0.114752, 0.632241, 0.046084, 0.146536, 0.000755, 0.018837, 0.013331, 0.002738, 0.011809, 0.007049, 0.000042, 0.000000, 0.005719, 0.000107),  # 11 m or wider
    # 16 to 18 metres from goal
    (0.212071, 0.520657, 0.093093, 0.055885, 0.000977, 0.044448, 0.027288, 0.003544, 0.015285, 0.012468, 0.000055, 0.000000, 0.014091, 0.000138),  # 0 to 5 m wide
    (0.161022, 0.547637, 0.069179, 0.148088, 0.000564, 0.023726, 0.025404, 0.003976, 0.008823, 0.003336, 0.000032, 0.000000, 0.006203, 0.002010),  # 5 to 11 m wide
    (0.147994, 0.660291, 0.050820, 0.110255, 0.000623, 0.015544, 0.004604, 0.000127, 0.003348, 0.001552, 0.000035, 0.000000, 0.004719, 0.000088),  # 11 m or wider
    # 18 to 21 metres from goal
    (0.217764, 0.527220, 0.083732, 0.072440, 0.022650, 0.049588, 0.020855, 0.000095, 0.002488, 0.001154, 0.000026, 0.000000, 0.001923, 0.000065),  # 0 to 5 m wide
    (0.183726, 0.557225, 0.090950, 0.105456, 0.010598, 0.027877, 0.015208, 0.000147, 0.003877, 0.001798, 0.000040, 0.000000, 0.002996, 0.000102),  # 5 to 11 m wide
    (0.117135, 0.635506, 0.067388, 0.127640, 0.000420, 0.030633, 0.016056, 0.000086, 0.002259, 0.001048, 0.000024, 0.000000, 0.001746, 0.000059),  # 11 m or wider
    # 21 to 25 metres from goal
    (0.180722, 0.562886, 0.129023, 0.043223, 0.040115, 0.022566, 0.015700, 0.002324, 0.001250, 0.001597, 0.000036, 0.000000, 0.000468, 0.000090),  # 0 to 5 m wide
    (0.160739, 0.570508, 0.111533, 0.053942, 0.041227, 0.039713, 0.018367, 0.001601, 0.000861, 0.001100, 0.000025, 0.000000, 0.000322, 0.000062),  # 5 to 11 m wide
    (0.151815, 0.596541, 0.080638, 0.073219, 0.040094, 0.022104, 0.029063, 0.000107, 0.002824, 0.003108, 0.000029, 0.000000, 0.000384, 0.000074),  # 11 m or wider
    # 25 to 30 metres from goal
    (0.193718, 0.504545, 0.097045, 0.033919, 0.107714, 0.037611, 0.023118, 0.000085, 0.000816, 0.001042, 0.000023, 0.000000, 0.000305, 0.000059),  # 0 to 5 m wide
    (0.160478, 0.560576, 0.105864, 0.025741, 0.090541, 0.038212, 0.016220, 0.000087, 0.000829, 0.001058, 0.000024, 0.000000, 0.000310, 0.000060),  # 5 to 11 m wide
    (0.130212, 0.556520, 0.112300, 0.043746, 0.118650, 0.023205, 0.013047, 0.000085, 0.000812, 0.001037, 0.000023, 0.000000, 0.000304, 0.000059),  # 11 m or wider
    # 30 metres or more from goal
    (0.166252, 0.450575, 0.095136, 0.108263, 0.120056, 0.037854, 0.014426, 0.000273, 0.002604, 0.003324, 0.000075, 0.000000, 0.000974, 0.000188),  # 0 to 5 m wide
    (0.178834, 0.498364, 0.044620, 0.064190, 0.168142, 0.026718, 0.010905, 0.000302, 0.002880, 0.003677, 0.000083, 0.000000, 0.001077, 0.000208),  # 5 to 11 m wide
    (0.178031, 0.462253, 0.046116, 0.099032, 0.157641, 0.024300, 0.023864, 0.000199, 0.001901, 0.005760, 0.000055, 0.000000, 0.000711, 0.000137),  # 11 m or wider
)
# END MIX TABLE


class UnknownInput(ValueError):
    pass


def _interp_nodes(value, anchors):
    """The two neighbouring anchors, and the weight carried by the first."""
    if value <= anchors[0]:
        return 0, 0, 1.0
    if value >= anchors[-1]:
        return len(anchors) - 1, len(anchors) - 1, 1.0
    i = 0
    while anchors[i + 1] < value:
        i += 1
    return i, i + 1, (anchors[i + 1] - value) / (anchors[i + 1] - anchors[i])


def _mix_weights(distance, abs_lateral):
    """Shares of each body part and phase of play observed around this spot."""
    nl = len(_MIX_LATERAL_ANCHORS)
    i0, i1, wd = _interp_nodes(distance, _MIX_DISTANCE_ANCHORS)
    j0, j1, wl = _interp_nodes(abs_lateral, _MIX_LATERAL_ANCHORS)
    near_near = _MIX[i0 * nl + j0]
    far_near = _MIX[i1 * nl + j0]
    near_wide = _MIX[i0 * nl + j1]
    far_wide = _MIX[i1 * nl + j1]
    blended = [wd * wl * near_near[k] + (1.0 - wd) * wl * far_near[k]
               + wd * (1.0 - wl) * near_wide[k]
               + (1.0 - wd) * (1.0 - wl) * far_wide[k]
               for k in range(len(_MIX_KEYS))]
    total = sum(blended)
    return dict(zip(_MIX_KEYS, (v / total for v in blended)))


def _logistic(z):
    if z >= 0:
        return 1.0 / (1.0 + exp(-z))
    e = exp(z)
    return e / (1.0 + e)


def shot_angle(from_goal_line_m, from_centre_m):
    """Angle in radians subtended by the goal mouth from a point on the pitch.

    Widest directly in front of goal and close in. Narrows as you move back
    or sideways. This is what carries most of the geometry in the model.
    """
    d = float(from_goal_line_m)
    y = abs(float(from_centre_m))
    return atan2(GOAL_WIDTH_M * d, d * d + y * y - _HALF_GOAL * _HALF_GOAL)


def xg(from_goal_line_m, from_centre_m, body_part="foot", situation="regular"):
    """Probability that a shot from this position becomes a goal.

    from_goal_line_m  perpendicular distance from the goal line, in metres.
                      A shot on the six-yard line is about 5.5.
    from_centre_m     sideways distance from the centre of the goal, in
                      metres. Sign is ignored, 4 and -4 give the same answer.
    body_part         "foot", "head", or "any".
    situation         one of SITUATIONS, or "any". "regular" is an open-play
                      shot that was not the end of an assist, counter or set
                      piece.

    Returns a float between 0 and 1.

    "any" averages over what is actually hit from that part of the pitch, so
    "any" close in and central leans towards headers because most shots from
    there are headers. It answers "what is a shot from here worth on average",
    which is a different question from "what is this particular shot worth".
    Use a specific body part and phase of play when you know them.

    Penalties are not covered. The model was trained with them excluded
    because a penalty is a fixed situation that would distort everything
    around it. Use the league penalty conversion rate instead.
    """
    if body_part != ANY and body_part not in BODY_PARTS:
        raise UnknownInput(
            f"body_part must be one of {BODY_PARTS + (ANY,)}, got {body_part!r}")
    if situation != ANY and situation not in _SITUATION:
        raise UnknownInput(
            f"situation must be one of {SITUATIONS + (ANY,)}, got {situation!r}")

    d = float(from_goal_line_m)
    y = abs(float(from_centre_m))
    if d < 0:
        raise UnknownInput("from_goal_line_m cannot be negative")

    distance = hypot(d, y)
    if distance < 0.3:          # guard the singularity on the goal line
        distance = 0.3
    ld = log(distance)
    angle = shot_angle(d, y)

    base = (_B["intercept"]
            + _B["log_distance"] * ld
            + _B["distance"] * distance
            + _B["angle"] * angle
            + _B["angle_x_logdist"] * angle * ld)

    if body_part != ANY and situation != ANY:
        return _logistic(base
                         + _B["is_header"] * (1.0 if body_part == "head" else 0.0)
                         + _SITUATION[situation])

    weights = _mix_weights(distance, y)
    combos = [(b, s)
              for b in (BODY_PARTS if body_part == ANY else (body_part,))
              for s in (SITUATIONS if situation == ANY else (situation,))]
    total = sum(weights[c] for c in combos)
    if total <= 0:              # cannot happen with a shrunk table, belt and braces
        raise UnknownInput("no observed shots to average over for this combination")
    return sum(weights[(b, s)] / total
               * _logistic(base
                           + _B["is_header"] * (1.0 if b == "head" else 0.0)
                           + _SITUATION[s])
               for b, s in combos)


def xg_from_sofascore(x, y, body_part="foot", situation="regular",
                      pitch_length_m=105.0, pitch_width_m=68.0):
    """Convenience wrapper for Sofascore's 0 to 100 shot coordinates.

    x is distance from the goal being attacked, y runs across the pitch with
    50 at the centre. Other providers use different frames, so check yours
    before using this rather than assuming.
    """
    return xg(float(x) / 100.0 * pitch_length_m,
              (float(y) - 50.0) / 100.0 * pitch_width_m,
              body_part=body_part, situation=situation)


if __name__ == "__main__":
    print(f"aleague-xg {__version__}, trained on {TRAINING_SHOTS:,} shots "
          f"from {MODEL_SEASONS}\n")
    examples = [
        ("penalty spot, foot, open play",      11.0, 0.0, "foot", "regular"),
        ("penalty spot, head, from a corner",  11.0, 0.0, "head", "corner"),
        ("six yard line, central, foot",        5.5, 0.0, "foot", "regular"),
        ("six yard line, central, header",      5.5, 0.0, "head", "corner"),
        ("far post, 4m out, header, corner",    4.0, 3.5, "head", "corner"),
        ("edge of the box, central, foot",     18.0, 0.0, "foot", "regular"),
        ("edge of the box, counter",           18.0, 0.0, "foot", "fast_break"),
        ("wide in the box, 10m out",           10.0, 12.0, "foot", "regular"),
        ("30 metres out, central",             30.0, 0.0, "foot", "regular"),
        ("penalty spot, anything",             11.0, 0.0, "any",  "any"),
        ("six yard line, central, anything",    5.5, 0.0, "any",  "any"),
        ("six yard line, any body part, corner", 5.5, 0.0, "any", "corner"),
        ("edge of the box, foot, any phase",   18.0, 0.0, "foot", "any"),
    ]
    for label, d, l, bp, sit in examples:
        print(f"  {label:<36} {xg(d, l, bp, sit):.3f}")
