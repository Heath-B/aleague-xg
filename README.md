# aleague-xg

A free expected goals model for the A-League Men, built from public match data
and released so that anyone can use it without a subscription.

Every xG model with A-League coverage sits behind a paywall. Sofascore's public
feed, which this model is built from, carries no xG value for the A-League Men
at all. So this exists to fill a gap rather than to compete with something that
was already free.

One file, no dependencies, thirteen numbers you can read. There is also an
interactive calculator for people who do not write code.

**[Open the calculator](https://heath-b.github.io/aleague-xg/)**

---

## Quick start

Copy `aleague_xg.py` into your project. It needs Python 3.8 or later and
nothing else.

```python
from aleague_xg import xg

xg(from_goal_line_m=11.0, from_centre_m=0.0)                      # 0.206
xg(5.5, 0.0, body_part="head", situation="corner")                # 0.179
xg(16.5, 0.0, body_part="any",  situation="any")                  # 0.097
```

Both distances are in metres. The first is how far the shot is from the goal
line, the second is how far it sits sideways from the centre of the goal. The
sign of the second is ignored, so 4 and -4 give the same answer.

## Coordinates

The model is provider neutral. It takes metres, not anyone's pitch grid, so you
convert from whatever your source uses before calling it.

| Input | Meaning | Example |
| --- | --- | --- |
| `from_goal_line_m` | Perpendicular distance from the goal line | Penalty spot is 11.0 |
| `from_centre_m` | Sideways distance from the centre of the goal | Near post is about 3.7 |

A convenience wrapper is included for Sofascore's 0 to 100 grid, where x is
distance from the goal being attacked and y runs across the pitch with 50 at
the centre.

```python
from aleague_xg import xg_from_sofascore
xg_from_sofascore(x=10.5, y=50.0, body_part="head", situation="corner")
```

Check your own provider's frame before assuming it matches. Getting this wrong
is the commonest way to get nonsense out of an xG model.

## Inputs

`body_part` is `"foot"`, `"head"` or `"any"`.

`situation` is one of `"regular"`, `"assisted"`, `"corner"`, `"fast_break"`,
`"free_kick"`, `"set_piece"`, `"throw_in_set_piece"` or `"any"`.

`"regular"` means an open-play shot that was not the end of an assist, a
counter or a set piece. Penalties are not covered at all, because a penalty is
a fixed situation from one spot that would distort everything around it. Use
the league penalty conversion rate instead.

### What "any" does

Pass `"any"` for the body part, the phase of play, or both, and the model
averages itself over what is actually struck from that part of the pitch. Close
to goal that leans towards headers from corners, further out towards foot shots
in open play.

It answers what a shot from a spot is worth before you know how it was taken,
which is a different question from what one particular shot is worth. Use the
specific values whenever you know them.

The mix comes from a measured table of 33 pitch areas, interpolated between
area centres so the answer moves smoothly rather than stepping at a boundary.
With the phase of play known and only the body part averaged, it reproduces
actual goals almost exactly in every phase, corners included. Full numbers are
in [VALIDATION.md](VALIDATION.md).

---

## What it was trained on

14,065 shots from the 2023/24, 2024/25 and 2025/26 A-League Men regular
seasons, of which 1,463 were goals, a base rate of 10.40%. Penalties,
shootouts and finals are excluded.

Five features, all of them available before the ball is struck. Distance,
the angle the goal subtends from the shot location, whether it was a header,
the phase of play, and an interaction between angle and distance. Nothing else.

## How well it performs

| Measure | Value |
| --- | --- |
| Log loss, cross-validated | 0.28786 |
| Log loss, league rate alone | 0.33382 |
| Brier, cross-validated | 0.08165 |
| McFadden pseudo R squared | 0.1377 |
| Total xG against actual goals | 1463.1 against 1463 |

On a temporal hold-out, trained on the first two seasons and tested on a
2025/26 it had never seen, it scores 0.28503 against 0.32077 for the league
rate alone, so it does carry to a new season. It over-predicted that season by
about 7% (*z* = -1.50), which is worth knowing if you apply it to a single
season in isolation.

It is exactly calibrated on corner shots, 186 goals against 186.0 expected
(*z* = 0.00), and that holds by quintile and by body part. That matters because
measuring corner over and under-performance was the reason this was built.

The header coefficient gives an odds ratio of 0.41, which reproduces by a
different method, and on a sample six times larger, the finding from the corner
analysis this grew out of.

---

## What it cannot see, and where it is wrong

Being clear about this is the point of publishing it rather than a weakness
to bury.

**No context.** Where the defenders and the goalkeeper were, how much pressure
the shooter was under, and what happened in the seconds before the shot. No
public xG model built on this kind of feed explains more than a fraction of
what decides a goal, and this one is no exception.

**The header penalty does not grow with distance, and it should.** The model
applies a fixed header penalty wherever the shot is taken. The data says
otherwise. Holding the phase of play comparable, the observed odds ratio of a
foot shot to a header runs 1.33 within 6 m, 1.76 from 6 to 11 m, then 6.36 from
11 to 16 m, while the model holds it near 1.33, 1.99 and 2.22. The consequence
is that mid-range headers read high, over-predicting the 11 to 16 m band
(16 goals against 28.4 expected, *z* = -2.33), and long-range headers read far
too high. A fix has been tested and is documented in
[VALIDATION.md](VALIDATION.md), but it is not in this version.

**It runs out of data before it runs out of pitch.** Ninety five per cent of
headers in the sample came from inside 14.5 m, and only one was taken beyond
24 m. Shots of any kind thin out past 35 m. The calculator flags this on screen.
If you call the module directly, treat headers past 24 m and shots past 45 m as
the curve continuing rather than as evidence.

**Spatial fit is imperfect.** Sorting shots purely by location, rather than by
predicted value, is a sharper test than the usual one, and under it the model
under-rates the sixth and seventh deciles and over-rates the fourth
(Hosmer-Lemeshow χ²(8) = 25.97). That is a limit of five features on three
seasons.

**It is one league and three seasons.** The learning curve was still descending
at 14,065 shots, so more data would make a better model.

---

## Rebuilding it

```
build/make_xg_model.py        fits the model, writes the coefficients and the report
build/make_xg_mix.py          measures the mix table behind the "any" option
build/patch_calculator.py     adds the "any" controls to the calculator page
build/patch_calculator_range.py   adds the out-of-range warning, counts measured from the data
```

Run them in that order. They need `numpy` and nothing else.

**The training data is not in this repository.** It was collected from
Sofascore's public match feed and redistributing it is not mine to do, so the
build scripts will not run end to end for you as published. What is included is
everything derived from it, the fitted coefficients, the mix table, the
calibration tables and the full validation report, so the method is auditable
and the results are checkable even though the raw feed is not here.

One thing that trips people up. The `xg_lite` column in the scored shot file
holds the out-of-fold cross-validated value, which is the right number for
evaluating the model. The module returns the full-sample fitted value. The two
differ, and neither is wrong.

## Files

```
aleague_xg.py                 the model, the only file you need
docs/index.html               the calculator, self-contained, served by GitHub Pages
model/xg_model_coefficients.json   fitted coefficients and standard errors
model/xg_mix_table.json       the mix behind the "any" option
model/model_summary.txt       coefficients, fit statistics, calibration table
model/mix_validation.txt      full validation of the "any" option
build/                        the scripts that produced all of the above
VALIDATION.md                 the written validation report
```

## Version

1.0.0. The coefficients will change when the model is refitted, so pin a
version if you need results to stay reproducible.

## Licence

MIT, so use it for anything. Credit is appreciated but not required.

Built by Heath Brain from Sofascore's public match feed.
