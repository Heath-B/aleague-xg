# Validation

Everything here is reproducible from the build scripts given the training data.
Where a result reflects badly on the model it is reported anyway, because a
model you can only trust when it flatters itself is not worth publishing.

## Contents

- [Is the complexity earned](#is-the-complexity-earned)
- [Specifications that were tested and rejected](#specifications-that-were-tested-and-rejected)
- [The header fix, tested but not adopted](#the-header-fix-tested-but-not-adopted)
- [Calibration](#calibration)
- [Does it carry to a new season](#does-it-carry-to-a-new-season)
- [The "any" option](#the-any-option)
- [Coordinate screening](#coordinate-screening)
- [Why there is no comparison with another model](#why-there-is-no-comparison-with-another-model)
- [Open items](#open-items)

---

## Is the complexity earned

Each row adds to the row above. Five-fold cross-validated log loss, on the
screened sample of 14,046 shots.

| Specification | CV log loss | Gain (per mille) | Parameters |
| --- | --- | --- | --- |
| League rate alone | 0.33417 | | 1 |
| Distance only | 0.30147 | +32.70 | 2 |
| Distance in two forms | 0.30148 | -0.01 | 3 |
| Plus angle | 0.29907 | +2.42 | 5 |
| Plus header | 0.29187 | +7.20 | 6 |
| Plus phase of play (current model) | 0.28807 | +3.80 | 12 |
| Plus header by distance | 0.28802 | +0.05 | 13 |

Distance does most of the work, then the header flag, then the phase of play.

One row does not earn its place. Adding distance in linear form alongside the
logged form gains nothing at all on cross-validated log loss. It was included
because a single log-distance term left the middle of the range miscalibrated,
so it earns on calibration rather than on accuracy. That is a defensible
reason, but it should be revisited.

## Specifications that were tested and rejected

Nine candidates were compared before the published model was chosen. Three
later candidates addressed specific faults. All on the full 14,065 shots.

| Added term | CV log loss | Coefficient | SE | z |
| --- | --- | --- | --- | --- |
| none, published model | 0.28786 | | | |
| header by log distance | 0.28784 | -0.448 | 0.213 | -2.10 |
| corner by log distance | 0.28803 | +0.095 | 0.170 | +0.56 |
| corner by header | 0.28790 | -0.007 | 0.193 | -0.03 |

**Corners do not need their own slopes.** Both corner interactions are
indistinguishable from zero. The phase of play enters as an intercept shift and
that is sufficient. A separate corner-only model is also the wrong answer, 186
corner goals against 1,463 overall widens every standard error by a factor of
about 2.8, which would leave the geometry terms unidentifiable.

## The header fix, tested but not adopted

The known weakness described in the README has a tested fix. It is not in
version 1.0.0, and the reasoning for holding it is recorded here.

Adding a header by log-distance term repairs the fault:

| Distance | Headers | Goals | Expected now | z | Expected with fix | z |
| --- | --- | --- | --- | --- | --- | --- |
| 0 to 6 m | 308 | 91 | 81.7 | +1.03 | 91.1 | -0.01 |
| 6 to 11 m | 1403 | 143 | 139.8 | +0.27 | 135.5 | +0.65 |
| 11 to 16 m | 543 | 16 | 28.4 | -2.33 | 23.7 | -1.58 |
| 16 to 24 m | 36 | 1 | 1.0 | -0.04 | 0.8 | +0.28 |

It also tames the extrapolation. A header from the centre reads as follows.

| Distance | Now | With the fix |
| --- | --- | --- |
| 16 m | 1 in 21 | 1 in 28 |
| 24 m | 1 in 50 | 1 in 81 |
| 40 m | 1 in 183 | 1 in 407 |
| 53 m | 1 in 464 | 1 in 1267 |

Every aggregate calibration is untouched. Headers 251 against 251.0, foot
shots 1212 against 1212.0, corners 186 against 186.0, all shots 1463 against
1463.0, identical with and without the term.

The term is significant (*z* = -2.09) and its direction is physically right.
What it does not do is improve predictive accuracy, gaining 0.05 per mille of
log loss, because mid and long-range headers are a small share of all shots. On
the temporal hold-out it is marginally worse (0.28561 against 0.28503).

Adopting it therefore means choosing local correctness over aggregate accuracy.
That is a judgement about what the model is for rather than a result, and it
is being left until there is more evidence.

## Calibration

Cross-validated predictions, in deciles of predicted value.

| Bin | n | Mean xG | Actual | xG sum | Goals |
| --- | --- | --- | --- | --- | --- |
| 1 | 1407 | 0.0212 | 0.0270 | 29.8 | 38 |
| 2 | 1407 | 0.0338 | 0.0291 | 47.6 | 41 |
| 3 | 1407 | 0.0428 | 0.0398 | 60.2 | 56 |
| 4 | 1407 | 0.0523 | 0.0441 | 73.6 | 62 |
| 5 | 1407 | 0.0635 | 0.0462 | 89.4 | 65 |
| 6 | 1406 | 0.0759 | 0.0811 | 106.7 | 114 |
| 7 | 1406 | 0.0935 | 0.1046 | 131.5 | 147 |
| 8 | 1406 | 0.1199 | 0.1238 | 168.5 | 174 |
| 9 | 1406 | 0.1676 | 0.1814 | 235.7 | 255 |
| 10 | 1406 | 0.3699 | 0.3634 | 520.0 | 511 |

**A harder test.** Sorting shots purely by location, rather than by predicted
value, stops a close header and a long-range shot of similar value from
cancelling each other out, and is a sharper test of spatial fit. Under it the
model scores Hosmer-Lemeshow χ²(8) = 25.97, under-rating the sixth and seventh
deciles and over-rating the fourth. That is a real limitation of five features
on three seasons and it is not fixed in this version.

**Corners**, which is what the model was built to measure, come out exact.
186 goals against 186.0 expected (*z* = 0.00), holding by quintile and by body
part.

## Does it carry to a new season

Trained on 2023/24 and 2024/25, tested on 2025/26, which it had never seen.

| | Log loss on the unseen season |
| --- | --- |
| The model | 0.28503 |
| League rate alone | 0.32077 |

It expected 459.1 goals against 427 actual (*z* = -1.50), so it ran about 7%
high on that season. Not significant, but worth knowing before applying it to
one season in isolation.

## The "any" option

The mix table holds 33 pitch areas, each carrying the observed shares of body
part and phase of play, shrunk towards the league-wide mix so no area is empty,
with areas below 150 shots falling back to their distance band. Values are
interpolated between area centres.

Validated on a grid deliberately finer than the table's own areas, so a coarse
table could not flatter itself.

| Measure | Value |
| --- | --- |
| Root mean square gap against the fitted value | 0.381 pp |
| Worst area gap | 2.136 pp |
| League total against the per-shot values | +0.48% |
| Steps where the value rises as you move away from goal | 0 |

With the phase of play known and only the body part averaged, it reproduces
actual goals almost exactly.

| Phase | Shots | Expected | Actual | z |
| --- | --- | --- | --- | --- |
| Open play | 2066 | 239.3 | 239 | -0.02 |
| Assisted | 7379 | 761.4 | 759 | -0.09 |
| Corner | 2198 | 185.2 | 186 | +0.06 |
| Counter-attack | 1021 | 166.3 | 166 | -0.02 |
| Free kick | 411 | 22.0 | 22 | -0.00 |
| Other set piece | 708 | 71.2 | 71 | -0.02 |
| Throw-in | 282 | 20.1 | 20 | -0.02 |

Averaging adds no error of its own. Run the fully specified model through the
same location-sorted groups and it scores χ²(8) = 25.97 against the averaged
value's 25.81, essentially identical, with the gaps falling in the same groups.
The averaged value inherits the model's spatial error and contributes none.

## Coordinate screening

Nineteen rows carry coordinates that cannot be right, including a header logged
at 82.1 m from a corner and several shots between 64 and 75 m. None of them is
a goal, so excluding them changes nothing material, and the published model is
fitted without screening. The 99.9th percentile of header distance is 23.1 m.

## Why there is no comparison with another model

The original plan was to benchmark against Sofascore's own xG on the same
shots. That is not possible. Their shotmap carries no xG field for the A-League
Men, absent entirely rather than empty, across 5,883 shots checked in 200
matches. There is no free A-League xG to compare against, which is the reason
this model exists.

The evaluations above are the substitute, testing the model against simpler
versions of itself, against a season it has not seen, and against the data
directly.

## Open items

Recorded rather than hidden.

1. **Leave-one-club-out is unresolved.** Refitting thirteen times, each
   dropping one club, moved some coefficient by 0.653 when Wellington Phoenix
   was excluded. Which coefficient was not captured, and it is larger than is
   comfortable. This needs following up before anyone relies on the per-term
   values.
2. **The header by distance term** is tested and ready, and held pending more
   evidence.
3. **The linear distance term** earns nothing on cross-validated log loss and
   should be re-examined.
4. **More seasons.** The learning curve was still descending at 14,065 shots.
5. **Preferred foot** is not in the model. There are 11,736 foot shots across
   474 players in the sample, enough to support the split if the data can be
   sourced.
