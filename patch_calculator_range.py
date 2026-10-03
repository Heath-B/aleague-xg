"""
Adds an out-of-range warning to the calculator, so the page says when a figure
is the formula extrapolating rather than anything the data supports.

Counts are measured from shots_all.csv at build time, not typed in by hand.

    python3 patch_calculator_range.py docs/index.html
"""
import csv, math, sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "docs/index.html"
PL, PW = 105.0, 68.0

HEAD_THIN, HEAD_OUT = 16.0, 24.0
FOOT_THIN, FOOT_OUT = 35.0, 45.0

head, foot = [], []
for r in csv.DictReader(open("shots_all.csv")):
    if r["situation"] in {"penalty", "shootout"} or r["round_name"].strip() != "":
        continue
    try:
        x = float(r["x"]); y = float(r["y"])
    except ValueError:
        continue
    d = x / 100.0 * PL
    lat = (y - 50.0) / 100.0 * PW
    rec = (max(math.hypot(d, lat), 0.3), 1 if r["shot_type"] == "goal" else 0)
    (head if r["body_part"] == "head" else foot).append(rec)


def band(rows, lo, hi):
    b = [r for r in rows if lo <= r[0] < hi]
    return len(b), sum(g for _, g in b)


hn, hg = band(head, HEAD_THIN, HEAD_OUT)
ho, hog = band(head, HEAD_OUT, 1e9)
fn, fg = band(foot, FOOT_THIN, FOOT_OUT)
fo, fog = band(foot, FOOT_OUT, 1e9)
print(f"headers {HEAD_THIN:.0f}-{HEAD_OUT:.0f} m  {hn} shots, {hg} goals")
print(f"headers beyond {HEAD_OUT:.0f} m   {ho} shots, {hog} goals")
print(f"foot {FOOT_THIN:.0f}-{FOOT_OUT:.0f} m     {fn} shots, {fg} goals")
print(f"foot beyond {FOOT_OUT:.0f} m      {fo} shots, {fog} goals")


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


JS = f"""
// How far the training data actually reaches. Measured by
// patch_calculator_range.py, not typed in by hand. Past these distances the
// number on screen is the fitted curve continuing, with nothing behind it.
const RANGE = {{
  head: {{ thin: {HEAD_THIN}, out: {HEAD_OUT},
    thinNote: "Only {plural(hn, 'header')} in three seasons came from this far out, "
      + "and {plural(hg, 'was') if hg == 1 else str(hg) + ' were'} scored. Treat this as indicative.",
    outNote: "{'No header in this sample was taken' if ho == 0 else 'Only ' + plural(ho, 'header') + ' in three seasons came'} "
      + "from beyond {HEAD_OUT:.0f} metres. This figure is the formula extrapolating, not evidence." }},
  foot: {{ thin: {FOOT_THIN}, out: {FOOT_OUT},
    thinNote: "Only {plural(fn, 'shot')} in three seasons came from this band, so it is thinly supported.",
    outNote: "{'No shot in this sample was taken' if fo == 0 else 'Only ' + plural(fo, 'shot') + ' in three seasons came'} "
      + "from beyond {FOOT_OUT:.0f} metres. This figure is the formula extrapolating, not evidence." }}
}};

function rangeNote(dist, bodyPart){{
  // "any" is almost entirely foot shots this far out, so it uses that scale.
  const r = bodyPart === "head" ? RANGE.head : RANGE.foot;
  if (dist >= r.out) return ["out", r.outNote];
  if (dist >= r.thin) return ["thin", r.thinNote];
  return null;
}}
"""

src = open(TARGET).read()
if "rangeNote" in src:
    sys.exit(f"{TARGET} already carries the range warning.")


def swap(old, new):
    global src
    assert old in src, old[:70]
    src = src.replace(old, new, 1)


swap("""footer{border-top:1px solid var(--rule);""",
     """.warn{
  margin:12px 0 0; padding:10px 12px; border-radius:7px; font-size:.85rem;
  line-height:1.45; background:var(--warm-soft); color:var(--ink);
  border-left:3px solid var(--warm);
}
.warn b{color:var(--warm)}
.warn[hidden]{display:none}
footer{border-top:1px solid var(--rule);""")

swap("""  --turf:#e4e8e6;
  --turf-line:#aab4b0;""",
     """  --turf:#e4e8e6;
  --turf-line:#aab4b0;
  --warm-soft:#f6e7e0;""")
swap("""--warm:#d9906c;
    --turf:#1f2529; --turf-line:#3d474e;
    color-scheme:dark;""",
     """--warm:#d9906c; --warm-soft:#3a2a22;
    --turf:#1f2529; --turf-line:#3d474e;
    color-scheme:dark;""")
swap("""--warm:#d9906c;
  --turf:#1f2529; --turf-line:#3d474e;
  color-scheme:dark;""",
     """--warm:#d9906c; --warm-soft:#3a2a22;
  --turf:#1f2529; --turf-line:#3d474e;
  color-scheme:dark;""")

swap("""        <ul class="facts">""",
     """        <p class="warn" id="range-warn" hidden></p>
        <ul class="facts">""")

swap("""const GOAL_WIDTH = 7.32, HALF_GOAL = GOAL_WIDTH / 2;""",
     JS.strip() + """

const GOAL_WIDTH = 7.32, HALF_GOAL = GOAL_WIDTH / 2;""")

swap("""  const cmp = document.getElementById("compare");
  const pct = x => Math.round(x * 100) + "%";""",
     """  const warn = document.getElementById("range-warn");
  const note = rangeNote(Math.hypot(out, off), bodyPart);
  warn.hidden = note === null;
  if (note){
    warn.innerHTML = (note[0] === "out" ? "<b>Outside the data.</b> " : "<b>Thin here.</b> ")
      + note[1];
  }

  const cmp = document.getElementById("compare");
  const pct = x => Math.round(x * 100) + "%";""")

swap("""    <p><b>Penalties are not covered.</b>""",
     """    <p><b>Where it stops being evidence.</b> Almost every header in the sample was taken inside 16 metres, and shots of any kind thin out past 35. Beyond those the page says so, because what you are reading there is the fitted curve continuing rather than anything that happened. One known weakness sits inside that range too. The header penalty is currently a fixed amount wherever the shot is taken, while the data shows headers falling away faster than foot shots as distance grows, so mid-range headers read a little high. That is being fixed in the next version.</p>
    <p><b>Penalties are not covered.</b>""")

open(TARGET, "w").write(src)
print(f"\npatched {TARGET}")
