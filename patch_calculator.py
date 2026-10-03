"""
Adds the "any" option to the calculator page, using the mix table built by
make_xg_mix.py. Run after make_xg_mix.py.

    python3 patch_calculator.py docs/index.html
"""
import json, sys, re

TARGET = sys.argv[1] if len(sys.argv) > 1 else "docs/index.html"
M = json.load(open("xg_mix_table.json"))

keys = [k.split("|") for k in M["keys"]]
js_keys = ",".join(f'["{b}","{s}"]' for b, s in keys)
js_d = ", ".join(f"{v:g}" for v in M["distance_anchors"])
js_l = ", ".join(f"{v:g}" for v in M["lateral_anchors"])
js_cells = ",\n  ".join("[" + ",".join(f"{v:g}" for v in row) + "]" for row in M["cells"])

MIX_JS = f"""
// Mix of body part and phase of play observed by area of the pitch, from
// make_xg_mix.py. Used only when "Any" is chosen. Rows run through the
// distance bands, and within each one through the sideways bands. Values are
// interpolated between band centres so the answer moves smoothly.
const MIX_KEYS = [{js_keys}];
const MIX_D = [{js_d}];
const MIX_L = [{js_l}];
const MIX = [
  {js_cells}
];
const PHASE_LABEL = {{
  regular: "open play", assisted: "an assisted move", corner: "a corner",
  fast_break: "a counter-attack", set_piece: "another set piece",
  free_kick: "a direct free kick", throw_in_set_piece: "a throw-in"
}};

function mixNodes(v, a){{
  if (v <= a[0]) return [0, 0, 1];
  if (v >= a[a.length - 1]) return [a.length - 1, a.length - 1, 1];
  let i = 0;
  while (a[i + 1] < v) i++;
  return [i, i + 1, (a[i + 1] - v) / (a[i + 1] - a[i])];
}}
function mixWeights(dist, lat){{
  const nl = MIX_L.length;
  const [i0, i1, wd] = mixNodes(dist, MIX_D);
  const [j0, j1, wl] = mixNodes(lat, MIX_L);
  const w = []; let t = 0;
  for (let k = 0; k < MIX_KEYS.length; k++){{
    const v = wd * wl * MIX[i0 * nl + j0][k] + (1 - wd) * wl * MIX[i1 * nl + j0][k]
            + wd * (1 - wl) * MIX[i0 * nl + j1][k] + (1 - wd) * (1 - wl) * MIX[i1 * nl + j1][k];
    w.push(v); t += v;
  }}
  return w.map(v => v / t);
}}
function weightsAt(out, off){{
  const y = Math.abs(off);
  return mixWeights(Math.max(Math.hypot(out, y), 0.3), y);
}}
function headShareAt(out, off, situation){{
  const w = weightsAt(out, off);
  let h = 0, t = 0;
  for (let k = 0; k < MIX_KEYS.length; k++){{
    if (situation !== "any" && MIX_KEYS[k][1] !== situation) continue;
    t += w[k];
    if (MIX_KEYS[k][0] === "head") h += w[k];
  }}
  return h / t;
}}
function topPhaseAt(out, off, bodyPart){{
  const w = weightsAt(out, off), agg = {{}};
  let t = 0;
  for (let k = 0; k < MIX_KEYS.length; k++){{
    if (bodyPart !== "any" && MIX_KEYS[k][0] !== bodyPart) continue;
    agg[MIX_KEYS[k][1]] = (agg[MIX_KEYS[k][1]] || 0) + w[k];
    t += w[k];
  }}
  let best = null;
  for (const s in agg) if (best === null || agg[s] > agg[best]) best = s;
  return [best, agg[best] / t];
}}
"""

src = open(TARGET).read()
orig = src
if "bp-any" in src:
    sys.exit(f"{TARGET} already has the any option. Rebuild the page from the "
             f"unpatched source before running this again.")


def swap(old, new):
    global src
    assert old in src, old[:70]
    src = src.replace(old, new, 1)


# ------------------------------------------------------------------ controls
swap('''            <button type="button" id="bp-head" aria-pressed="false">Head</button>
          </div>''',
     '''            <button type="button" id="bp-head" aria-pressed="false">Head</button>
            <button type="button" id="bp-any" aria-pressed="false">Any</button>
          </div>''')

swap('''          <select id="situation">
            <option value="regular">Open play</option>''',
     '''          <select id="situation">
            <option value="regular" selected>Open play</option>''')

swap('''            <option value="throw_in_set_piece">From a throw-in</option>
          </select>''',
     '''            <option value="throw_in_set_piece">From a throw-in</option>
            <option value="any">Any phase, league mix</option>
          </select>''')

swap('''<p>Click anywhere on the pitch to see how likely a shot from that spot is to be scored. Built on 14,065 shots from the 2023/24 to 2025/26 A-League Men regular seasons.</p>''',
     '''<p>Click anywhere on the pitch to see how likely a shot from that spot is to be scored. Built on 14,065 shots from the 2023/24 to 2025/26 A-League Men regular seasons. Choose <b>Any</b> if you do not know how the shot was struck or what created it.</p>''')

# ------------------------------------------------------------------ mix table
swap('''const GOAL_WIDTH = 7.32, HALF_GOAL = GOAL_WIDTH / 2;''',
     MIX_JS.strip() + '''

const GOAL_WIDTH = 7.32, HALF_GOAL = GOAL_WIDTH / 2;''')

# ------------------------------------------------------------------ xg
swap('''function xg(out, off, head, situation){
  const y = Math.abs(off);
  let d = Math.hypot(out, y);
  if (d < 0.3) d = 0.3;
  const ld = Math.log(d), a = shotAngle(out, y);
  const z = B.intercept + B.log_distance * ld + B.distance * d
          + B.angle * a + B.angle_x_logdist * a * ld
          + (head ? B.is_header : 0) + SITUATION[situation];
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}''',
     '''function xgOne(out, off, head, situation){
  const y = Math.abs(off);
  let d = Math.hypot(out, y);
  if (d < 0.3) d = 0.3;
  const ld = Math.log(d), a = shotAngle(out, y);
  const z = B.intercept + B.log_distance * ld + B.distance * d
          + B.angle * a + B.angle_x_logdist * a * ld
          + (head ? B.is_header : 0) + SITUATION[situation];
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}
// With "any" the model is averaged over what is actually hit from this spot,
// which answers what a shot from here is worth before you know how it was
// struck or what created it.
function xg(out, off, bodyPart, situation){
  if (bodyPart !== "any" && situation !== "any"){
    return xgOne(out, off, bodyPart === "head", situation);
  }
  const w = weightsAt(out, off);
  let num = 0, den = 0;
  for (let k = 0; k < MIX_KEYS.length; k++){
    const bp = MIX_KEYS[k][0], sit = MIX_KEYS[k][1];
    if (bodyPart !== "any" && bp !== bodyPart) continue;
    if (situation !== "any" && sit !== situation) continue;
    num += w[k] * xgOne(out, off, bp === "head", sit);
    den += w[k];
  }
  return num / den;
}''')

# ------------------------------------------------------------------ state
swap('''let state = { out: 11, off: 0, head: false, situation: "regular" };''',
     '''let state = { out: 11, off: 0, bodyPart: "foot", situation: "regular" };''')

swap('''document.getElementById("bp-foot").addEventListener("click", () => setHead(false));
document.getElementById("bp-head").addEventListener("click", () => setHead(true));
function setHead(v){
  state.head = v;
  document.getElementById("bp-foot").setAttribute("aria-pressed", String(!v));
  document.getElementById("bp-head").setAttribute("aria-pressed", String(v));
  render();
}''',
     '''["foot", "head", "any"].forEach(v => {
  document.getElementById("bp-" + v).addEventListener("click", () => setBody(v));
});
function setBody(v){
  state.bodyPart = v;
  ["foot", "head", "any"].forEach(k => {
    document.getElementById("bp-" + k).setAttribute("aria-pressed", String(k === v));
  });
  render();
}''')

# ------------------------------------------------------------------ render
swap('''  const { out, off, head, situation } = state;
  const p = xg(out, off, head, situation);''',
     '''  const { out, off, bodyPart, situation } = state;
  const p = xg(out, off, bodyPart, situation);''')

swap('''  const other = xg(out, off, !head, situation);
  const cmp = document.getElementById("compare");
  if (head){
    cmp.innerHTML = `The same chance struck with the <b>foot</b> is worth ${other.toFixed(2)}. `
      + `Across the whole sample a foot shot has roughly twice the odds of a header from the same place.`;
  } else {
    cmp.innerHTML = `Headed from the same spot it is worth <b>${other.toFixed(2)}</b>. `
      + `Headers convert worse everywhere on the pitch, and only look comparable overall because they are taken much closer in.`;
  }''',
     '''  const cmp = document.getElementById("compare");
  const pct = x => Math.round(x * 100) + "%";
  if (bodyPart === "any"){
    const foot = xg(out, off, "foot", situation);
    const head = xg(out, off, "head", situation);
    const share = headShareAt(out, off, situation);
    cmp.innerHTML = `Around this spot <b>${pct(share)}</b> of shots are headed, so the figure above `
      + `sits between a foot shot at ${foot.toFixed(2)} and a header at ${head.toFixed(2)}.`;
  } else if (situation === "any"){
    const top = topPhaseAt(out, off, bodyPart);
    cmp.innerHTML = `Averaged over what creates shots from here. The most common is `
      + `<b>${PHASE_LABEL[top[0]]}</b>, at ${pct(top[1])} of them.`;
  } else if (bodyPart === "head"){
    const other = xg(out, off, "foot", situation);
    cmp.innerHTML = `The same chance struck with the <b>foot</b> is worth ${other.toFixed(2)}. `
      + `Across the whole sample a foot shot has roughly twice the odds of a header from the same place.`;
  } else {
    const other = xg(out, off, "head", situation);
    cmp.innerHTML = `Headed from the same spot it is worth <b>${other.toFixed(2)}</b>. `
      + `Headers convert worse everywhere on the pitch, and only look comparable overall because they are taken much closer in.`;
  }''')

# ------------------------------------------------------------------ footer
swap('''<p><b>Penalties are not covered.</b>''',
     '''<p><b>What "any" does.</b> It averages the model over the body parts and phases of play actually seen around that spot, so close to goal it leans towards headers from corners and further out towards foot shots in open play. It answers what a shot from there is worth on average, which is a different question from what one particular shot is worth. Use a specific body part and phase when you know them.</p>
    <p><b>Penalties are not covered.</b>''')

open(TARGET, "w").write(src)
print(f"patched {TARGET}: {len(orig):,} -> {len(src):,} bytes")
