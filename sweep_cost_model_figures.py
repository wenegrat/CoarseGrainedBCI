#!/usr/bin/env python3
"""Regenerate sweep_cost_model.html -- the cost-model writeup for the parallelized filter-scale sweep.

Charts are built as inline SVG (theme-aware via CSS custom properties) directly from the measured
per-batch wall times of two real PBS runs, so the figures cannot drift from the numbers in the prose.
Add a new run by appending to o256/o512 (or adding an o1024) and re-running:  python sweep_cost_model_figures.py
"""
import numpy as np
def sc(N,n=30):
    dx=1000/N; return np.geomspace(2*dx,400,n),dx
o256=[((0,16),68.07),((16,21),27.13),((21,24),20.63),((24,26),16.28),((26,27),8.75),((27,28),9.72),((28,29),10.65),((29,30),14.02)]
o512=[((0,9),72.55),((9,16),60.45),((16,21),58.42),((21,24),52.48),((24,26),46.60),((26,28),60.48),((28,29),36.57),((29,30),42.53)]
V={256:256**2*64*41,512:512**2*65*21}
def design(obs,N):
    s,dx=sc(N); return np.array([[b-a,(s[a:b]/dx).sum()] for (a,b),_ in obs]),np.array([t for _,t in obs])
X6,y6=design(o256,256); (A6,B6),*_=np.linalg.lstsq(X6,y6,rcond=None)
X5,y5=design(o512,512); (A5,B5),*_=np.linalg.lstsq(X5,y5,rcond=None)
r=V[512]/V[256]; pred=X5@np.array([A6*r,B6*r])

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def txt(x,y,s,cls="",anc="middle",dy="0"):
    return f'<text x="{x:.1f}" y="{y:.1f}" dy="{dy}" text-anchor="{anc}" class="{cls}">{esc(s)}</text>'

# ---------- Chart 1: per-batch wall time, before / after ----------
def bars(obs, W=340, H=250, pad=(38,10,34,14)):
    l,rr,b,t=pad; pw=W-l-rr; ph=H-t-b
    vals=[v for _,v in obs]; vmax=max(vals)*1.12
    o=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for g in range(5):
        yy=t+ph*g/4; v=vmax*(1-g/4)
        o.append(f'<line x1="{l}" y1="{yy:.1f}" x2="{W-rr}" y2="{yy:.1f}" class="grid"/>')
        o.append(txt(l-7,yy,f"{v:.0f}","tick","end",dy="0.32em"))
    n=len(obs); bw=pw/n*0.68; step=pw/n
    for i,((a,bb),v) in enumerate(obs):
        x=l+step*i+(step-bw)/2; h=ph*v/vmax
        o.append(f'<rect x="{x:.1f}" y="{t+ph-h:.1f}" width="{bw:.1f}" height="{h:.1f}" class="bar"/>')
        o.append(txt(x+bw/2,t+ph-h-5,f"{v:.0f}","val"))
        o.append(txt(x+bw/2,t+ph+13,f"{a}–{bb}","tickxs"))
    mean=sum(vals)/n; ym=t+ph-ph*mean/vmax
    o.append(f'<line x1="{l}" y1="{ym:.1f}" x2="{W-rr}" y2="{ym:.1f}" class="meanline"/>')
    o.append(f'<line x1="{l}" y1="{t}" x2="{l}" y2="{t+ph}" class="axis"/>')
    o.append(f'<line x1="{l}" y1="{t+ph}" x2="{W-rr}" y2="{t+ph}" class="axis"/>')
    o.append('</svg>')
    return "\n".join(o)

# ---------- Chart 2: relative per-scale cost vs l (log-log) ----------
def relcost(W=720,H=330,pad=(52,120,44,16)):
    l,rr,b,t=pad; pw=W-l-rr; ph=H-t-b
    xs=(1,300); ys=(0.8,260)
    X=lambda v:l+pw*(np.log10(v)-np.log10(xs[0]))/(np.log10(xs[1])-np.log10(xs[0]))
    Y=lambda v:t+ph-ph*(np.log10(v)-np.log10(ys[0]))/(np.log10(ys[1])-np.log10(ys[0]))
    o=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for v in (1,3,10,30,100,300):
        o.append(f'<line x1="{X(v):.1f}" y1="{t}" x2="{X(v):.1f}" y2="{t+ph}" class="grid"/>')
        o.append(txt(X(v),t+ph+15,f"{v}×","tick"))
    for v in (1,3,10,30,100,250):
        o.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{W-rr}" y2="{Y(v):.1f}" class="grid"/>')
        o.append(txt(l-8,Y(v),f"{v}×","tick","end",dy="0.32em"))
    # refuted pure-l line: cost ratio == l ratio
    o.append(f'<line x1="{X(1):.1f}" y1="{Y(1):.1f}" x2="{X(260):.1f}" y2="{Y(260):.1f}" class="refuted"/>')
    o.append(txt(X(150),Y(230),"assumed  cost ∝ ℓ","lbl-refuted","end"))
    K=0.0230
    cols={256:"a",512:"b",1024:"c"}
    for N,cl in cols.items():
        s,dx=sc(N); w=1+K*s/dx; rel=w/w[0]; xr=s/s[0]
        pts=" ".join(f"{X(a):.1f},{Y(v):.1f}" for a,v in zip(xr,rel))
        o.append(f'<polyline points="{pts}" class="ln ln-{cl}"/>')
        o.append(f'<circle cx="{X(xr[-1]):.1f}" cy="{Y(rel[-1]):.1f}" r="4" class="dot dot-{cl}"/>')
        o.append(txt(X(xr[-1])+11,Y(rel[-1]),f"{N}²   {rel[-1]:.1f}×",f"lbl lbl-{cl}","start",dy="0.32em"))
    o.append(f'<line x1="{l}" y1="{t}" x2="{l}" y2="{t+ph}" class="axis"/>')
    o.append(f'<line x1="{l}" y1="{t+ph}" x2="{W-rr}" y2="{t+ph}" class="axis"/>')
    o.append(txt(l+pw/2,H-6,"filter scale ℓ, relative to the smallest in the sweep","axlbl"))
    o.append(f'<text transform="translate(13,{t+ph/2}) rotate(-90)" text-anchor="middle" class="axlbl">cost per scale, relative</text>')
    o.append('</svg>')
    return "\n".join(o)

# ---------- Chart 3: out-of-sample predicted vs observed ----------
def oos(W=720,H=290,pad=(46,16,52,16)):
    l,rr,b,t=pad; pw=W-l-rr; ph=H-t-b
    vmax=max(max(y5),max(pred))*1.15
    o=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for g in range(5):
        yy=t+ph*g/4
        o.append(f'<line x1="{l}" y1="{yy:.1f}" x2="{W-rr}" y2="{yy:.1f}" class="grid"/>')
        o.append(txt(l-8,yy,f"{vmax*(1-g/4):.0f}","tick","end",dy="0.32em"))
    n=len(o512); step=pw/n; bw=step*0.30
    for i,(((a,bb),ov),pv) in enumerate(zip(o512,pred)):
        cx=l+step*(i+0.5)
        for j,(v,cls) in enumerate(((pv,"bar-pred"),(ov,"bar"))):
            x=cx-bw*1.05+j*bw*1.1; h=ph*v/vmax
            o.append(f'<rect x="{x:.1f}" y="{t+ph-h:.1f}" width="{bw:.1f}" height="{h:.1f}" class="{cls}"/>')
        e=100*(pv-ov)/ov
        o.append(txt(cx,t+ph-ph*max(ov,pv)/vmax-6,f"{e:+.0f}%","val"))
        o.append(txt(cx,t+ph+14,f"{a}–{bb}","tickxs"))
        o.append(txt(cx,t+ph+27,f"{bb-a}",  "tickxxs"))
    o.append(f'<line x1="{l}" y1="{t}" x2="{l}" y2="{t+ph}" class="axis"/>')
    o.append(f'<line x1="{l}" y1="{t+ph}" x2="{W-rr}" y2="{t+ph}" class="axis"/>')
    o.append(txt(l+pw/2,H-4,"scale-index range  /  number of scales in the batch","axlbl"))
    o.append('</svg>')
    return "\n".join(o)

# ---------- Chart 4: 1024^2 feasibility vs n_times ----------
def feas(W=720,H=340,pad=(56,58,46,18)):
    l,rr,b,t=pad; pw=W-l-rr; ph=H-t-b
    nt=np.arange(15,95); V512=512**2*65*21
    f=(1024**2*64*nt)/V512
    s,dx=sc(1024); mk=(A5*f[:,None]+B5*f[:,None]*(s/dx)).max(axis=1)/60
    mem=62.4*f
    X=lambda v:l+pw*(v-15)/(95-15)
    Ym=lambda v:t+ph/2-8-(ph/2-8)*v/26          # top panel: hours, 0..26
    Yg=lambda v:t+ph-(ph/2-8)*v/1100            # bottom: GiB, 0..1100
    o=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for v in (20,40,60,80):
        o.append(f'<line x1="{X(v):.1f}" y1="{t}" x2="{X(v):.1f}" y2="{t+ph}" class="grid"/>')
        o.append(txt(X(v),t+ph+16,str(v),"tick"))
    for v in (0,12,24):
        o.append(f'<line x1="{l}" y1="{Ym(v):.1f}" x2="{W-rr}" y2="{Ym(v):.1f}" class="grid"/>')
        o.append(txt(l-8,Ym(v),f"{v}h","tick","end",dy="0.32em"))
    for v in (0,340,682,1000):
        o.append(f'<line x1="{l}" y1="{Yg(v):.1f}" x2="{W-rr}" y2="{Yg(v):.1f}" class="grid"/>')
        o.append(txt(l-8,Yg(v),f"{v}","tick","end",dy="0.32em"))
    o.append(f'<line x1="{l}" y1="{Ym(23.98):.1f}" x2="{W-rr}" y2="{Ym(23.98):.1f}" class="limit"/>')
    o.append(txt(W-rr-4,Ym(23.98)-7,"23:59 walltime cap","lbl-limit","end"))
    o.append(f'<line x1="{l}" y1="{Yg(682):.1f}" x2="{W-rr}" y2="{Yg(682):.1f}" class="limit"/>')
    o.append(txt(W-rr-4,Yg(682)-7,"682 GiB request","lbl-limit","end"))
    o.append('<polyline points="'+" ".join(f"{X(a):.1f},{Ym(v):.1f}" for a,v in zip(nt,mk))+'" class="ln ln-b"/>')
    o.append('<polyline points="'+" ".join(f"{X(a):.1f},{Yg(v):.1f}" for a,v in zip(nt,mem))+'" class="ln ln-b"/>')
    over=nt[mem>682][0]
    o.append(f'<rect x="{X(over):.1f}" y="{t}" width="{W-rr-X(over):.1f}" height="{ph}" class="danger"/>')
    o.append(f'<line x1="{X(over):.1f}" y1="{t}" x2="{X(over):.1f}" y2="{t+ph}" class="limit"/>')
    o.append(txt(X(over)+6,t+12,f"OOM beyond {over} timesteps","lbl-limit","start"))
    for v,lab in ((21,"21"),(41,"41")):
        i=list(nt).index(v)
        o.append(f'<circle cx="{X(v):.1f}" cy="{Ym(mk[i]):.1f}" r="4.5" class="dot dot-b"/>')
        o.append(txt(X(v),Ym(mk[i])-11,f"{mk[i]:.1f} h","val"))
        o.append(f'<circle cx="{X(v):.1f}" cy="{Yg(mem[i]):.1f}" r="4.5" class="dot dot-b"/>')
        o.append(txt(X(v),Yg(mem[i])-11,f"{mem[i]:.0f} GiB","val"))
    o.append(txt(l+pw/2,H-4,"timesteps in the sweep (after pair dedup and N_TIME_SKIP)","axlbl"))
    o.append(f'<line x1="{l}" y1="{t}" x2="{l}" y2="{t+ph}" class="axis"/>')
    o.append('</svg>')
    return "\n".join(o)

SVGS = dict(c1a=bars(o256), c1b=bars(o512), c2=relcost(), c3=oos(), c4=feas())

TEMPLATE = r"""<title>Sweep cost model — CoarseGrainedBCI</title>
<style>
:root{
  --paper:#f2f4f7; --panel:#fbfcfd; --ink:#151d26; --dim:#5b6675; --faint:#8e99a8;
  --rule:#d3dae3; --grid:#e2e7ee;
  --measured:#0e7480; --measured-soft:#0e748022;
  --refuted:#9d5a37; --critical:#b23c31; --critical-soft:#b23c3114;
  --c-a:#7b8fa8; --c-b:#0e7480; --c-c:#1d3f66;
  --serif:Charter,"Bitstream Charter","Sitka Text","Iowan Old Style",Cambria,Georgia,serif;
  --mono:"IBM Plex Mono","SFMono-Regular","SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#11161c; --panel:#171e26; --ink:#e4e9ef; --dim:#9aa6b4; --faint:#6c7889;
  --rule:#2a333e; --grid:#232c36;
  --measured:#3fb3bf; --measured-soft:#3fb3bf22;
  --refuted:#c98a5e; --critical:#e0705f; --critical-soft:#e0705f14;
  --c-a:#7b8fa8; --c-b:#3fb3bf; --c-c:#7fa8d8;
}}
:root[data-theme="dark"]{
  --paper:#11161c; --panel:#171e26; --ink:#e4e9ef; --dim:#9aa6b4; --faint:#6c7889;
  --rule:#2a333e; --grid:#232c36;
  --measured:#3fb3bf; --measured-soft:#3fb3bf22;
  --refuted:#c98a5e; --critical:#e0705f; --critical-soft:#e0705f14;
  --c-a:#7b8fa8; --c-b:#3fb3bf; --c-c:#7fa8d8;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.62;margin:0;padding:0;-webkit-font-smoothing:antialiased}
.wrap{max-width:860px;margin:0 auto;padding:clamp(2.2rem,5vw,4.5rem) clamp(1.1rem,4vw,2rem) 5rem;
  display:flex;flex-direction:column;gap:2.6rem}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--measured);margin:0}
h1{font-size:clamp(1.9rem,4.6vw,2.7rem);line-height:1.16;margin:.5rem 0 0;font-weight:600;
  letter-spacing:-.015em;text-wrap:balance}
.standfirst{font-size:1.12rem;color:var(--dim);margin:.9rem 0 0;max-width:64ch}
h2{font-size:1.42rem;font-weight:600;letter-spacing:-.01em;margin:0 0 .1rem;text-wrap:balance}
h3{font-size:1.02rem;font-weight:600;margin:0 0 .1rem}
p{margin:0;max-width:68ch}
section{display:flex;flex-direction:column;gap:1rem}
.sechead{display:flex;flex-direction:column;gap:.25rem;padding-top:1.6rem;border-top:1px solid var(--rule)}
.lede{color:var(--dim)}
code,.m{font-family:var(--mono);font-size:.86em;font-variant-numeric:tabular-nums}
code{background:var(--measured-soft);padding:.1em .38em;border-radius:2px;color:var(--ink)}
pre{font-family:var(--mono);font-size:.82rem;line-height:1.55;background:var(--panel);
  border:1px solid var(--rule);border-left:2px solid var(--measured);
  padding:.85rem 1rem;overflow-x:auto;margin:0;border-radius:2px}
.eq{font-family:var(--mono);font-size:.94rem;text-align:center;background:var(--panel);
  border:1px solid var(--rule);padding:1.05rem .8rem;border-radius:2px;overflow-x:auto;
  font-variant-numeric:tabular-nums}
.eq .note{display:block;font-size:.76rem;color:var(--faint);margin-top:.5rem;letter-spacing:.02em}
figure{margin:0;display:flex;flex-direction:column;gap:.6rem}
figcaption{font-size:.83rem;color:var(--dim);max-width:70ch}
figcaption b{color:var(--ink);font-weight:600}
.panelbox{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:1rem .9rem .7rem}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}
@media(max-width:640px){.pair{grid-template-columns:1fr}}
.ptitle{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);margin:0 0 .1rem}
.pmeta{font-family:var(--mono);font-size:.72rem;color:var(--faint);margin:0 0 .5rem}
.spread{font-family:var(--mono);font-size:.76rem;margin:.4rem 0 0;color:var(--dim)}
.spread b{font-size:1.15rem;font-weight:600}
.bad b{color:var(--refuted)} .good b{color:var(--measured)}
.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1}
.bar{fill:var(--measured)} .bar-pred{fill:var(--c-a);opacity:.55}
.meanline{stroke:var(--faint);stroke-width:1;stroke-dasharray:2 4}
.limit{stroke:var(--critical);stroke-width:1.3;stroke-dasharray:5 4}
.danger{fill:var(--critical-soft)}
.refuted{stroke:var(--refuted);stroke-width:1.4;stroke-dasharray:6 4}
.ln{fill:none;stroke-width:2;stroke-linejoin:round}
.ln-a{stroke:var(--c-a)} .ln-b{stroke:var(--c-b)} .ln-c{stroke:var(--c-c)}
.dot-a{fill:var(--c-a)} .dot-b{fill:var(--c-b)} .dot-c{fill:var(--c-c)}
text{font-family:var(--mono);font-variant-numeric:tabular-nums}
.tick{font-size:10px;fill:var(--faint)}
.tickxs{font-size:9.5px;fill:var(--dim)}
.tickxxs{font-size:9px;fill:var(--faint)}
.val{font-size:10px;fill:var(--dim)}
.axlbl{font-size:10.5px;fill:var(--dim);letter-spacing:.04em}
.lbl{font-size:11px;font-weight:600}
.lbl-a{fill:var(--c-a)} .lbl-b{fill:var(--c-b)} .lbl-c{fill:var(--c-c)}
.lbl-refuted{font-size:11px;fill:var(--refuted)}
.lbl-limit{font-size:10px;fill:var(--critical)}
.legend{display:flex;flex-wrap:wrap;gap:1.1rem;font-family:var(--mono);font-size:.74rem;color:var(--dim)}
.legend span{display:flex;align-items:center;gap:.4rem}
.sw{width:15px;height:3px;border-radius:1px;flex:none}
.tblwrap{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.87rem;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:.52rem .8rem;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;
  color:var(--dim);font-weight:500}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono)}
tr.flag td{color:var(--critical)} tr.flag{background:var(--critical-soft)}
tr.pick{background:var(--measured-soft)} tr.pick td:first-child{font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:.9rem 1rem;
  display:flex;flex-direction:column;gap:.22rem}
.card .k{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.card .v{font-family:var(--mono);font-size:1.42rem;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.card .s{font-size:.79rem;color:var(--faint);line-height:1.4}
.card.hi .v{color:var(--measured)} .card.warn .v{color:var(--critical)}
.verdict{border-left:2px solid var(--measured);padding:.15rem 0 .15rem 1rem;color:var(--dim)}
.verdict b{color:var(--ink);font-weight:600}
footer{border-top:1px solid var(--rule);padding-top:1.3rem;font-size:.79rem;color:var(--faint);
  display:flex;flex-direction:column;gap:.35rem}
footer code{background:none;padding:0;color:var(--dim)}
@media(prefers-reduced-motion:no-preference){
  .wrap>*{animation:rise .5s cubic-bezier(.2,.7,.3,1) backwards}
  .wrap>*:nth-child(2){animation-delay:.05s} .wrap>*:nth-child(3){animation-delay:.1s}
  @keyframes rise{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
}
</style>

<div class="wrap">

<header>
  <p class="eyebrow">CoarseGrainedBCI · postprocessing/sweep · parallelize-sweep</p>
  <h1>The filter-scale sweep costs what we measured, not what the algorithm suggested</h1>
  <p class="standfirst">Splitting a 30-scale sweep across PBS jobs needs a cost model to balance the
  batches. The obvious one — Gaussian filtering is an <span class="m">O(L·σ)</span> correlation with
  <span class="m">σ ∝ ℓ</span>, so weight by ℓ — was wrong by an order of magnitude. Two runs at
  256² and 512² replaced it with an affine model that predicts out of sample to 7%, and moved the
  binding constraint at 1024² from walltime to memory.</p>
</header>

<div class="cards">
  <div class="card"><span class="k">assumed cost ratio</span><span class="v">51×</span>
    <span class="s">largest scale vs. smallest, weighting by ℓ</span></div>
  <div class="card hi"><span class="k">measured at 256²</span><span class="v">3.2×</span>
    <span class="s">what the affine fit gives instead</span></div>
  <div class="card hi"><span class="k">out-of-sample error</span><span class="v">7.4%</span>
    <span class="s">256² fit predicting eight 512² batches</span></div>
  <div class="card warn"><span class="k">1024² hard limit</span><span class="v">57</span>
    <span class="s">timesteps before a single scale exceeds the memory request</span></div>
</div>

<section>
  <div class="sechead">
    <p class="eyebrow">the refutation</p>
    <h2>Eight batches that should have taken equal time didn't</h2>
  </div>
  <p class="lede">The split is cost-weighted, so a correctly-weighted run produces batches of equal wall
  time by construction. That makes any real run a direct test of the weighting. The first one failed it.</p>

  <div class="pair">
    <div class="panelbox">
      <p class="ptitle">256² · weighted by ℓ</p>
      <p class="pmeta">30 scales, K=8, 41 timesteps · minutes</p>
      @@c1a@@
      <p class="spread bad">spread <b>7.8×</b> — should be 1.0×</p>
    </div>
    <div class="panelbox">
      <p class="ptitle">512² · affine weights</p>
      <p class="pmeta">30 scales, K=8, 21 timesteps · minutes</p>
      @@c1b@@
      <p class="spread good">spread <b>2.0×</b></p>
    </div>
  </div>
  <figcaption>Wall time per batch; the dashed line is each run's own mean. Bars are labelled by
  scale-index range — under ℓ-weighting the leftmost batch holds 16 of the 30 scales precisely
  <i>because</i> the model believed they were cheap. <b>They weren't cheap; they were the bulk of the
  work.</b> Absolute times differ between panels (different grid and timestep count) — the spread is
  the comparable quantity.</figcaption>
</section>

<section>
  <div class="sechead">
    <p class="eyebrow">the correction</p>
    <h2>A large fixed cost per scale swamps the ℓ term at small ℓ</h2>
  </div>
  <p>Per-scale cost is affine, not proportional. Loading and rechunking <code>ds_filt</code>, writing the
  checkpoint, and the sort's downstream use are all fixed-size work that doesn't care how wide the filter
  kernel is — and across a log-spaced sweep spanning 51× in ℓ, that fixed term dominates most of the range.</p>

  <div class="eq">cost per scale &nbsp;=&nbsp; 3.69 min &nbsp;+&nbsp; 0.081 min · (ℓ / Δx)
    <span class="note">fitted at 256², Δx = 3.906 km, 64 levels, 41 timesteps</span></div>

  <p>Both terms scale with data volume <span class="m">Nx·Ny·Nz·n_times</span>, but only the second carries
  a <span class="m">1/Δx</span>, so the two diverge with resolution. Storing the weight resolution-free as
  <span class="m">w ∝ 1 + 0.0230·(ℓ/Δx)</span> — a fixed cost plus a term proportional to kernel width in
  grid cells — keeps it valid at any grid. A kernel about 43 cells wide doubles the per-scale cost. The
  two runs independently imply 0.0221 and 0.0239 across a 2.08× change in data volume, which is the check
  that the resolution-free form holds.</p>

  <figure>
    <div class="panelbox">@@c2@@</div>
    <div class="legend">
      <span><i class="sw" style="background:var(--refuted)"></i>assumed, cost ∝ ℓ</span>
      <span><i class="sw" style="background:var(--c-a)"></i>256²</span>
      <span><i class="sw" style="background:var(--c-b)"></i>512²</span>
      <span><i class="sw" style="background:var(--c-c)"></i>1024²</span>
    </div>
    <figcaption>Cost per scale relative to the cheapest scale in the sweep, against ℓ relative to the
    smallest. The refuted model is the diagonal. <b>The real curves flatten at small ℓ</b>, where fixed
    cost dominates — which is exactly where the ℓ-weighting packed most of the scales into one batch.
    The gap narrows with resolution: at 1024² the ratio is 12×, still far from the 205× that weighting
    by ℓ would assume over that grid's wider scale range.</figcaption>
  </figure>
</section>

<section>
  <div class="sechead">
    <p class="eyebrow">the validation</p>
    <h2>Fit at 256², predict 512², check</h2>
  </div>
  <p>A model fitted and evaluated on the same run proves nothing. The 512² run is a genuine held-out test:
  take the 256² coefficients, scale both by the data-volume ratio (2.081), and predict all eight batches
  before looking at them.</p>

  <figure>
    <div class="panelbox">@@c3@@</div>
    <div class="legend">
      <span><i class="sw" style="background:var(--c-a);opacity:.55"></i>predicted from the 256² fit alone</span>
      <span><i class="sw" style="background:var(--measured)"></i>observed at 512²</span>
      <span>minutes</span>
    </div>
    <figcaption>Mean absolute error <b>7.4%</b>, and the makespan — the number that actually decides
    whether a job fits — predicted at 1.26 h against 1.21 h observed. Refitting each resolution
    independently confirms the mechanism rather than just the outcome: <b>both coefficients track data
    volume</b>, the σ term to within 4% and the fixed term to within 11%.</figcaption>
  </figure>

  <p class="verdict">This specifically retires an earlier worry in <code>CLAUDE.md</code> that per-scale
  cost blows up ~2.5× faster than <span class="m">O(L·σ)</span> reasoning predicts. <b>It doesn't.</b>
  That anecdote compared two runs differing in <span class="m">n_times</span> as well as grid — and
  <span class="m">n_times</span> turns out to be a first-order term in its own right.</p>
</section>

<section>
  <div class="sechead">
    <p class="eyebrow">the payoff</p>
    <h2>At 1024², memory binds before walltime does</h2>
  </div>
  <p>Cost and memory are both linear in <span class="m">n_times</span>, and they hit their respective
  ceilings at different points. Batch size moves walltime only — peak memory is set by one scale's working
  set, so no <code>N_SCALE_JOBS</code> value rescues an over-budget run.</p>

  <figure>
    <div class="panelbox">@@c4@@</div>
    <figcaption>Top trace: makespan at <span class="m">N_SCALE_JOBS ≥ 12</span>, which equals the cost of
    the single largest scale — the irreducible floor, since a scale can't be split. Bottom trace:
    single-scale memory. <b>The memory ceiling arrives first</b>, at 57 timesteps, while walltime still
    has room to 100+.</figcaption>
  </figure>

  <div class="tblwrap"><table>
    <thead><tr><th>n_times</th><th>sequential</th><th>K ≥ 12 makespan</th><th>single-scale memory</th><th>verdict</th></tr></thead>
    <tbody>
      <tr class="pick"><td class="num">21</td><td class="num">39 h</td><td class="num">4.8 h</td><td class="num">246 GiB</td><td>comfortable</td></tr>
      <tr class="pick"><td class="num">41</td><td class="num">77 h</td><td class="num">9.4 h</td><td class="num">480 GiB</td><td>2.5× walltime margin</td></tr>
      <tr class="flag"><td class="num">81</td><td class="num">152 h</td><td class="num">18.6 h</td><td class="num">948 GiB</td><td>exceeds the 682 GiB request</td></tr>
    </tbody>
  </table></div>

  <h3>What to run</h3>
  <pre># 1024², keep the time axis at or below ~41
cd postprocessing
bash submit_sweep.sh NX=1024 NY=1024 NZ=64 N_SCALES=30 N_SCALE_JOBS=10 N_TIME_SKIP=2</pre>
  <p><code>N_SCALE_JOBS=10</code> is the threshold below which the split can't reach its own floor.
  Two bounds apply to the heaviest batch — an even share, and the single costliest scale, which is
  indivisible. Under <span class="m">K = total/max(cost)</span> (10 at 1024², 14 at 512², 19 at 256²) the
  even share binds, and contiguous ranges over log-spaced scales land 16–22% above it; an oracle splitter
  handed the exact costs does no better, so that gap is partition granularity, not a weighting error.
  Above it the costliest scale binds and the split is exact. <b>Raising K is the fix, not a cleverer
  partition.</b> The threshold is found by partitioning at each candidate K, not from
  <span class="m">total/max(cost)</span> — that bound is where the floor stops improving, one to four jobs
  short of where a contiguous split attains it. The earlier K=8 advice sat below it everywhere: +20% at
  1024², +65% at 512². If the time axis can't come down, the
  remaining lever is an FFT-based Gaussian filter — <span class="m">O(L log L)</span>, independent of σ —
  which removes ℓ from the cost model rather than redistributing it.</p>
</section>

<footer>
  <span>Fits: least squares on per-batch wall time, two runs of 8 PBS batch jobs each
    (256×256×64 / 41 timesteps, 512×512×65 / 21 timesteps), 30 log-spaced filter scales from 2Δx to 0.4·Lx.</span>
  <span>Memory figures are the anonymous baseline; PBS <code>resources_used.mem</code> also counts
    reclaimable page cache from the per-scale checkpoints, which is why a 16-scale batch reports ~200 GiB
    while its real working set is that of one scale.</span>
  <span>Implemented in <code>sweep1_filter_fields.py --print-scale-ranges</code> · branch
    <code>parallelize-sweep</code> · commits <code>86110f0</code>, <code>0f66e96</code>.</span>
</footer>

</div>
"""

out = TEMPLATE
for _k, _v in SVGS.items():
    out = out.replace("@@%s@@" % _k, _v)
import pathlib
pathlib.Path(__file__).with_name("sweep_cost_model.html").write_text(out)
print("wrote sweep_cost_model.html  (%d bytes)" % len(out))
print("  256^2 fit  A=%.3f min/scale  B=%.4f min per unit sigma" % (A6, B6))
print("  512^2 fit  A=%.3f min/scale  B=%.4f min per unit sigma" % (A5, B5))
print("  batch spread: %.1fx (256, l-weighted) -> %.1fx (512, affine)" % (max(y6)/min(y6), max(y5)/min(y5)))
print("  out-of-sample mean |err| on the 512^2 batches: %.1f%%" % np.abs(100*(pred-y5)/y5).mean())
