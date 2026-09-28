"""Emit the RoboTwin 2.0 dataset table artifact."""
import base64, json, os, sys
import cv2

R="/coc/flash7/bli678/Projects/RoboTwin"
S="/tmp/claude-3582647/-coc-flash7-bli678-Projects-XPolicyLab/993275a1-dd44-456a-8fd3-8323e78492e9/scratchpad"
DD="/coc/flash7/scratch/robotwin_shared/dataset"
EMB=[("aloha-agilex","Aloha"),("arx-x5","ARX-X5"),("franka","Franka"),("piper","Piper"),("ur5","UR5")]
lb={r["task"]:r for r in json.load(open(f"{S}/act_leaderboard.json"))}
tasks=sorted(os.listdir(DD))

def thumb(t,w=168):
    return None  # images intentionally omitted from this table
    p=f"{R}/figs/tasks/{t}_home.png"
    if not os.path.exists(p): return None
    im=cv2.imread(p)
    if im is None: return None
    im=cv2.resize(im,(w,int(im.shape[0]*w/im.shape[1])),interpolation=cv2.INTER_AREA)
    ok,b=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,80])
    return base64.b64encode(b).decode() if ok else None

rows=[]
for t in tasks:
    have=set(os.listdir(os.path.join(DD,t)))
    rows.append(dict(t=t, d=lb[t]["display"],
                     sup=[e for e,_ in EMB if f"{e}_clean_50" in have],
                     easy=lb[t]["act_easy"], hard=lb[t]["act_hard"], img=thumb(t)))
n_img=sum(1 for r in rows if r["img"])
cov={e:sum(1 for r in rows if e in r["sup"]) for e,_ in EMB}
full=sum(1 for r in rows if len(r["sup"])==5)
mean_easy=sum(r["easy"] for r in rows)/len(rows)
mean_hard=sum(r["hard"] for r in rows)/len(rows)

def cell(r,e):
    return ('<td class="c y">&#10003;</td>' if e in r["sup"] else '<td class="c n">&middot;</td>')
def heat(v):
    # 0 -> flat, 100 -> full accent
    a=max(0.0,min(1.0,v/100.0))
    return f'style="--h:{a:.3f}"'

rows.sort(key=lambda r: (-r["easy"], -r["hard"], r["t"]))
trs=[]
for r in rows:
    trs.append(
      f'<tr>'
      f'<td class="tk"><b>{r["d"]}</b><span>{r["t"]}</span></td>'
      + "".join(cell(r,e) for e,_ in EMB)
      + f'<td class="c num n5">{len(r["sup"])}</td>'
      f'<td class="num sc" {heat(r["easy"])}>{r["easy"]:.0f}</td>'
      f'<td class="num sc dim" {heat(r["hard"])}>{r["hard"]:.0f}</td></tr>')

head="".join(f"<th class='c'>{lbl}</th>" for _,lbl in EMB)
html=f"""<title>RoboTwin 2.0 Task Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{--bg:#F2F4F3;--card:#FFF;--alt:#E8ECEA;--ln:#CFD8D4;--lns:#E2E8E5;--ink:#111917;--mid:#4C5A56;--soft:#71827D;
--acc:#0E7C6B;--accl:#0A5F52;--hot:#B4531F;--ok:#1F7A4C;--no:#B9C4C0;
--fd:"Chivo","Helvetica Neue",Arial,sans-serif;--fb:"IBM Plex Sans",system-ui,sans-serif;--fm:"IBM Plex Mono",ui-monospace,monospace;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0B1211;--card:#131C1A;--alt:#182220;--ln:#283835;--lns:#1D2927;
--ink:#DEE8E5;--mid:#9DB0AB;--soft:#7A8D88;--acc:#3FBFA7;--accl:#6BD9C4;--hot:#E08A52;--ok:#5FC98F;--no:#3A4A46;}}}}
:root[data-theme="dark"]{{--bg:#0B1211;--card:#131C1A;--alt:#182220;--ln:#283835;--lns:#1D2927;
--ink:#DEE8E5;--mid:#9DB0AB;--soft:#7A8D88;--acc:#3FBFA7;--accl:#6BD9C4;--hot:#E08A52;--ok:#5FC98F;--no:#3A4A46;}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--fb);margin:0;padding:0 18px 80px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.w{{max-width:1120px;margin:0 auto}}
header{{padding:52px 0 24px;border-bottom:2px solid var(--ink)}}
.eb{{font-family:var(--fm);font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--acc);margin:0 0 12px}}
h1{{font-family:var(--fd);font-weight:700;font-size:clamp(28px,4.6vw,42px);line-height:1.06;letter-spacing:-.02em;margin:0 0 12px;text-wrap:balance}}
.sf{{color:var(--mid);font-size:16.5px;max-width:64ch;margin:0}}
.kpis{{display:flex;flex-wrap:wrap;gap:9px;margin-top:20px}}
.kpi{{font-family:var(--fm);font-size:12.5px;border:1px solid var(--ln);background:var(--card);padding:7px 11px;display:flex;gap:9px;align-items:baseline}}
.kpi b{{color:var(--acc);font-weight:600}} .kpi span{{color:var(--soft)}}
h2{{font-family:var(--fd);font-weight:600;font-size:22px;margin:44px 0 6px;letter-spacing:-.01em}}
.nt{{color:var(--soft);font-size:14.5px;margin:0 0 18px;max-width:70ch}}
p{{max-width:70ch;font-size:15px}}
.sc-wrap{{overflow-x:auto;border:1px solid var(--lns);background:var(--card)}}
table{{border-collapse:collapse;width:100%;min-width:720px;font-size:14px}}
th,td{{padding:0 10px;border-bottom:1px solid var(--lns);text-align:left}}
thead th{{background:var(--alt);font-family:var(--fm);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--mid);padding:10px;position:sticky;top:0}}
th.c,td.c{{text-align:center}}
td.im{{width:96px;padding:5px 8px 3px}}
td.im img{{width:88px;height:66px;object-fit:cover;display:block;border:1px solid var(--lns);border-radius:2px;background:var(--alt)}}
.ph{{width:88px;height:66px;display:grid;place-items:center;border:1px dashed var(--ln);border-radius:2px;color:var(--no);font-size:17px}}
td.tk{{padding:8px 10px;line-height:1.32}}
td.tk b{{font-weight:600;font-size:14.5px;display:block}}
td.tk span{{font-family:var(--fm);font-size:11px;color:var(--soft)}}
td.y{{color:var(--ok);font-weight:700}} td.n{{color:var(--no)}}
.num{{font-family:var(--fm);font-variant-numeric:tabular-nums;text-align:right}}
td.n5{{color:var(--mid);font-size:12.5px}}
td.sc{{font-weight:600;padding:8px 12px;position:relative;color:var(--ink)}}
td.sc::before{{content:"";position:absolute;inset:3px 4px;background:var(--acc);opacity:calc(var(--h)*.30);border-radius:2px}}
td.sc.dim::before{{background:var(--hot)}}
td.sc>*{{position:relative}}
tbody tr:hover{{background:var(--alt)}}
footer{{margin-top:52px;padding-top:18px;border-top:1px solid var(--ln);font-family:var(--fm);font-size:11.5px;color:var(--soft);display:flex;flex-wrap:wrap;gap:8px 20px}}
code{{font-family:var(--fm);font-size:.88em;background:var(--alt);padding:1px 5px;border-radius:2px}}
</style>
<div class="w">
<header>
<p class="eb">RoboTwin 2.0 &middot; dataset composition</p>
<h1>Task Atlas</h1>
<p class="sf">All 50 benchmark tasks: which of the five embodiments ship demonstration data, and how the ACT
baseline scores on each. Embodiment support is read from the released <code>&lt;robot&gt;_clean_50</code>
collections on disk; scores are the published single-task SFT leaderboard.</p>
<div class="kpis">
<div class="kpi"><span>tasks</span> <b>50</b></div>
<div class="kpi"><span>all 5 embodiments</span> <b>{full}</b></div>
<div class="kpi"><span>ACT clean2clean</span> <b>{mean_easy:.1f}%</b></div>
<div class="kpi"><span>ACT clean2random</span> <b>{mean_hard:.1f}%</b></div>
</div>
</header>

<h2>The 50 tasks</h2>
<p class="nt">Ordered by <b>easy</b> (clean2clean) success, best first. <b>hard</b> = clean2random. Bar length tracks the score.</p>
<div class="sc-wrap"><table>
<thead><tr><th>task</th>{head}<th class="c">n</th><th class="num">easy</th><th class="num">hard</th></tr></thead>
<tbody>
{"".join(trs)}
</tbody></table></div>

<h2>Coverage</h2>
<div class="sc-wrap"><table style="min-width:420px">
<thead><tr><th>Embodiment</th><th class="num">Tasks</th><th class="num">Coverage</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{lbl}</td><td class="num">{cov[e]}</td><td class="num">{100*cov[e]/50:.0f}%</td></tr>' for e,lbl in EMB)}
</tbody></table></div>

<footer><span>50 tasks &middot; 5 embodiments</span><span>ACT = single-task SFT</span>
<span>leaderboard updated 2026-08-18</span><span>clean_50 = 50 demos/task/robot</span></footer>
</div>
"""
open(f"{S}/robotwin_atlas.html","w",encoding="utf-8").write(html)
print(f"wrote robotwin_atlas.html  ({len(html)/1024:.0f} KB, {n_img}/50 thumbnails)", file=sys.stderr)
