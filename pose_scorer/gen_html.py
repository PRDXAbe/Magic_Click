import json, os

OUTPUT_DIR = r"C:\projects\cam1 - Copy\captured_images\d46607ce-cb32-4cce-9196-5e70019f6ee5\scorer_output\pas_img"
IMAGE_DIR  = r"C:\projects\MagikClick-main (3)\MagikClick-main\passed_images"
OUT_FILE   = os.path.join(OUTPUT_DIR, "results_simple.html")

with open(os.path.join(OUTPUT_DIR, "results.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    img_abs = os.path.join(IMAGE_DIR, item["image"]).replace("\\", "/")
    item["_src"] = "file:///" + img_abs

json_str = json.dumps(data, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Scorer Results · d46607ce</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
:root{{--bg:#0d0d14;--card:#141420;--border:#252535;--accent:#8b6eff;--gold:#f5c842;--green:#2dd4a0;--red:#f87171;--blue:#60a5fa;--muted:#55566a;--text:#dde1f0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:2rem}}
h1{{font-size:1.6rem;font-weight:800;background:linear-gradient(90deg,#fff,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.sub{{color:var(--muted);font-size:.85rem;margin-top:.3rem}}
.topbar{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:2rem;flex-wrap:wrap;gap:1rem}}
.pills{{display:flex;gap:.5rem;flex-wrap:wrap}}
.pill{{background:var(--card);border:1px solid var(--border);border-radius:99px;padding:4px 14px;font-size:.75rem;font-weight:600}}
.pill span{{font-weight:800}}
.green{{color:var(--green)}}.red{{color:var(--red)}}.purple{{color:var(--accent)}}
.tabs{{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:1.5rem}}
.tab{{background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);font-family:inherit;font-size:.88rem;font-weight:600;padding:.6rem 1rem;cursor:pointer;transition:color .15s,border-color .15s}}
.tab.active{{color:var(--accent);border-bottom-color:var(--accent)}}
.panel{{display:none}}.panel.active{{display:block}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:1rem}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden;cursor:pointer;transition:transform .18s,border-color .18s,box-shadow .18s}}
.card:hover{{transform:translateY(-4px);border-color:var(--accent);box-shadow:0 8px 28px rgba(139,110,255,.18)}}
.card.r1{{border-color:var(--gold)}}.card.r2{{border-color:#b0bec5}}.card.r3{{border-color:#b87333}}
.thumb{{width:100%;aspect-ratio:3/4;object-fit:cover;display:block;background:#1a1a2a}}
.info{{padding:.75rem .9rem}}
.rank-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:.25rem}}
.rlabel{{font-size:.68rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.band{{font-size:.65rem;font-weight:700;padding:2px 8px;border-radius:99px}}
.Excellent{{background:rgba(245,200,66,.15);color:var(--gold);border:1px solid rgba(245,200,66,.3)}}
.Good{{background:rgba(45,212,160,.12);color:var(--green);border:1px solid rgba(45,212,160,.3)}}
.Acceptable{{background:rgba(96,165,250,.12);color:var(--blue);border:1px solid rgba(96,165,250,.3)}}
.Poor{{background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.3)}}
.final{{font-size:1.5rem;font-weight:800}}
.subscores{{display:flex;gap:.5rem;margin-top:.4rem;flex-wrap:wrap}}
.sscore{{font-size:.68rem;font-weight:600;color:var(--muted)}}.sscore span{{color:var(--text);font-weight:700}}
.barwrap{{margin-top:.5rem;height:3px;background:var(--border);border-radius:99px;overflow:hidden}}
.bar{{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),#c084fc)}}
.rgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.75rem}}
.rcard{{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden;opacity:.6;transition:opacity .15s}}
.rcard:hover{{opacity:1}}
.rthumb{{width:100%;aspect-ratio:3/4;object-fit:cover;display:block;filter:grayscale(60%);background:#1a1a2a}}
.rinfo{{padding:.5rem .7rem;font-size:.68rem}}
.rreason{{color:var(--red);font-weight:700;text-transform:capitalize}}
.rname{{color:var(--muted);margin-top:2px;word-break:break-all}}
.overlay{{position:fixed;inset:0;background:rgba(0,0,0,.88);backdrop-filter:blur(8px);z-index:50;display:flex;align-items:center;justify-content:center;padding:1.5rem;opacity:0;pointer-events:none;transition:opacity .2s}}
.overlay.open{{opacity:1;pointer-events:all}}
.modal{{background:var(--card);border:1px solid var(--border);border-radius:18px;display:flex;max-width:820px;width:100%;max-height:88vh;overflow:hidden;transform:scale(.95);transition:transform .2s;position:relative}}
.overlay.open .modal{{transform:scale(1)}}
.mimg{{flex:0 0 280px}}.mimg img{{width:100%;height:100%;object-fit:cover;display:block;border-radius:18px 0 0 18px;background:#1a1a2a}}
.mbody{{flex:1;padding:1.8rem;overflow-y:auto}}
.mclose{{position:absolute;top:.8rem;right:.8rem;background:#1e1e30;border:1px solid var(--border);color:var(--text);border-radius:8px;padding:4px 12px;font-size:.8rem;cursor:pointer;font-family:inherit}}
.mscore{{font-size:3.5rem;font-weight:800;background:linear-gradient(135deg,var(--accent),#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}}
.shdr{{font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:1.2rem 0 .6rem;padding-bottom:.35rem;border-bottom:1px solid var(--border)}}
.mrow{{display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem}}
.mname{{flex:0 0 130px;font-size:.78rem;color:var(--muted)}}
.mbarbg{{flex:1;height:6px;background:var(--border);border-radius:99px;overflow:hidden}}
.mbarfill{{height:100%;border-radius:99px}}
.msc{{flex:0 0 38px;text-align:right;font-size:.78rem;font-weight:700}}
.mskip{{flex:0 0 30px;font-size:.65rem;color:var(--border);text-align:right}}
.dtext{{font-size:.7rem;color:var(--muted);margin:2px 0 .4rem 130px}}
@media(max-width:600px){{.modal{{flex-direction:column}}.mimg{{flex:none}}.mimg img{{border-radius:18px 18px 0 0;max-height:220px}}body{{padding:1rem}}}}
</style>
</head>
<body>
<div class="topbar">
  <div><h1>Session Results</h1><div class="sub">d46607ce &nbsp;·&nbsp; {len(data)} images</div></div>
  <div class="pills" id="pills"></div>
</div>
<div class="tabs">
  <button class="tab active" onclick="sw('s',this)">Scored</button>
  <button class="tab" onclick="sw('r',this)">Rejected</button>
</div>
<div id="panel-s" class="panel active"><div class="grid" id="grid"></div></div>
<div id="panel-r" class="panel"><div class="rgrid" id="rgrid"></div></div>
<div class="overlay" id="ov" onclick="bg(event)">
  <div class="modal">
    <button class="mclose" onclick="close_()">✕</button>
    <div class="mimg" id="mi"></div>
    <div class="mbody" id="mb"></div>
  </div>
</div>
<script>
const ALL={json_str};
const scored=ALL.filter(x=>x.status==='SCORED').sort((a,b)=>a.rank-b.rank);
const rejected=ALL.filter(x=>x.status!=='SCORED');
const withFace=scored.filter(x=>x.face_group&&x.face_group.group_score!=null).length;
document.getElementById('pills').innerHTML=`
<div class="pill">Scored <span class="green">${{scored.length}}</span></div>
<div class="pill">With face <span class="purple">${{withFace}}</span></div>
<div class="pill">Rejected <span class="red">${{rejected.length}}</span></div>`;
function fmt(v,d=1){{return v==null?'—':Number(v).toFixed(d);}}
function mc(s){{if(s==null)return'#333';if(s>=80)return'#2dd4a0';if(s>=50)return'#60a5fa';return'#f87171';}}
document.getElementById('grid').innerHTML=scored.map(r=>`
<div class="card ${{r.rank<=3?'r'+r.rank:''}}" onclick="open_('${{r.image}}')">
<img class="thumb" src="${{r._src}}" loading="lazy" onerror="this.style.opacity='.2'" />
<div class="info">
<div class="rank-row"><span class="rlabel">Rank #${{r.rank}}</span><span class="band ${{r.score_band}}">${{r.score_band}}</span></div>
<span class="final">${{fmt(r.final_score)}}</span>
<div class="subscores">
<div class="sscore">Face <span>${{r.face_group?.group_score!=null?fmt(r.face_group.group_score):'—'}}</span></div>
<div class="sscore">Body <span>${{r.body_group?.group_score!=null?fmt(r.body_group.group_score):'—'}}</span></div>
</div>
<div class="barwrap"><div class="bar" style="width:${{r.final_score}}%"></div></div>
</div></div>`).join('');
document.getElementById('rgrid').innerHTML=rejected.map(r=>`
<div class="rcard">
<img class="rthumb" src="${{r._src}}" loading="lazy" onerror="this.style.display='none'" />
<div class="rinfo"><div class="rreason">${{(r.reject_reason||r.status||'').replace(/_/g,' ')}}</div><div class="rname">${{r.image.substring(0,22)}}…</div></div>
</div>`).join('');
function open_(name){{
  const r=ALL.find(x=>x.image===name);
  if(!r||r.status!=='SCORED')return;
  document.getElementById('mi').innerHTML=`<img src="${{r._src}}" />`;
  function mrows(g,lbl){{
    if(!g?.modules)return'';
    const gs=g.group_score!=null?fmt(g.group_score):'—';
    const rows=Object.entries(g.modules).map(([k,m])=>{{
      const s=m.skipped?null:m.score;const col=mc(s);
      return `<div class="mrow"><div class="mname">${{k.replace(/_/g,' ')}}</div>
<div class="mbarbg"><div class="mbarfill" style="width:${{s??0}}%;background:${{col}}"></div></div>
<div class="msc" style="color:${{col}}">${{s!=null?fmt(s,0):'—'}}</div>
${{m.skipped?'<div class="mskip">skip</div>':''}}</div>
${{m.detail&&!m.skipped?`<div class="dtext">${{m.detail}}</div>`:''}}`;
    }}).join('');
    return `<div class="shdr">${{lbl}} · ${{gs}}</div>${{rows}}`;
  }}
  document.getElementById('mb').innerHTML=`
<div class="mscore">${{fmt(r.final_score,2)}}</div>
<div style="margin:.3rem 0 1rem"><span class="band ${{r.score_band}}">${{r.score_band}}</span>&nbsp;&nbsp;<span style="font-size:.75rem;color:var(--muted)">Rank #${{r.rank}}</span></div>
${{mrows(r.face_group,'Face Group')}}
${{mrows(r.body_group,'Body Group')}}
<div class="shdr">Frame</div>
<div style="font-size:.75rem;color:var(--muted)">Offset score: ${{fmt(r.frame_check?.offset_score)}} &nbsp;·&nbsp; Centre: (${{fmt(r.frame_check?.body_centre?.x,3)}}, ${{fmt(r.frame_check?.body_centre?.y,3)}})</div>
<div class="shdr">File</div><div style="font-size:.7rem;color:var(--muted);word-break:break-all">${{r.image}}</div>`;
  document.getElementById('ov').classList.add('open');
}}
function close_(){{document.getElementById('ov').classList.remove('open');}}
function bg(e){{if(e.target===document.getElementById('ov'))close_();}}
function sw(p,btn){{document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));btn.classList.add('active');document.getElementById('panel-'+p).classList.add('active');}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')close_();}});
</script>
</body></html>"""

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Done. {os.path.getsize(OUT_FILE)//1024} KB written to {OUT_FILE}")
