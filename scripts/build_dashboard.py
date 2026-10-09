"""Create a self-contained offline HTML viewer for real Spark results."""
import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = pd.read_csv(ROOT / 'reports/tables/station_year.csv')
    boundary = json.loads((ROOT / 'reports/assets/singapore_boundary.geojson').read_text())
    overview = json.loads((ROOT / 'reports/task1_overview.json').read_text())
    findings = json.loads((ROOT / 'reports/task1_findings.json').read_text())
    data = json.dumps({'stations': json.loads(rows.to_json(orient='records')),
                       'boundary': boundary, 'overview': overview, 'findings': findings}, ensure_ascii=False).replace('<', '\\u003c')
    gallery = []
    captions = {
        '02_station_comparison': ('站点比较', '相同年度内比较雨频与雨强；地图和表格可以查看所有站点。'),
        '03_monthly_patterns': ('月度规律', '把“经常下雨”和“下雨时更猛烈”分开观察。'),
        '04_year_month_heatmap': ('年份 × 月份', '比较雨量时，同时查看观测覆盖率。'),
        '05_annual_changes': ('年度变化', '固定站点样本，避免监测网络变化影响年际比较。'),
        '06_diurnal_patterns': ('一天内的规律', '全部使用新加坡时间，并查看不同月份的日内模式。'),
        '07_data_coverage': ('数据覆盖情况', '低覆盖率不等于少雨；缺失记录没有补成零。'),
        '08_spatial_correlation': ('站点之间的关联', '比较日降雨相关性，并观察关联是否随站点距离变化。'),
        '09_distribution_and_persistence': ('降雨分布与持续性', '未来半小时超阈值比例和短时持续性是描述性统计，不是模型预测。'),
    }
    for stem, (title, caption) in captions.items():
        encoded = base64.b64encode((ROOT / f'reports/figures/{stem}.png').read_bytes()).decode()
        gallery.append(f'<details><summary>{title}</summary><p>{caption}</p><img alt="{title}" src="data:image/png;base64,{encoded}"></details>')
    html = TEMPLATE.replace('__DATA__', data).replace('__GALLERY__', '\n'.join(gallery))
    (ROOT / 'reports/task1_dashboard.html').write_text(html, encoding='utf-8')
    print('Offline dashboard ready', flush=True)


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Singapore Rainfall · Task 1</title>
<style>
:root{--ink:#20324a;--muted:#687a8f;--teal:#008b83;--line:#dde5ed}*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:var(--ink);font:15px/1.6 system-ui,-apple-system,sans-serif}.wrap{max-width:1240px;margin:auto;padding:32px 24px}header{margin-bottom:24px}.eyebrow{letter-spacing:.14em;color:var(--teal);font-size:12px;font-weight:700}h1{font-size:36px;margin:6px 0}p{margin:8px 0;color:var(--muted)}.cards{display:flex;gap:14px;margin:20px 0}.card{flex:1;padding:16px 20px;background:white;border:1px solid var(--line);border-radius:12px}.card strong{font-size:28px;display:block}.controls{display:flex;flex-wrap:wrap;align-items:end;gap:16px;margin:18px 0}label{display:block;font-size:12px;color:var(--muted)}select,input{font:inherit;padding:8px 12px;border:1px solid #c8d4df;background:white;border-radius:8px;color:var(--ink)}.panel{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0}.mapgrid{display:grid;grid-template-columns:2fr 1fr;gap:18px}svg{width:100%;height:auto}.note{font-size:12px;color:var(--muted)}#tip{display:none;position:fixed;z-index:10;pointer-events:none;background:#20324a;color:white;padding:10px 13px;border-radius:8px;max-width:300px;font-size:12px;box-shadow:0 4px 18px #20324a33}.tablewrap{max-height:550px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}th{position:sticky;top:0;background:#f4f8fb;font-weight:600}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.low{color:#9aa6b4}tr:hover{background:#f2f9f8;cursor:pointer}.station{font-size:22px;font-weight:700;margin-bottom:4px}.legend{display:flex;align-items:center;gap:12px;font-size:12px}.gradient{width:150px;height:12px;border-radius:8px;background:linear-gradient(90deg,#440154,#21918c,#fde725)}details{background:white;border:1px solid var(--line);border-radius:12px;margin:14px 0;padding:15px 20px}summary{font-weight:650;font-size:18px;cursor:pointer}details img{width:100%;height:auto;margin-top:12px}footer{font-size:12px;margin-top:24px;color:var(--muted)}a{color:#2563a6}@media(max-width:800px){.mapgrid{grid-template-columns:1fr}.cards{flex-wrap:wrap}.card{min-width:150px}h1{font-size:28px}.wrap{padding:18px 12px}}
</style></head><body><main class="wrap">
<header><div class="eyebrow">NEA DATA · 2017—2024 · APACHE SPARK</div><h1>新加坡降雨的时间与空间规律</h1><p>Task 1 全量探索分析。地图上的点代表雨量观测站，可切换年份、指标，并点击站点查看年度变化。</p></header>
<div class="cards"><div class="card">有效五分钟记录<strong id="records"></strong></div><div class="card">固定比较站点<strong id="panelcount"></strong></div><div class="card">无降雨区间占比<strong id="dry"></strong></div></div>
<div class="panel"><div class="controls"><div><label for="year">年份</label><select id="year"></select></div><div><label for="metric">地图指标</label><select id="metric"><option value="wet_frequency_pct">降雨频率（%）</option><option value="wet_intensity_mm_5min">下雨时强度（mm / 5分钟）</option><option value="rain_per_observed_24h_mm">按观测时长归一化雨量（mm / 24h）</option><option value="coverage">年度观测覆盖率（%）</option></select></div><div><label for="search">筛选站点</label><input id="search" placeholder="站点 ID 或名称"></div></div>
<div class="mapgrid"><div><svg id="map" viewBox="0 0 850 560" role="img" aria-label="新加坡雨量站地图"></svg><div class="legend"><span id="min"></span><div class="gradient"></div><span id="max"></span><span id="unit"></span></div><p class="note">彩色点：年度覆盖率 ≥90%。灰色叉：覆盖不足，仅显示位置。颜色范围在不同年份间保持一致。<br>地图展示站点观测，不做空间插值；2016 年陆地边界仅作地理参照。</p></div><div><div id="stationname" class="station">点击任一站点</div><p id="stationdetail">查看站点覆盖率与八年降雨频率。</p><svg id="history" viewBox="0 0 350 270" aria-label="所选站点年度降雨频率"></svg><p class="note">频率 = 有雨的有效五分钟区间 / 所有有效区间。缺失不等于零；低覆盖率年份用灰色显示。</p></div></div></div>
<div class="panel"><h2 style="margin:0;font-size:20px">所有观测站比较</h2><p id="tablenote"></p><div class="tablewrap"><table><thead><tr><th>站点</th><th>名称</th><th>覆盖率</th><th>有雨区间</th><th>湿区间强度</th><th>雨量 / 观测24h</th><th>异常值标记</th></tr></thead><tbody id="rows"></tbody></table></div><p class="note">表格按当前地图指标降序排列。异常标记指 >50 mm / 5分钟读数；这不是官方质量阈值，主分析保留这些记录。</p></div>
<h2 style="margin:28px 0 8px">时间规律、空间关联与数据质量</h2>__GALLERY__
<footer>降雨数据：<a href="https://data.gov.sg/collections/2279/view">NEA / data.gov.sg</a>（官方提醒可能缺失，未经过气候数据质量控制）。<br>地理边界：<a href="https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/">geoBoundaries</a>，源自 URA 2016 边界，ODbL 1.0。<br>固定样本：八年均有观测，每年覆盖 ≥80%，经纬度跨度各 ≤0.005°。雨量 / 观测24h 并非官方日累计或年度总量；归一化无法纠正降雨时更容易缺测的情况。<br>代码与图表由 Codex 协助编写；统计结果由 Spark 实际计算，最终报告需记录和核验 AI 使用。</footer>
</main><div id="tip"></div><script>
const D=__DATA__,NS='http://www.w3.org/2000/svg';let selected=null;
const el=id=>document.getElementById(id),fmt=(v,n=2)=>v==null?'—':Number(v).toFixed(n);
function svg(tag,attrs,parent){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);parent.appendChild(n);return n}
const px=x=>45+(x-103.59)/.52*755,py=y=>510-(y-1.14)/.35*470;
function colour(x){const stops=[[68,1,84],[33,145,140],[253,231,37]],t=Math.max(0,Math.min(1,x))*2,i=Math.min(1,Math.floor(t)),f=t-i;return 'rgb('+stops[i].map((v,j)=>Math.round(v+(stops[i+1][j]-v)*f)).join(',')+')'}
function metricValue(r,m){return m==='coverage'?100*r[m]:r[m]}
function showStation(id){selected=id;const rs=D.stations.filter(r=>r.station_id===id).sort((a,b)=>a.year-b.year),last=rs[rs.length-1];el('stationname').textContent=id+' · '+last.station_name;el('stationdetail').textContent='经纬度 '+fmt(last.lon,4)+', '+fmt(last.lat,4)+'；有记录的年份 '+rs.length+' 个。';const s=el('history');s.innerHTML='';const max=Math.max(1,...rs.map(r=>r.wet_frequency_pct))*1.12;for(let y=0;y<=4;y++){let v=y*max/4;svg('line',{x1:40,y1:220-v/max*180,x2:335,y2:220-v/max*180,stroke:'#e5eaf0'},s);svg('text',{x:34,y:224-v/max*180,'text-anchor':'end',fill:'#687a8f','font-size':10},s).textContent=fmt(v,1)+'%'}for(let i=0;i<8;i++){let r=rs.find(r=>r.year===2017+i),x=45+i*36;svg('text',{x:x+10,y:240,'text-anchor':'middle','font-size':9,fill:'#687a8f'},s).textContent=String(2017+i).slice(2);if(r){let h=r.wet_frequency_pct/max*180;svg('rect',{x,y:220-h,width:22,height:h,rx:3,fill:r.coverage>=.9?'#008b83':'#a7b0bc'},s);svg('text',{x:x+10,y:213-h,'text-anchor':'middle','font-size':9,fill:'#20324a'},s).textContent=fmt(r.wet_frequency_pct,1)}}svg('text',{x:40,y:22,'font-size':12,fill:'#20324a'},s).textContent='各年有雨五分钟区间占比';render()}
function render(){const year=Number(el('year').value),metric=el('metric').value,q=el('search').value.toLowerCase(),s=el('map');s.innerHTML='';svg('rect',{x:0,y:0,width:850,height:560,fill:'#edf5fa',rx:8},s);for(const f of D.boundary.features){const polys=f.geometry.type==='Polygon'?[f.geometry.coordinates]:f.geometry.coordinates;for(const p of polys){const d=p[0].map((c,i)=>(i?'L':'M')+px(c[0]).toFixed(1)+','+py(c[1]).toFixed(1)).join(' ')+'Z';svg('path',{d,fill:'#e2e9dd',stroke:'#a2b5a3','stroke-width':.8},s)}}for(let x=103.6;x<=104.1;x+=.1){svg('line',{x1:px(x),y1:40,x2:px(x),y2:510,stroke:'#a7bfd2','stroke-opacity':.25},s);svg('text',{x:px(x),y:540,'text-anchor':'middle','font-size':11,fill:'#687a8f'},s).textContent=x.toFixed(1)+'°E'}for(let y=1.2;y<=1.4;y+=.1){svg('line',{x1:45,y1:py(y),x2:800,y2:py(y),stroke:'#a7bfd2','stroke-opacity':.25},s);svg('text',{x:10,y:py(y)+4,'font-size':11,fill:'#687a8f'},s).textContent=y.toFixed(1)+'°N'}const eligible=D.stations.filter(r=>r.coverage>=.9&&r[metric]!=null),lo=metric==='coverage'?0:Math.min(...eligible.map(r=>metricValue(r,metric))),hi=metric==='coverage'?100:Math.max(...eligible.map(r=>metricValue(r,metric)));el('min').textContent=fmt(lo,1);el('max').textContent=fmt(hi,1);el('unit').textContent=metric.includes('pct')||metric==='coverage'?'%':metric==='wet_intensity_mm_5min'?'mm / 湿5分钟':'mm / 观测24h';const rows=D.stations.filter(r=>r.year===year&&(r.station_id+' '+r.station_name).toLowerCase().includes(q)).sort((a,b)=>metricValue(b,metric)-metricValue(a,metric));for(const r of rows){if(!(r.lon>=103.59&&r.lon<=104.11&&r.lat>=1.14&&r.lat<=1.49))continue;const good=r.coverage>=.9,c=good?colour((metricValue(r,metric)-lo)/(hi-lo||1)):'#a7b0bc';let n;if(good){n=svg('circle',{cx:px(r.lon),cy:py(r.lat),r:r.station_id===selected?8:5.5,fill:c,stroke:r.station_id===selected?'#20324a':'white','stroke-width':r.station_id===selected?2:1},s)}else{n=svg('path',{d:`M${px(r.lon)-4},${py(r.lat)-4}l8,8m-8,0l8,-8`,stroke:c,'stroke-width':2,fill:'none'},s)}n.style.cursor='pointer';n.addEventListener('click',()=>showStation(r.station_id));n.addEventListener('mousemove',e=>{const t=el('tip');t.style.display='block';t.style.left=Math.min(e.clientX+14,innerWidth-310)+'px';t.style.top=e.clientY+10+'px';t.textContent=r.station_id+' · '+r.station_name+' | 有雨 '+fmt(r.wet_frequency_pct)+'% | 强度 '+fmt(r.wet_intensity_mm_5min)+' mm/5min | 覆盖 '+fmt(r.coverage*100,1)+'%'});n.addEventListener('mouseleave',()=>el('tip').style.display='none')}
el('rows').innerHTML='';for(const r of rows){let tr=document.createElement('tr');if(r.coverage<.9)tr.className='low';for(const v of[r.station_id,r.station_name,fmt(r.coverage*100,1)+'%',fmt(r.wet_frequency_pct)+'%',fmt(r.wet_intensity_mm_5min),fmt(r.rain_per_observed_24h_mm),String(r.n_flagged_over50)]){let td=document.createElement('td');td.textContent=v;tr.appendChild(td)}tr.onclick=()=>showStation(r.station_id);el('rows').appendChild(tr)}el('tablenote').textContent=year+' 年：'+rows.length+' 个站点；其中 '+rows.filter(r=>r.coverage>=.9).length+' 个达到 90% 覆盖率。灰色行不用于地图上的可靠排名。';svg('text',{x:780,y:35,'font-size':15,fill:'#20324a'},s).textContent='N ↑';svg('line',{x1:70,y1:490,x2:70+10/111.32/.52*755,y2:490,stroke:'#20324a','stroke-width':3},s);svg('text',{x:70,y:480,'font-size':11,fill:'#20324a'},s).textContent='10 km'}
for(let y=2017;y<=2024;y++){let o=document.createElement('option');o.value=y;o.textContent=y;el('year').appendChild(o)}el('year').value=2024;el('records').textContent=D.overview.clean_rows_2017_2024.toLocaleString();el('panelcount').textContent=D.findings.panel_stations;el('dry').textContent=fmt(D.findings.overall_dry_interval_pct,1)+'%';['year','metric'].forEach(id=>el(id).onchange=render);el('search').oninput=render;render();showStation(D.overview.balanced_panel_station_ids[0]);
</script></body></html>'''


if __name__ == '__main__':
    main()
