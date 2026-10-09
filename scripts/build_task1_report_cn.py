"""Generate a standalone Chinese LaTeX report from verified aggregate tables.

All nine figure groups use inline TikZ/PGFPlots data. The generated .tex needs
no image, CSV, font, or other project files beside standard TeX packages.
Compile with Tectonic or XeLaTeX. This script does not generate the PDF itself.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports'


def coords(xs, ys, meta=None):
    points = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        point = f'({float(x):.7f},{float(y):.7f})'
        if meta is not None:
            point += f'[{float(meta[i]):.7f}]'
        points.append(point)
    return ' '.join(points)


def plot(xs, ys, options='teal,thick', meta=None):
    return r'\addplot[' + options + '] coordinates {' + coords(xs, ys, meta) + '};\n'


def axis(body, options='', offset=0, width='11cm', height='7.5cm'):
    return (r'\begin{axis}[reportaxis,at={(' + str(offset) + 'cm,0)},anchor=south west,'
            f'width={width},height={height},' + options + ']\n' + body + r'\end{axis}' + '\n')


def picture(body):
    return '\n' + r'\begin{center}\begin{tikzpicture}' + '\n' + body + r'\end{tikzpicture}\end{center}' + '\n'


def page(title, body):
    return r'\clearpage\section*{' + title + '}\n' + body + '\n'


def table(headers, rows, spec):
    def esc(x):
        return str(x).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    return (r'\begin{center}\begin{tabular}{' + spec + '}\n' + r'\toprule' + '\n'
            + ' & '.join(esc(h) for h in headers) + r'\\\midrule' + '\n'
            + '\n'.join(' & '.join(esc(x) for x in row) + r'\\' for row in rows)
            + '\n' + r'\bottomrule\end{tabular}\end{center}' + '\n')


def heat(matrix, xlabels, ylabels, low, high, offset=0, dx=.66, dy=.59, annotation=True):
    """Explicit vector heatmap; missing cells are grey, never zero-filled."""
    result = [r'\begin{scope}[xshift=' + str(offset) + 'cm]\n']
    nrows, ncols = matrix.shape
    for y in range(nrows):
        for x in range(ncols):
            value = float(matrix.iloc[y, x])
            if np.isfinite(value):
                t = float(np.clip((value-low)/(high-low), 0, 1))
                rgb = [(.94, .98, .78)[i]*(1-t)+(.06, .23, .49)[i]*t for i in range(3)]
                fill = '{rgb,1:red,' + f'{rgb[0]:.4f};green,{rgb[1]:.4f};blue,{rgb[2]:.4f}' + '}'
            else:
                t = 0
                fill = 'missing'
            x0, y0 = x*dx, (nrows-1-y)*dy
            result.append(r'\fill[fill=' + fill + f'] ({x0:.3f},{y0:.3f}) rectangle ({x0+dx:.3f},{y0+dy:.3f});\n')
            if annotation and np.isfinite(value):
                colour = 'white' if t > .62 else 'ink'
                result.append(r'\node[font=\scriptsize,text=' + colour + f'] at ({x0+dx/2:.3f},{y0+dy/2:.3f})' + '{' + f'{value:.1f}' + '};\n')
    for x, label in enumerate(xlabels):
        result.append(r'\node[font=\scriptsize,rotate=45,anchor=east] at (' + f'{(x+.5)*dx:.3f},-0.13)' + '{' + str(label) + '};\n')
    for y, label in enumerate(ylabels):
        result.append(r'\node[font=\scriptsize,anchor=east] at (-0.12,' + f'{(nrows-y-.5)*dy:.3f})' + '{' + str(label) + '};\n')
    result.append(r'\node[font=\footnotesize,anchor=west] at (0,-1.05){浅色 $\rightarrow$ 深色：' + f'{low:g}' + r' $\rightarrow$ ' + f'{high:g}' + r'；灰色为无合格观测};' + '\n')
    result.append(r'\end{scope}'+'\n')
    return ''.join(result)


def main():
    source = REPORT / 'Task1_Report_CN.tex'
    if source.exists() and '\\section*{4. Task 2' in source.read_text(encoding='utf-8'):
        raise SystemExit('The saved source now contains the reviewed full bilingual report. '
                         'Edit it in place; this legacy Task 1 generator would overwrite Task 2. '
                         'Use build_task1_report.py to rebuild the English source.')
    ov = json.loads((REPORT/'task1_overview.json').read_text())
    findings = json.loads((REPORT/'task1_findings.json').read_text())
    sy = pd.read_csv(REPORT/'tables/station_year.csv')
    monthly = pd.read_csv(REPORT/'tables/panel_month_year.csv')
    hourly = pd.read_csv(REPORT/'tables/panel_station_month_hour.csv')
    hourly = hourly[hourly.coverage >= .8]
    pairs = pd.read_csv(REPORT/'tables/station_correlation_pairs.csv')
    audits = [json.loads((REPORT/f'quality/{y}.json').read_text()) for y in range(2017,2025)]
    annual = pd.DataFrame(findings['annual_panel_metrics'])
    resource = [json.loads(line) for line in (REPORT/'resource_usage.jsonl').read_text().splitlines() if line]
    report = [PREAMBLE, INTRO]
    report.append(table(['年份','雨量 / 观测24h（mm）','雨频（%）','湿区间雨强（mm/5min）','平均覆盖率（%）'],
                        [[int(r.year),f'{r.rain_per_observed_24h_mm:.2f}',f'{r.wet_frequency_pct:.2f}',
                          f'{r.wet_intensity_mm_5min:.3f}',f'{r.coverage*100:.1f}'] for r in annual.itertuples()], 'rrrrr'))
    report.append(r'\noindent 表中各年均使用相同的 21 个站点，站点等权。雨量指标按观测时长归一化，不能解释为官方年度累计或真实日累计。月度图和日内图的合格样本组成仍可变化。')
    audit_table = table(['年份','源记录数','删除重复数','偏离网格超过2秒','负雨量数','站点数'],
                        [[a['source_year'],f"{a['raw_rows']:,}",a['duplicate_rows_removed'],a['off_grid_over_2_seconds'],a['negative_rain'],a['stations']] for a in audits], 'rrrrrr')
    report.append(page('1. 数据来源、字段与清洗', CLEANING + audit_table + CLEANING_END))
    report.append(page('2. 指标定义与公平比较', METRICS))
    report.append(page('3. 计算环境与结果核验', ENVIRONMENT.replace('__PEAK__',f"{max(r['rss_gib'] for r in resource):.2f}").replace('__DISK__',f"{min(r['free_disk_gib'] for r in resource):.1f}")))

    # Figure 1: maps retain the original reference coastline and real gauges.
    boundary = json.loads((REPORT/'assets/singapore_boundary.geojson').read_text())
    mapped = sy[sy.year == 2024]
    good, bad = mapped[mapped.coverage >= .9], mapped[mapped.coverage < .9]
    base = ''
    for feature in boundary['features']:
        geometry = feature['geometry']
        polys = [geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
        for poly in polys:
            ring = poly[0]
            base += r'\path[fill=land,draw=gray!60,line width=.2pt] ' + ' -- '.join(
                f'(axis cs:{p[0]:.7f},{p[1]:.7f})' for p in ring) + r' -- cycle;' + '\n'
    body = ''
    for i,(metric,title,label,cmap) in enumerate([
        ('wet_frequency_pct','下雨的频率','有雨五分钟区间（\\%）','viridis'),
        ('wet_intensity_mm_5min','下雨时的强度','mm / 湿五分钟','hot2')]):
        points = base + plot(good.lon,good.lat,'scatter,only marks,mark=*,mark size=2.5pt,scatter src=explicit,draw=white',good[metric].to_numpy())
        points += plot(bad.lon,bad.lat,'gray,only marks,mark=x,mark size=2.2pt')
        body += axis(points, f'title={{{title}}},xlabel={{经度（东经）}},ylabel={{纬度（北纬）}},'
                     'xmin=103.59,xmax=104.11,ymin=1.14,ymax=1.49,xtick={103.6,103.7,103.8,103.9,104.0,104.1},scaled ticks=false,axis equal image,'
                     f'point meta min={good[metric].min():.7f},point meta max={good[metric].max():.7f},'
                     f'colormap/{cmap},colorbar,colorbar style={{ylabel={{{label}}}}}',offset=12.8*i,width='10.7cm',height='8cm')
    report.append(page('图 1：2024 年观测站降雨地图', picture(body)+r'\noindent 2024 年有 70 个站点，其中 56 个年度覆盖率不低于 90\%。彩色点只表示该观测站的统计量；灰色叉为低覆盖率站点。S33（Jurong Pier Road）的雨频最高，为 3.55\%。地图不进行空间插值，也不代表岛上每个位置的降雨量。边界采用 geoBoundaries 中源自 URA 的 2016 年陆地轮廓（ODbL 1.0），仅作地理参照。'))

    # Figure 2: station frequency/intensity scatter and frequency extremes.
    body = axis(plot(good.wet_frequency_pct,good.wet_intensity_mm_5min,'teal,only marks,mark=*,mark size=2.3pt'),
                'title={雨频与雨强衡量不同现象},xlabel={有雨区间（\\%）},ylabel={mm / 湿五分钟}')
    ranked = pd.concat([good.nsmallest(10,'wet_frequency_pct'),good.nlargest(10,'wet_frequency_pct')]).drop_duplicates('station_id').sort_values('wet_frequency_pct')
    labels = ','.join(ranked.station_id)
    body += axis(plot(ranked.wet_frequency_pct,np.arange(len(ranked)),'xbar,fill=teal,draw=none,bar width=5pt'),
                 'title={雨频最高与最低的各10站},xlabel={有雨区间（\\%）},ytick={'+','.join(map(str,range(len(ranked))))+'},yticklabels={'+labels+'},xmin=0',offset=12.8,width='10.7cm',height='8cm')
    report.append(page('图 2：观测站之间的比较',picture(body)+r'\noindent 使用同一年度和不低于 90\% 的覆盖率门槛。雨频最高的是 S33，而湿区间雨强最高的是 S211（Kranji Road）。在该年的合格样本中，多处中部、西部站点的雨频较高，部分东部和沿海站点较低。频率高并不必然意味着每次下雨更猛烈。低覆盖率站点不能因观测雨量少就被视作干燥地区；测量误差和非随机缺测仍可能影响排名。'))

    # Figure 3: preserve missing month gaps instead of interpolating them.
    body = ''
    colours = ['blue!35','violet!55','blue!70','cyan!70!black','teal!65','green!55!black','lime!65!black','orange!70']
    for k,(metric,title,ylabel) in enumerate([('wet_frequency_pct','月度降雨频率','有雨区间（\\%）'),('wet_intensity_mm_5min','月度湿区间雨强','mm / 湿五分钟')]):
        lines=''
        for i,y in enumerate(range(2017,2025)):
            frame=monthly[monthly.year==y].set_index('month').reindex(range(1,13))
            lines += plot(frame.index,frame[metric],colours[i]+',thin,unbounded coords=jump')
            if k==0: lines += r'\addlegendentry{' + str(y) + '}\n'
        mean=monthly.groupby('month')[metric].mean()
        lines += plot(mean.index,mean,'ink,very thick,mark=*,mark size=1.6pt')
        if k==0:lines += r'\addlegendentry{可用年份平均}'+'\n'
        options=f'title={{{title}}},xlabel={{月份（新加坡时间）}},ylabel={{{ylabel}}},xmin=1,xmax=12,xtick={{1,...,12}}'
        if k==0: options += ',legend style={at={(.5,-.22)},anchor=north,legend columns=5,font=\\scriptsize,draw=none}'
        body+=axis(lines,options,offset=12.8*k,height='7cm')
    report.append(page('图 3：一年内的月度规律',picture(body)+r'\vspace{5mm}\noindent 从年度固定站点池中筛选每个年月覆盖率不低于 80\% 的站点，先站点等权，再对可用年份等权平均。11 月平均雨频最高（4.28\%），7 月最低（1.85\%）；3 月平均湿区间雨强约为 1.02 mm，而 1 月约为 0.67 mm。雨频和雨强可以呈现不同的季节模式。每月可用年份为 6--8 年，合格站点可变化，因此不能把这些均值当作完全平衡的季节比较。MSS 所述较湿的 11 月至次年 1 月提供背景；本数据不能独立证明季风因果关系。'))

    # Figure 4: year-month amount and eligible observation coverage.
    amt=monthly.pivot(index='year',columns='month',values='rain_per_observed_24h_mm').reindex(index=range(2017,2025),columns=range(1,13))
    cov=monthly.pivot(index='year',columns='month',values='coverage').reindex(index=range(2017,2025),columns=range(1,13))*100
    body=r'\node[anchor=west] at (0,5.4){雨量 / 观测24h（mm）};'+'\n'+heat(amt,range(1,13),range(2017,2025),0,14)
    body+=r'\node[anchor=west] at (12.8,5.4){合格样本平均覆盖率（\%）};'+'\n'+heat(cov,range(1,13),range(2017,2025),80,100,offset=12.8)
    report.append(page('图 4：年份与月份的交叉比较',picture(body)+r'\noindent 季节模式并非每年完全重复。灰格表示该年月没有达到门槛的站点，不能解释为没有降雨。96 个年月中有 89 个合格单元，每个保留单元有 17--21 个站点。右图是被保留样本的覆盖率，不代表被排除站点也有同样完整的数据。严格要求同一站点的某个月在八年都达到门槛时，仅 5、6、7、9、10、12 月可比较；该敏感性结果单独保存在 strict\_balanced\_month\_year.csv。'))

    # Figure 5: fixed annual panel.
    body=''
    for i,(metric,title,label,col) in enumerate([('rain_per_observed_24h_mm','雨量','mm / 观测24h','blue!70!black'),('wet_frequency_pct','雨频','有雨区间（\\%）','teal'),('wet_intensity_mm_5min','湿区间雨强','mm / 湿五分钟','orange!80!black')]):
        body+=axis(plot(annual.year,annual[metric],f'ybar,fill={col},draw=none,bar width=11pt,nodes near coords,nodes near coords style={{font=\\scriptsize}},point meta=y'),
                   f'title={{{title}}},xlabel={{年份}},ylabel={{{label}}},ymin=0,ymax={annual[metric].max()*1.2:.4f},'
                   'xtick={2017,...,2024},scaled ticks=false,xticklabel style={rotate=45,anchor=east},/pgf/number format/precision=2',offset=8.5*i,width='7cm',height='7.5cm')
    report.append(page('图 5：固定站点的年度变化',picture(body)+r'\noindent 相同 21 个站点、每站每年覆盖率至少 80\%。平均雨量 / 观测24h 从 2019 年的 4.81 mm 到 2021 年的 7.84 mm；雨频和湿区间雨强的变化并不完全同步。将总雨量拆成频率和强度，有助于解释年际差异。但覆盖率归一化不能恢复缺失风暴，2018 年平均覆盖率约为 82.9\%，明显低于多数年份。八年仅支持描述性比较，本报告不进行长期气候趋势或显著性断言。'))

    # Figure 6: equal weights on eligible station-year-month-hour strata.
    curve=hourly.groupby('hour').wet_frequency_pct.mean()
    body=axis(plot(curve.index,curve,'teal,very thick,mark=*,mark size=1.8pt'),'title={日内雨频变化},xlabel={小时（新加坡时间）},ylabel={有雨区间（\\%）},xtick={0,3,...,21},xmin=0,xmax=23')
    h=hourly.groupby(['month','hour']).wet_frequency_pct.mean().unstack().reindex(index=range(1,13),columns=range(24))
    body+=r'\node[anchor=west] at (12.8,5.2){月份与小时的雨频（\%）};'+'\n'+heat(h,range(24),range(1,13),0,10,offset=12.8,dx=.34,dy=.38,annotation=False)
    report.append(page('图 6：一天内何时更容易下雨',picture(body)+r'\noindent 雨频最高的小时为 16:00--17:00，右图显示该周期随月份变化；例如 6 月的高值时段更靠近上午至中午。筛选条件为站点、年份、月份、小时组合的覆盖率不低于 80\%，合格组合等权平均。不同组合的组成仍可变化。MSS 描述的下午降雨周期是背景信息；本分析没有额外温度、风或湿度观测来解释其机制。'))

    # Figure 7: coverage and changing station network.
    scatter=''
    for year in range(2017,2025):
        rows=sy[sy.year==year]
        scatter+=plot(year+np.linspace(-.2,.2,len(rows)),rows.coverage*100,'blue!55,only marks,mark=*,mark size=1.2pt')
    scatter+=plot([2016.6,2024.4],[80,80],'orange,dashed,thick')
    body=axis(scatter,'title={站点年度覆盖率},xlabel={年份},ylabel={覆盖率（\\%）},xtick={2017,...,2024},scaled ticks=false,ymin=0,ymax=105')
    counts=sy.groupby('year').station_id.nunique(); qualifying=sy[sy.coverage>=.9].groupby('year').station_id.nunique().reindex(counts.index,fill_value=0)
    bars=plot(counts.index-.16,counts,'ybar,fill=blue!30,draw=none,bar width=11pt')+r'\addlegendentry{有有效观测}'+'\n'
    bars+=plot(counts.index+.16,qualifying,'ybar,fill=teal,draw=none,bar width=11pt')+r'\addlegendentry{覆盖率至少90\%}'+'\n'
    body+=axis(bars,'title={监测网络随年份变化},xlabel={年份},ylabel={站点数},xtick={2017,...,2024},scaled ticks=false,ymin=0,ymax=90,legend style={font=\\scriptsize,draw=none}',offset=12.8)
    report.append(page('图 7：数据覆盖与监测网络变化',picture(body)+r'\noindent 分母采用完整日历年中的预期五分钟槽数，因此站点只在部分年份运行时也会体现为低覆盖率。年度站点数量从 2019 年的 54 个到 2020 年的 80 个，并不恒定。直接比较所有站点的年平均可能混入网络组成变化；固定站点池可减轻此问题，但也会排除缺少长期记录的位置。站点增删、搬迁和缺测原因需要更多元数据才能区分。'))

    # Figure 8: actual correlation matrix and all station pairs.
    corr=pd.read_csv(REPORT/'tables/station_daily_correlation.csv',index_col=0)
    body=heat(corr,corr.columns,corr.index,0,1,dx=.29,dy=.29,annotation=False)
    points=plot(pairs.distance_km,pairs.daily_rain_pearson_r,'teal!60,only marks,mark=*,mark size=1.4pt')
    bins=pairs.groupby(pd.cut(pairs.distance_km,np.arange(0,46,5)),observed=True).agg(distance=('distance_km','mean'),correlation=('daily_rain_pearson_r','mean'))
    points+=plot(bins.distance,bins.correlation,'ink,very thick,mark=*,mark size=1.4pt')
    body+=axis(points,'title={距离与日雨量相关性},xlabel={站点距离（km）},ylabel={Pearson相关系数},ymin=.15,ymax=1,xmin=0,xmax=45',offset=12.8,width='10.7cm',height='8cm')
    report.append(page('图 8：站点之间的空间关联',picture(body)+r'\noindent 日雨量相关性仅使用日覆盖率不低于 95\% 的记录；每对站点至少需要 365 个共同日期。21 站点形成 210 对，相关性中位数为 0.534。不超过 5 km 的 17 对平均相关性为 0.809，超过 30 km 的 12 对为 0.336；右图深线表示每 5 km 距离分组的均值。不同站点对的日期区间可不同，干燥日也会贡献相关性，站点对之间不是独立样本。距离用最新坐标计算球面近似，结果不能证明雨云移动轨迹或因果关系。'))

    # Figure 9: rainfall distribution and future-window prevalence are descriptive.
    distribution=pd.read_csv(REPORT/'tables/rain_distribution.csv').groupby('rain_bin')['count'].sum().sort_index()
    wet=distribution.iloc[1:];proportions=100*wet/wet.sum()
    wetlabels=','.join('{'+label.split(': ',1)[-1]+'}' for label in wet.index)
    body=axis(plot(range(len(wet)),proportions,'ybar,fill=blue!70!black,draw=none,bar width=12pt'),'title={有雨区间的雨量分布},ylabel={湿区间占比（\\%）},xtick={'+','.join(map(str,range(len(wet))))+'},xticklabels={'+wetlabels+'},xticklabel style={rotate=35,anchor=east},ymin=0',width='7cm',height='7.2cm')
    half=ov['halfhour_2024_descriptive_only']
    prevalence=[100*half['p_gt_0mm'],100*half['p_gt_1mm'],100*half['p_gt_5mm'],100*half['p_gt_10mm'],100*half['p_gt_20mm']]
    body+=axis(plot(range(5),prevalence,'ybar,fill=teal,draw=none,bar width=12pt'),'title={未来六个观测槽超阈值},xlabel={阈值（mm）},ylabel={窗口比例（\\%）},xtick={0,...,4},xticklabels={0,1,5,10,20},ymin=0',offset=8.5,width='7cm',height='7.2cm')
    pers=pd.DataFrame(findings['rain_persistence_2024']).sort_values('previous_wet')
    body+=axis(plot(range(2),pers.current_wet_pct,'ybar,fill=orange!80!black,draw=none,bar width=24pt'),'title={短时降雨持续性},ylabel={当前区间有雨（\\%）},xtick={0,1},xticklabels={前一段无雨,前一段有雨},ymin=0',offset=17,width='7cm',height='7.2cm')
    report.append(page('图 9：降雨分布、持续性与未来阈值',picture(body)+r'\noindent 八年全部有效观测中有 97.08\% 的五分钟区间无雨，这不是“97\% 的日子无雨”。2024 年相邻且恰好相差五分钟的观测中，前一段有雨时下一段有雨的比例为 67.03\%，前一段无雨时为 1.04\%。6,374,476 个完整未来六槽窗口中，严格超过 5 mm 的比例为 0.84\%。六个槽的标签覆盖 30 分钟，但物理测量区间的起止语义仍未确认，重叠窗口也不独立。这些是观测关联及目标发生比例，不是已训练模型的成绩。'))

    report.append(page('4. 限制、额外观察与 Task 2 衔接', LIMITATIONS))
    report.append(page('5. AI 使用声明、来源与复现', DECLARATION))
    report.append(r'\end{document}'+'\n')
    target=REPORT/'Task1_Report_CN.tex'
    from report_edits import revise
    target.write_text(revise(''.join(report)),encoding='utf-8')
    print('LaTeX source:',target,'bytes:',target.stat().st_size)


PREAMBLE = r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper,landscape]{ctexart}
\usepackage[margin=18mm,headheight=16pt,headsep=5mm,footskip=9mm]{geometry}
\usepackage{amsmath,booktabs,array,xcolor,hyperref,tikz,pgfplots,fancyhdr}
\pgfplotsset{compat=1.18}
\definecolor{ink}{HTML}{20324A}
\definecolor{teal}{HTML}{008B83}
\definecolor{missing}{HTML}{D9DFE6}
\definecolor{land}{HTML}{E2E9DD}
\hypersetup{unicode=true,colorlinks=true,linkcolor=teal,urlcolor=teal,pdftitle={新加坡降雨2017--2024：Task 1中文报告},pdfauthor={项目小组（Codex协助）}}
\ctexset{section={format=\Large\bfseries\color{teal}}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small 新加坡降雨 2017--2024 / Task 1}
\fancyhead[R]{\small 中文技术报告}
\fancyfoot[L]{\footnotesize 2026 年 10 月 8 日 / 真实数据全量分析}
\fancyfoot[R]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{.3pt}
\renewcommand{\arraystretch}{1.15}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\pgfplotsset{reportaxis/.style={font=\footnotesize,grid=major,grid style={gray!15},axis line style={gray!60},tick style={gray!60},label style={font=\footnotesize},title style={font=\small},legend style={font=\scriptsize},/pgf/number format/1000 sep={}}}
\begin{document}
'''

INTRO = r'''{\LARGE\bfseries\color{ink} 新加坡历史降雨的时间与空间规律}\par
{\large Task 1：2017--2024 年全量 Spark 探索分析}\par
\vspace{2mm}
\textbf{范围与状态。}本报告完整说明 Task 1。Task 2 模型训练与评估尚未开展；最终课程报告还需加入全体组员姓名和学号。此处不虚构组员信息。

\textbf{数据与方法。}使用 NEA / data.gov.sg 的八个年度数据集，源记录共 48,538,713 条，去重后 48,538,627 条，涉及 91 个站点 ID。全量清洗和聚合由 Spark 执行；跨年比较采用 21 个稳定站点，避免监测网络变化混入年度差异。

\textbf{主要结果。}全部有效五分钟区间中 97.08\% 无雨。合格日内样本的雨频高峰为新加坡时间 16:00--17:00。月度可用样本平均雨频以 11 月最高、7 月最低，但各月的可用年份及站点组成不完全相同。固定站点样本中，按观测时长归一化的雨量以 2019 年最低、2021 年最高。近距离站点的日雨量相关性总体高于远距离站点；八年不能据此推断长期气候变化。
'''

CLEANING = r'''\textbf{来源与字段。}每条记录包含 observation timestamp、update timestamp、station ID/name/device ID、经纬度、reading update timestamp、reading value、reading type、reading unit 及日期字段。本数据只有降雨相关观测，不包含气温、湿度和风等变量。API 预览只用于确认字段和年度 ID；所有天气结论来自完整文件。

\textbf{来源追踪。}每个完整 CSV 的清单保存官方数据集 ID、API 行数、下载时间、字节数和原始 CSV 字节的 SHA256。清洗时将实读 CSV 行数与该年度记录的 API 行数逐一核对。

\textbf{清洗步骤。}Spark 按 Asia/Singapore 解析时间；保留有限、非负、单位为 mm、类型为预期五分钟累计、站点 ID 和时间合法且 CSV 记录有效的行。只有与五分钟网格的距离不超过两秒时才对齐，原始 epoch 同时保留；其他离网格记录排除并审计。按 station ID 与对齐时刻去重，优先保留 reading update 最新的行，再使用原始观测时刻及确定的字段顺序打破并列。年度文件合并后再次检查跨文件重复。
'''
CLEANING_END = r'''\textbf{实际审计。}86 条重复记录全部来自 2017 年；八年没有负雨量或偏离网格超过两秒的记录。全量聚合的跨文件重复数为零。不存在年度覆盖率不低于 90\% 却所有读数均为零的站点年。缺失时段从不补成零。主分析会保留并标记大于 50 mm/5min 的读数，本次实际标记数为零；50 mm 是敏感性筛查值，并非官方质量阈值。合法读数仍可能包含传感器误差。
'''
METRICS = r'''设站点在给定时间段内的有效观测数为 $N$，雨量为 $r_i\geq0$，有雨区间数为 $W=\sum_i\mathbf{1}(r_i>0)$，预期日历槽数为 $E$。本文使用：
\begin{align*}
\text{覆盖率} &= N/E,\qquad E=288\times\text{日历天数},\\
\text{雨频（\%）} &=100W/N,\\
\text{湿区间雨强（mm/5min）} &=\sum_i r_i/W\quad(W>0),\\
\text{雨量 / 观测24h（mm）} &=288\sum_i r_i/N.
\end{align*}
预期日历槽数显式处理闰年；低覆盖率也可能反映站点只运行了部分年份。雨强的分母仅包含有雨区间。对于单个站点和同一时间段，归一化雨量等于 $288\times\text{雨频}/100\times\text{湿区间雨强}$。分别取站点平均后，该乘积恒等式一般不再成立。归一化雨量不是官方日累计或年度累计，也不能恢复遗漏风暴。

\textbf{年度固定样本。}要求同一站点八年均有观测、每年覆盖率至少 80\%，并且经纬度跨度各不超过 $0.005^\circ$，最终保留 21 站。坐标筛查只是一项启发式规则，不能证明站点完全未搬迁。各站点等权，并非按地理面积加权。

\textbf{月度与日内样本。}主月度图使用年度固定站点池中每个年月覆盖率至少 80\% 的站点，先站点等权、再可用年份等权；月度可用年份为 6--8 年，合格站点可变化。严格的八年同站点月度比较仅剩 5、6、7、9、10、12 月，因此作为敏感性结果保存，不能单独表示全年的季节模式。日内分析对合格的站点、年份、月份、小时组合等权。

\textbf{空间比较。}2024 年地图只给覆盖率至少 90\% 的站点着色。日相关性使用覆盖率至少 95\% 的站点日累计；每对站点至少需要 365 个共同日期。
'''
ENVIRONMENT = r'''\textbf{实际执行环境。}最初在 gpushare 的项目目录配置运行，但 SSH 和 JupyterLab 的间歇连接故障妨碍了可靠执行与取回结果。用户授权改用本机临时运行：Apple Silicon Mac，10 个逻辑 CPU、16 GiB RAM、macOS 26.5.2 ARM64，Python 3.12.14、Java Temurin 17.0.20.1+1、Spark 3.5.8。

\textbf{资源配置。}Spark 使用 \texttt{local[2]}，是一台机器上两个工作线程，而非多节点集群。driver heap 为 2 GiB，shuffle partitions 为 8，default parallelism 为 2；JVM ActiveProcessorCount 设为 2，BLAS/OpenMP 线程数为 1，进程以 nice 19 运行。macOS 不提供 Linux CPU affinity API，因此不声称有操作系统 CPU 绑定保证。未使用 GPU，也未进行全局 Java 或 Python 包安装。

\textbf{运行保护。}串行运行各年度清洗和最终聚合，每五秒采样项目子进程树 RSS 及剩余磁盘空间，RSS 超过 4 GiB 或空闲空间少于 5 GiB 时停止该子进程。实际采样峰值为 \textbf{__PEAK__ GiB}，最小空闲空间为 \textbf{__DISK__ GiB}。这是采样监控，不是操作系统硬配额。原始 gzip 仅在审计及 Parquet 输出成功后删除；来源和字节哈希保留。

\textbf{Spark 与展示层的分工。}全量数据的清洗、去重、时间分组和聚合使用 Spark DataFrame/SQL 及窗口。仅将小规模聚合表用于 pandas/Matplotlib 绘图、日聚合相关性和离线网页展示。中文版将同样的结果嵌入 TikZ/PGFPlots，先生成独立 LaTeX 源文件，再编译为 PDF；不需要外部图片或数据文件。

\textbf{核验。}源文件与 API 行数、去重计数、闰年覆盖率分母、站点年唯一性、指标恒等式、固定站点及月度成员、相关系数范围、九张图数量均已核验通过。独立 Spark 小例子验证了未来窗口不能包含当前或过去记录、缺一个未来槽应使窗口无效、超阈值使用严格大于而非大于等于，以及时间对齐容差。窗口标签范围为 $(t,t+1800\text{s}]$，只保留恰好六个有效槽；测量区间的物理起止语义仍需确认。
'''
LIMITATIONS = r'''\textbf{观测质量与缺测。}官方数据集提醒观测可能缺失，并且尚未经过气候数据质量控制。若暴雨期间更容易缺测，按时长归一化也无法消除偏差；覆盖率达到门槛同样不保证雨量无误。大于 50 mm/5min 的筛查实际没有触发，不意味着所有较小读数都可靠。

\textbf{代表性与统计依赖。}监测站不是等面积采样。不能将各站雨量相加后称为“全国降雨量”。固定站点池减少组成变化，却可能低估缺少长期记录的地区。坐标筛查不能替代站点运行、搬迁和维护记录。不同相关系数可来自不同日期，站点对相互依赖；未来窗口又大量重叠。因此这些统计不能简单当作独立样本做显著性检验。八年不足以形成稳健的长期气候变化判断。

\textbf{时间语义。}观测标签按两秒容差对齐到五分钟网格，只解决标签一致性，不证明标签是物理测量区间的起点或终点。将未来六槽用作半小时预测目标前，需要确认或明确记录此假设。还应区分观测时刻和数据实际可供预测使用的时刻。

\textbf{Task 2 的后续方向。}雨量呈现大量零值和明显短时持续性。模型可比较只用站点、日历和输入阈值的基线，与加入最近 5、10、15、30、60 分钟雨量和邻近站观测的模型。若添加近期观测，必须明确预测接口如何在预测时刻检索历史数据，不可使用未来数据。邻站信息同样应只使用当时已可获取的记录。

训练、验证和测试应按时间切分。阈值范围、特征和超参数应在训练/验证数据中选择；评估除 ROC-AUC 外还应关注 PR-AUC、Brier score、校准曲线，以及不同阈值与站点的性能。2024 年已用于 Task 1 探索，不能声称它在所有探索中完全未接触。这里提出的是后续设计方向，当前尚未训练或评价模型，不报告虚构的预测成绩。
'''
DECLARATION = r'''\textbf{AI 使用声明。}OpenAI Codex 协助项目规划、API 字段检查、运行环境配置、PySpark 清洗与聚合代码、执行及正确性核验、图表和交互页面代码、报告编写与翻译，以及结果解读。全部天气数值来自八个真实年度文件的 Spark 全量处理；没有将预览、合成或 AI 生成数值当作年度结果。中文报告图表使用同一份已核验聚合表，以内嵌 TikZ/PGFPlots 数据绘制。组员应审阅方法、源代码、结果和限制，并在最终提交中声明 AI 使用范围。

\textbf{资料来源。}
\begin{enumerate}
\item NEA，Historical Rainfall across Singapore，2017--2024 年度数据集：\href{https://data.gov.sg/collections/2279/view}{data.gov.sg 集合 2279}。各年度 ID、下载时间、API 行数及哈希保存在 \texttt{reports/downloads/}。
\item Meteorological Service Singapore，\href{https://www.weather.gov.sg/climate-climate-of-singapore}{Climate of Singapore}。仅用于季节与日内周期的气候背景，不代替项目观测，也不证明机制或因果关系。
\item geoBoundaries，\href{https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/}{Singapore ADM0 元数据}。轮廓源自 URA 2016 年资料，ODbL 1.0；下载的发布元数据和 GeoJSON 保存在 \texttt{reports/assets/}，不应解释为现今海岸线测绘。
\end{enumerate}

\textbf{可复现材料。}源码和运行说明位于 \texttt{scripts/} 与 \texttt{README.md}；审计位于 \texttt{reports/quality/}；环境与采样日志分别为 \texttt{environment.json} 和 \texttt{resource\_usage.jsonl}；聚合表、月度成员、敏感性比较、来源清单和核验结果一并保存。离线交互页面支持选择 2017--2024 年、地图指标和站点历史。无需为了打开报告重新执行全量 Spark。

\textbf{课程提交状态。}本报告是 Task 1 中文技术版本，并非整个项目最终提交。完成 Task 2 后须整合模型、评估和组员姓名/学号。按约定，整个项目完成后再推送至 \url{https://github.com/Panyuxiang0830/singapore_rainfall.git}，核验代码、报告和模型的持久备份后再清理服务器的项目目录。
'''

if __name__ == '__main__':
    main()
