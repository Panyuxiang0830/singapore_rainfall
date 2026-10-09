"""Reviewed editorial changes for the fixed 2017--2024 Task 1 report.

Apply only to freshly generated Chinese source; update English translations
when editing report text. Numbers refer to the verified published run.
"""

def revise(source):
    s = source
    def line(prefix,new):
        nonlocal s
        found=[x for x in s.splitlines() if x.startswith(prefix)]
        if len(found)!=1: raise ValueError((prefix,len(found)))
        s=s.replace(found[0],new)
    
    line(r'\textbf{年度固定样本。}',r'\textbf{年度固定样本。}八年共出现 91 个站点 ID，其中 21 个同时满足：八年均有观测、每年覆盖率至少 80\%、经纬度跨度各不超过 $0.005^\circ$。其余 70 个只是不进入固定站点的跨年比较，清洗时没有删除这些站点的有效观测。21 个站点等权平均；不同分析使用的站点范围见下一页。')
    line(r'\textbf{月度与日内样本。}',r'\textbf{月度与日内样本。}图 3、4、6 从这 21 个站点中进一步筛选数据完整的年月或小时组合。月度先对站点等权平均，再对可用年份等权平均；每个合格年月有 17--21 个站点，各月有 6--8 个可用年份。严格要求同站点同月份在八年均达标时，只剩 5、6、7、9、10、12 月，作为额外对照。')
    line(r'\textbf{空间比较。}',r'\textbf{空间比较。}图 1、2 使用 2024 年的站点，当年共 70 个，其中 56 个覆盖率至少 90\%，用于着色和排名，另 14 个不参加排名。图 8 使用固定 21 站的日累计，要求日覆盖率至少 95\%，每对至少有 365 个共同有效日期。80\%、90\%、95\% 是本项目选定的数据完整程度门槛。')
    start=s.index(r'\clearpage\section*{3. 计算环境与结果核验}')
    end=s.index(r'\clearpage\section*{图 1：',start)
    s=s[:start]+r'''\clearpage\section*{3. 计算环境与各项分析的样本范围}
    \textbf{实际计算环境。}Task 1 主体分析此前在 Apple Silicon Mac 上完成：10 个逻辑 CPU、16 GiB RAM，macOS 26.5.2 ARM64，Python 3.12.14、Java Temurin 17.0.20.1+1、Spark 3.5.8。2026 年 10 月 9 日，Task 2 标签与图 6 中的半小时超阈值统计在 gpushare 复核：Python 3.11.10、Azul Zulu Java 17.0.20.1、Spark 3.5.8；容器 CPU 配额为 42 核、内存上限约 667.68 GiB，项目实际仅使用两个 CPU、2 GiB driver heap，监测进程树 RSS 上限为 4 GiB。
    
    \textbf{Spark 配置与分工。}使用 \texttt{local[2]}，即单机两个 Spark 工作线程；driver heap 为 2 GiB，shuffle partitions 为 8，default parallelism 为 2。按年度串行处理，不使用 GPU。清洗、去重、时间分组及全量聚合由 Spark DataFrame/SQL 和窗口完成；汇总后的少量数据用于绘图与日雨量相关性计算。
    
    \textbf{各项分析用了哪些站点？}筛选服务于特定比较，不是将未进入固定样本的站点全部删除。
    \begin{center}\begin{tabular}{lll}
    \toprule
    分析 & 使用范围 & 筛选说明\\\midrule
    图 1、2：2024 地图与站点比较 & 当年 70 站中的 56 站 & 年覆盖率至少 90\%；另 14 站不排名\\
    图 3、4：月度变化 & 固定 21 站的合格子集 & 每个合格年月保留 17--21 站\\
    图 5：年度变化 & 同一组 21 站 & 其余 70 站不参与固定站点跨年比较\\
    图 6：日内变化 & 固定 21 站的合格组合 & 年月小时组合覆盖率至少 80\%\\
    图 7：覆盖率与网络变化 & 八年全部 91 个站点 ID & 每年实际出现的站点数不同\\
    图 8：距离与日雨量相关性 & 固定 21 站，210 对 & 日覆盖率至少 95\%，共同有效日至少 365\\
    图 9 左：有雨时的雨量分布 & 八年全部 91 站 & 仅统计有雨五分钟区间\\
    图 9 中、右：半小时超阈值与持续性 & 2024 年全部 70 站的有效观测 & 六个未来槽完整，或两个五分钟槽连续\\
    \bottomrule\end{tabular}\end{center}
    \textbf{清洗核对。}源记录数与 API 计数一致；删除 86 条重复记录，缺失时段不补成零。全部观测的无雨比例也使用全部 91 站，而不是仅使用 21 站。
    
    ''' + s[end:]
    # Keep only the distance scatter and existing real-data coordinates in Figure 8.
    start=s.index(r'\clearpage\section*{图 8：')
    end=s.index(r'\clearpage\section*{图 9：',start)
    b=s[start:end]
    a=b.index(r'\begin{axis}')
    z=b.index(r'\end{axis}',a)+len(r'\end{axis}')
    axis=b[a:z].replace('at={(12.8cm,0)}','at={(0cm,0)}').replace('width=10.7cm,height=8cm','width=18cm,height=7.5cm')
    s=s[:start]+r'''\clearpage\section*{图 8：站点距离与日雨量的相关性}
    \begin{center}\begin{tikzpicture}
    '''+axis+r'''
    \end{tikzpicture}\end{center}
    \noindent 每个点代表一对站点：横轴是两站距离，纵轴是两站在共同有效日期上的日雨量 Pearson 相关系数。深线是每 5 km 距离分组的平均相关性。21 站形成 210 对；距离不超过 5 km 的 17 对平均相关性为 0.809，超过 30 km 的 12 对为 0.336。\textbf{总体上，距离较近的站点，日雨量更容易一起增减。}这并不表示每一对近站的相关性都高于每一对远站。
    
    \textbf{站点并非独立。}附近站点可能同时经历同一场降雨；不同站点对还可能共享同一个站点。因此，91 个站点不能视为 91 份完全独立的天气样本，210 对相关系数也不是 210 个独立样本。共同有效日的筛选与前文相同。
    
    '''+s[end:]
    # Three plain reading instructions, retaining the existing factual plotted data.
    s=s.replace('图 9：降雨分布、持续性与未来阈值','图 9：雨下多大、半小时超阈值与短时持续性')
    s=s.replace('title={有雨区间的雨量分布},ylabel={湿区间占比（\\%）}', 'title={左：下雨时，每五分钟下多少？},xlabel={五分钟雨量（mm）},ylabel={有雨记录中的占比（\\%）}')
    s=s.replace('title={未来六个观测槽超阈值}', 'title={中：后续30分钟累计超过多少？}')
    s=s.replace('title={短时降雨持续性}', 'title={右：前一段下雨，下一段呢？}')
    line(r'\noindent 八年全部有效观测中有 97.08',r'''\noindent \textbf{左图：下雨时有多大。}分母是八年所有站点的\textbf{有雨}五分钟记录。柱子表示不同雨量范围所占比例；例如 41.72\% 的有雨记录在 $(0,0.2]$ mm，37.16\% 在 $(0.2,1]$ mm。二者合计约 78.88\%，说明多数有雨区间的雨量不超过 1 mm。无雨记录未画入左图；在全部有效记录中，无雨比例为 97.08\%。
    
    \textbf{中图：半小时累计超过阈值有多常见。}统计 2024 年的 6,374,476 个完整后续六槽窗口，以 $(t,t+30\text{min}]$ 内六个五分钟读数的和表示累计量。累计量保留九位小数，避免浮点求和在阈值边界产生误判。横轴是雨量阈值，纵轴是严格超过该阈值的窗口比例。超过 0 mm 的比例为 6.36\%，超过 5 mm 为 0.83\%；阈值越大，事件越少。横轴按五个类别等距排列，不表示阈值差距相等。
    
    \textbf{右图：降雨是否会接着下。}统计 2024 年同站点连续两个五分钟区间。前一段无雨时，下一段有雨的比例为 1.04\%；前一段有雨时，下一段有雨的比例为 67.03\%。这说明近期降雨可为 Task 2 提供有用信息。中图和右图是历史统计，\textbf{不是模型的预测成绩}。''')
    start=s.index(r'\clearpage\section*{4. 限制、额外观察与 Task 2 衔接}')
    end=s.index(r'\clearpage\section*{5. AI 使用声明、来源与复现}',start)
    s=s[:start]+s[end:]
    s=s.replace('5. AI 使用声明、来源与复现','4. AI 使用声明、来源与复现')
    line(r'\textbf{可复现材料。}',r'\textbf{可复现材料。}源码与运行说明位于 \texttt{scripts/} 和 \texttt{README.md}，中英文报告均保留 LaTeX 源文件。按要求不保存图表图片、统计 CSV、交互页面或运行核验记录；重跑分析脚本可生成这些结果。现有 LaTeX 图表已内嵌数值，可直接编译，无需重新下载八年数据。')
    line(r'\item NEA，',r'\item NEA，Historical Rainfall across Singapore，2017--2024 年度数据集：\href{https://data.gov.sg/collections/2279/view}{data.gov.sg 集合 2279}。运行下载脚本时生成年度来源和校验信息。')
    
    line(r'\noindent 使用同一年度',r'\noindent 使用 2024 年覆盖率至少 90\% 的 56 个站点。左图每个点是一座站：横轴是下雨频率，纵轴是下雨时的平均五分钟雨量；右图列出雨频最高与最低的各 10 站。S33 的雨频最高，而 S211（Kranji Road）的湿区间雨强最高。两图回答不同问题：\textbf{哪里经常下雨，哪里下雨时更大。}')
    line(r'\noindent 季节模式并非每年',r'\noindent 每一行是一个年份，每一列是一个月份。左图颜色越深，表示该年月的归一化雨量越大；横向看一年内的月份变化，纵向看同一个月在不同年份的差异。右图展示相同样本的平均覆盖率，帮助检查左图是否受到观测不足影响。灰格表示没有达标样本，不表示没有雨。96 个年月中有 89 个合格单元，每个合格单元来自固定 21 站中的 17--21 站。')
    line(r'\noindent 分母采用完整日历年',r'\noindent 这张图解释为什么年度比较需要固定站点。左图每个点是一个站点在某年的数据覆盖率，虚线是 80\% 门槛；右图每年两根柱子分别是有有效记录的站点数，以及覆盖率至少 90\% 的站点数。2019 年有 54 站，2020 年有 80 站，说明监测网络并不恒定。图 7 使用全部站点记录，未限制在固定 21 站；它描述数据完整性与网络变化，而不是降雨趋势。')
    return add_task2_feature_implications(remove_requested_figures(s))


def remove_requested_figures(source):
    """Remove original Figures 2, 4 and 7 and renumber remaining groups."""
    import re
    source = re.sub(
        r'\\clearpage\\section\*\{图 (2|4|7)：[^}]*\}.*?(?=\\clearpage\\section\*|\\end\{document\})',
        '', source, flags=re.S,
    )
    # Update grouped references and scope rows before renumbering.
    source = source.replace('图 3、4、6', '图 3、6')
    source = source.replace('图 1、2 使用', '图 1 使用')
    source = source.replace('用于着色和排名，另 14 个不参加排名', '用于地图着色，另 14 个以灰色标记')
    source = source.replace('图 1、2：2024 地图与站点比较', '图 1：2024 站点地图')
    source = source.replace('年覆盖率至少 90\\%；另 14 站不排名', '年覆盖率至少 90\\%；另 14 站灰色标记')
    source = source.replace('图 3、4：月度变化', '图 3：月度变化')
    source = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('图 7：覆盖率与网络变化')) + '\n'
    numbers = {1: 1, 3: 2, 5: 3, 6: 4, 8: 5, 9: 6}
    source = re.sub(r'图 (\d+)((?:、\d+)*)', lambda m: '图 ' + '、'.join(str(numbers[int(n)]) for n in [m[1], *re.findall(r'\d+', m[2])]), source)
    return "\n".join(line.lstrip() for line in source.splitlines()) + "\n"


def add_task2_feature_implications(source):
    """Add reviewed feature hypotheses to the six retained figures."""
    additions = [('\\noindent 2024 年有',
      '\\textbf{Task 2 特征启示（站点）。}候选特征包括站点 ID 和经纬度，以表示站点间差异。Task 2 不直接沿用跨年比较的 21 '
      '站限制，而按历史特征及未来标签的完整程度筛选训练窗口。'),
     ('\\vspace{5mm}\\noindent 从年度',
      '\\textbf{Task 2 '
      '特征启示（月份）。}保留月份或年内日期的周期编码，并通过有无月份特征的对照检验其贡献。新加坡受季风影响，但没有界限分明的旱季和雨季；较少降雨不等于不下雨，雨频也不等于累计雨量。'),
     ('\\noindent 相同 21 个',
      '\\textbf{Task 2 特征启示（年份）。}暂不将年份编号作为主模型特征，以免记住特定年份而难以预测未来年份。NEA 对 2019 '
      '年的回顾记载了偏干及强正位相印度洋偶极子背景；这是外部气候背景，不是本数据的因果证明。按时间划分训练、验证与测试，并报告跨年表现。'),
     ('\\noindent 雨频最高的小时',
      '\\textbf{Task 2 特征启示（小时与月份）。}使用新加坡当地小时的周期编码，并考虑月份与小时的交互项：下午雨频高峰的强弱和时刻会随月份变化。Logistic 和线性 SVM '
      '需显式构造交互特征；树模型可学习这类组合关系，其实际贡献仍由对照实验检验。'),
     ('\\textbf{站点并非独立。}',
      '\\textbf{Task 2 '
      '特征启示（邻站）。}先加入预测时已可获得的邻站历史雨量：距离加权平均、最大值、有雨比例，以及有效邻站数。图神经网络暂缓。日雨量相关性仅提供候选特征的依据，不能直接证明邻站信息能改善半小时预测；比较有无邻站特征的模型。'),
     ('\\textbf{右图：',
      '\\textbf{Task 2 特征启示（持续性与稀有事件）。}候选特征包括最近五分钟读数、过去 15/30/60 '
      '分钟累计、近期雨势变化和连续有雨时长。只使用预测时已可获得的记录，缺测不补为无雨。较高阈值对应稀有事件，应逐阈值报告正例数、PR-AUC、Brier score 和校准曲线；这些是待开展的模型实验。')]
    for prefix, text in additions:
        matches = [line for line in source.splitlines() if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError((prefix, len(matches)))
        if text not in source:
            source = source.replace(matches[0], matches[0] + "\n\n" + text)
    extra_source = '\\item NEA，\\href{https://www.nea.gov.sg/media/news/news/index/2019-is-singapore-s-joint-warmest-year-on-record}{2019 年气候回顾}。用于偏干年份与正位相印度洋偶极子的外部背景，未用作模型预测特征。'
    anchor = next(line for line in source.splitlines() if line.startswith(r"\item geoBoundaries，"))
    if extra_source not in source:
        source = source.replace(anchor, extra_source + "\n" + anchor)
    return source
