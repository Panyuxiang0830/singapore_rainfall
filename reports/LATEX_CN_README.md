# 中英文完整项目报告（LaTeX → PDF）

`Task1_Report_CN.tex` 与 `Task1_Report.tex` 保留旧文件名，但现已覆盖 Task 1 和 Task 2。
对应 PDF 为 `Task1_Report_CN.pdf` 与 `Task1_Report.pdf`，各 21 页。
十组图表直接内嵌 TikZ/PGFPlots 数值，不依赖单独图片或统计 CSV。
成员署名为潘禹翔、侯佳成、吴嘉蕊，按小组要求不列学号。

## 编辑与编译

继续在已有 LaTeX 编辑器中编辑中文源文件，同时更新
`scripts/report_translations.py` 或 `scripts/report_task2_content.py` 中的审阅英文，
再运行 `python scripts/build_task1_report.py` 生成英文。未同步翻译的中文改动会使生成器报错。
原 Task 1 中文生成器不能覆盖当前完整报告。

使用 Codex 内置编译器检查，两种语言均应成功。也可使用已有 Tectonic / XeLaTeX 导出：

```bash
tectonic --untrusted --outdir reports reports/Task1_Report_CN.tex
tectonic --untrusted --outdir reports reports/Task1_Report.tex
```

本次先编辑 LaTeX、通过内置编译诊断，再使用已有 Tectonic 导出 PDF；
不安装新的 TeX 发行版。用项目工作目录的 PyMuPDF 渲染进行排版核验。
报告数值来自真实 Spark 运行；重新训练请遵循 README。
