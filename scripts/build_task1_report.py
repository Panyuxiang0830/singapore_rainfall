"""Generate the synchronized standalone English LaTeX report.

The Chinese source contains reviewed text and real embedded chart values.
Compile the resulting source with Tectonic/XeLaTeX to export Task1_Report.pdf.
No deleted CSV tables, chart images or audit records are required.
"""
from pathlib import Path
from report_translations import translate

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'reports/Task1_Report_CN.tex'
    target = ROOT / 'reports/Task1_Report.tex'
    target.write_text(translate(source.read_text(encoding='utf-8')), encoding='utf-8')
    print('English LaTeX source:', target)


if __name__ == '__main__':
    main()
