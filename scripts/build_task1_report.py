"""Generate the synchronized standalone English LaTeX report.

The Chinese source contains reviewed text and real embedded chart values.
Compile the resulting source with Tectonic/XeLaTeX to export Task1_Report.pdf.
No deleted CSV tables, chart images or audit records are required.
"""
from pathlib import Path
from report_translations import translate
from add_distance_supplement import strip_distance_supplement, upsert

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'reports/Task1_Report_CN.tex'
    target = ROOT / 'reports/Task1_Report.tex'
    # The supplement has reviewed prose in both languages.  Remove its Chinese
    # block before the strict line-by-line translator runs, then restore both
    # language blocks so a later sync cannot erase it.
    target.write_text(translate(strip_distance_supplement(source.read_text(encoding='utf-8'))), encoding='utf-8')
    upsert(source, 'cn')
    upsert(target, 'en')
    print('English LaTeX source:', target)


if __name__ == '__main__':
    main()
