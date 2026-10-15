"""Check the committed supplementary distance artifacts agree."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results/distance_full'
MODELS = ROOT/'models/full_variants/distance'
EXPECTED = {
    0: ('natural', 0.628384977646, 0.890448731264, 0.035469093573),
    5: ('balanced', 0.239913000035, 0.892772043442, 0.007078775345),
    25: ('balanced', 0.023674462025, 0.918550037525, 0.000543867145),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    state = read(RESULTS/'feature_preprocessing.json')
    prep = read(RESULTS/'preparation.json')
    selection = read(RESULTS/'selection.json')
    manifest = read(MODELS/'manifest.json')
    assert state['variant'] == 'distance' and len(state['feature_names']) == 22
    assert set(state['feature_names']) == set(state['mean_by_feature'])
    assert state['fit_rows'] == prep['train_rows'] == 34542285
    assert selection == manifest
    assert (ROOT/'models/logistic_baseline_full/preprocessing/metadata/_SUCCESS').exists() or \
           (ROOT/'models/logistic_baseline_full/preprocessing/metadata/part-00000').exists()
    for q, (policy, pr, roc, brier) in EXPECTED.items():
        key = str(q)
        assert manifest['selected_algorithm_by_threshold'][key] == 'xgb'
        assert manifest['selected_policy_by_algorithm_and_threshold']['xgb'][key] == policy
        test = read(RESULTS/'test'/f'xgb_gt_{q}mm.json')
        curve = read(RESULTS/'curves'/f'final_gt_{q}mm.json')
        assert test['n'] == curve['n'] == 6374476
        assert test['selected_policy'] == curve['policy'] == policy
        for metric, expected in [('pr_auc', pr), ('roc_auc', roc), ('brier', brier)]:
            assert math.isclose(test[metric], expected, abs_tol=1e-8), (q, metric)
        assert math.isclose(test['pr_auc'], curve['pr_auc'], abs_tol=1e-8)
        assert math.isclose(test['roc_auc'], curve['roc_auc'], abs_tol=1e-8)
        metadata = manifest['model_metadata']['xgb'][policy][key]
        assert metadata['vector_dimension'] == 117
        assert metadata['training_rows'] == 34542285
        assert (MODELS/'xgb'/policy/f'gt_{q}mm'/'metadata'/'part-00000').is_file()
    for name in ('Task1_Report_CN.pdf', 'Task1_Report.pdf', 'Task1_Report_CN.tex', 'Task1_Report.tex'):
        assert (ROOT/'reports'/name).is_file(), name
    for name in ('Task1_Report_CN.tex', 'Task1_Report.tex'):
        report = (ROOT/'reports'/name).read_text(encoding='utf-8')
        assert 'DISTANCE_SUPPLEMENT_START' in report, name
    print('DISTANCE_SUPPLEMENT_OK: 117 dimensions; 3 saved models; 2024 results and reports agree')


if __name__ == '__main__':
    main()
