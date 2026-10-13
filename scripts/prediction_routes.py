"""Resolve saved model and output column without starting Spark."""
import json


def resolve(root, model_root, threshold, algorithm=None, original=False):
    default = root / 'models/model_comparison_full'
    tuned = root / 'models/tuned_xgb_gt5'
    use_tuned = (threshold == 5 and algorithm in (None, 'xgb') and not original
                 and model_root.resolve() == default.resolve())
    if use_tuned:
        selection = json.loads((tuned / 'final_selection.json').read_text())
        name = selection['model']
        if name not in ('final_calibrated', 'final_uncalibrated') or selection['threshold_mm'] != 5:
            raise ValueError('Unexpected tuned model selection')
        return dict(path=tuned / name, probability_column='probability' if name == 'final_calibrated'
                    else 'base_probability', algorithm='xgb', policy='balanced',
                    calibration='Monotone Platt on 2023 H1' if name == 'final_calibrated' else 'none',
                    version='tuned_gt5_depth6_20261013')
    manifest = json.loads((model_root / 'manifest.json').read_text())
    q = str(threshold)
    algo = algorithm or manifest['selected_algorithm_by_threshold'][q]
    policy = manifest['selected_policy_by_algorithm_and_threshold'][algo][q]
    return dict(path=model_root / algo / policy / f'gt_{threshold}mm',
                probability_column=f'prob_{threshold}', algorithm=algo, policy=policy,
                calibration=manifest['calibration'], version='original_full_comparison')
