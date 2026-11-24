"""CPU-only, read-only handoff audit; no model loading, conversion or label invention.

Example: --calib /path/calib.npz --eval /path/eval.npz --report /path/audit.json
Use --set model.precision_mode=nf4_fp32_from_fp16 for the chosen FP32 mode.
"""
import json
from zipfile import BadZipFile
from pathlib import Path
from _common import parse


def extra(ap):
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--eval', type=Path)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--calib-prompts', type=Path, help='Exact ordered collector prompt records')
    ap.add_argument('--eval-prompts', type=Path, help='Exact ordered collector prompt records')


def main():
    args, cfg = parse(__doc__, extra)
    from mcal.audit import audit_file, audit_pair
    inputs = [p.resolve() for p in (args.calib, args.eval) if p is not None]
    inputs += [p.resolve() for p in (args.calib_prompts, args.eval_prompts) if p is not None]
    if bool(args.calib_prompts) != bool(args.eval_prompts) and args.eval:
        raise SystemExit('Supply both prompt files to check calibration/evaluation text overlap')
    if args.eval_prompts and not args.eval:
        raise SystemExit('--eval-prompts requires --eval')
    if args.report.resolve() in inputs or args.report.exists():
        raise SystemExit('Choose a new report path; source traces and previous reports are preserved')
    report = dict(offline_checks_passed=False, gpu_validation_performed=False,
                  label_truth_verified=False, files=[], errors=[])
    try:
        calib, details = audit_file(args.calib, cfg, args.synthetic)
        report['files'].append(details)
        from mcal.provenance import read_prompt_records, check_prompt_binding, require_disjoint_text
        if args.calib_prompts:
            calib_records = read_prompt_records(args.calib_prompts)
            report['calib_prompt_check'] = check_prompt_binding(calib, calib_records)
        if args.eval:
            ev, details = audit_file(args.eval, cfg, args.synthetic)
            report['files'].append(details)
            report['split_check'] = audit_pair(calib, ev)
            if args.eval_prompts:
                eval_records = read_prompt_records(args.eval_prompts)
                report['eval_prompt_check'] = check_prompt_binding(ev, eval_records)
                report['text_overlap_check'] = require_disjoint_text(calib_records, eval_records)
        report['offline_checks_passed'] = True
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        report['errors'].append(str(exc))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['offline_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
