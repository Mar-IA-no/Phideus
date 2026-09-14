"""Copy authenticated, fully completed campaign summaries into readable tables.

No new statistics, fits, forwards, truth parsing or scientific-artifact writes.
Independent verification of raw evidence remains the final auditor's task.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'.agent-work/phideus-generative-report-20260909'
FINAL = ROOT/'data/atencion_armonica/generative_evidence_reader_v1/fresh/test_completion.json'
TESTS = ('iid', 'ood_beta', 'ood_polyphony', 'deformed_family')
ARMS = ('local', 'generative', 'decoupled')
METRICS = ('ari', 'exact_partition', 'pair_disagreement', 'k_inferred', 'k_error',
           'k_absolute_error', 'sub3_member_fraction', 'vi', 'vi_normalized',
           'split_entropy', 'merge_entropy', 'split_normalized', 'merge_normalized')
COLUMNS = ('split', 'reader', 'scope', 'count', 'metric', 'mean', 'low', 'high',
           'nominal_percent', 'source_path', 'source_sha256')


class Reader:
    def __init__(self, root):
        self.root, self.consumed = Path(root).resolve(), {}

    def read(self, ref, base=Path('.')):
        if (set(ref) not in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'})
                or not isinstance(ref['path'], str) or not isinstance(ref['sha256'], str)):
            raise ValueError('invalid report source reference')
        relative = base/Path(ref['path'])
        path = self.root/relative
        resolved = path.resolve()
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not resolved.is_relative_to(self.root)):
            raise ValueError('report source escapes the repository')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ref['sha256'] or ('bytes' in ref and
                (type(ref['bytes']) is not int or len(raw) != ref['bytes'])):
            raise ValueError('report source bytes differ')
        identity = {'path': relative.as_posix(), 'sha256': digest}
        if self.consumed.setdefault(identity['path'], digest) != digest:
            raise ValueError('one report path has conflicting hashes')
        return json.loads(raw), identity


def number(value, *, nullable=False):
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('nonfinite or nonnumeric report value')
    return value


def table_rows(summary, source):
    primary, systems = summary['learned']['primary'], summary['systems']
    split, count = primary['split'], primary['output_count']
    if (split not in TESTS or type(count) is not int or not 0 <= count <= 512
            or primary['count'] != 512 or systems['count'] != 512
            or systems['output_count'] != count or primary['coverage'] != count/512
            or systems['coverage'] != count/512
            or primary['metric_order'] != list(METRICS)
            or systems['metric_order'] != list(METRICS)
            or set(primary['arms']) != set(ARMS)
            or set(primary['contrasts']) != {'generative-minus-local', 'generative-minus-decoupled'}):
        raise ValueError('summary roster, metrics or support differs')
    rows = []

    def append(reader, scope, n, metric, mean, interval=None, nominal=None):
        number(mean, nullable=(n == 0))
        if n == 0 and mean is not None:
            raise ValueError('empty support cannot have a mean')
        if interval is not None:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError('invalid preserved interval')
            low, high = map(number, interval)
            if low > high:
                raise ValueError('reversed preserved interval')
        else:
            low = high = None
        rows.append(dict(zip(COLUMNS, (split, reader, scope, n, metric, mean,
            low, high, nominal, source['path'], source['sha256']))))

    for family, scope in (('arms', 'common_output_support'),
                           ('contrasts', 'paired_scene_contrast')):
        for reader, metrics in primary[family].items():
            if set(metrics) != set(METRICS):
                raise ValueError('incomplete metric mapping')
            for metric in METRICS:
                value = metrics[metric]
                expected = 97.5 if (family == 'contrasts' and split == 'deformed_family'
                                     and metric == 'ari') else 95
                if (set(value) != {'mean', 'interval', 'nominal_percent'}
                        or value['nominal_percent'] != expected
                        or (value['interval'] is None) != (count == 0)):
                    raise ValueError('preserved interval authority differs')
                append(reader, scope, count, metric, value['mean'], value['interval'], expected)

    def array(reader, scope, n, values):
        if n == 0:
            if values is not None:
                raise ValueError('empty reference support has values')
            values = [None]*len(METRICS)
        if not isinstance(values, list) or len(values) != len(METRICS):
            raise ValueError('reference metric extent differs')
        for metric, value in zip(METRICS, values):
            append(reader, scope, n, metric, value)

    for name in ('base', 'extended'):
        array(name, 'system_common_support', count, systems[name]['common_support_mean'])
    if set(systems['historical']) != {'2026090721', '2026090722', '2026090723'}:
        raise ValueError('historical checkpoint roster differs')
    for cp, values in sorted(systems['historical'].items()):
        array('historical-'+cp, 'system_common_support', count, values['common_support_mean'])
        array('historical-'+cp, 'system_all_scenes', 512, values['all_512_mean'])
    return rows


def collect():
    # Check before importing the campaign or opening a partial summary.
    if not FINAL.is_file():
        raise RuntimeError('report requires all four tests and their completed replays')
    sys.path.insert(0, str(ROOT))
    from experiments.atencion_armonica import resume_generative_json_tests as supervisor
    completion = supervisor.verified_tests()
    reader = Reader(ROOT)
    final, _ = reader.read(completion)
    if set(final['tests']) != set(TESTS):
        raise ValueError('completion does not contain exactly four tests')
    summaries, rows = {}, []
    for split in TESTS:
        entry = final['tests'][split]
        seal, _ = reader.read(entry['prediction_seal'])
        evaluation, index_ref = reader.read(entry['evaluation'])
        binding = {'split': split, 'test_freeze': seal['binding']['test_freeze'],
                   'prediction_seal': entry['prediction_seal']}
        if (seal['split'] != split or seal['prediction_count'] != 45
                or seal['status'] != 'FRESH_PREDICTIONS_SEALED_NO_TRUTH_ACCESS'
                or seal['truth_access'] is not False
                or evaluation['binding'] != binding or evaluation['prediction_count'] != 45
                or evaluation['status'] != 'EVALUATED_NOT_PROMOTED'
                or evaluation['scene_ids'] != list(range(512))):
            raise ValueError('evaluation and seal identity differ')
        summary, ref = reader.read(evaluation['summary'], Path(index_ref['path']).parent)
        if (summary['binding'] != binding or summary['status'] != 'EVALUATED_NOT_PROMOTED'
                or summary['learned']['primary']['split'] != split):
            raise ValueError('summary has another binding or split')
        rows.extend(table_rows(summary, ref))
        summaries[split] = {'source': ref, 'learned': summary['learned'],
                            'systems': summary['systems'], 'sham': summary['sham']}
    return {'authority': 'COPIED_AUTHENTICATED_SUMMARIES_NOT_INDEPENDENT_RECOMPUTATION',
            'completion': completion, 'accumulated_seconds': final['accumulated_seconds'],
            'source_sha256': dict(sorted(reader.consumed.items())), 'tests': summaries}, rows


def main():
    result, rows = collect()
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n'
    destination = WORK/'result'
    destination.mkdir(exist_ok=False)
    with (destination/'summary.json').open('x') as file:
        file.write(payload)
    with (destination/'metrics.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'output': str(destination), 'rows': len(rows), 'tests': len(result['tests'])}))


if __name__ == '__main__':
    main()
