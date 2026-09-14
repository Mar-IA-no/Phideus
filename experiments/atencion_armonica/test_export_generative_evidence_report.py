"""Small formatting/negative tests; no campaign execution or GPU imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('report_export', HERE/'export_generative_evidence_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
SOURCE = {'path': 'fixture.json', 'sha256': 'a'*64}


def summary(count=1, split='iid'):
    def stats(contrast=False):
        return {m: {'mean': 0.1 if count else None,
                    'interval': [0.0, 0.2] if count else None,
                    'nominal_percent': 97.5 if contrast and split == 'deformed_family' and m == 'ari' else 95}
                for m in report.METRICS}
    array = [0.1]*13 if count else None
    return {'learned': {'primary': {'split': split, 'count': 512,
        'output_count': count, 'coverage': count/512, 'metric_order': list(report.METRICS),
        'arms': {a: stats() for a in report.ARMS},
        'contrasts': {a: stats(True) for a in ('generative-minus-local', 'generative-minus-decoupled')}}},
        'systems': {'count': 512, 'output_count': count, 'coverage': count/512,
        'metric_order': list(report.METRICS),
        'base': {'common_support_mean': array}, 'extended': {'common_support_mean': array},
        'historical': {str(cp): {'common_support_mean': array, 'all_512_mean': [0.1]*13}
                       for cp in (2026090721, 2026090722, 2026090723)}}}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report.WORK.mkdir(parents=True, exist_ok=True)

    def test_all_metrics_scopes_and_primary_intervals(self):
        for split in report.TESTS:
            with self.subTest(split=split):
                rows = report.table_rows(summary(split=split), SOURCE)
                self.assertEqual(len(rows), 169)
                self.assertEqual(len({(r['reader'], r['scope'], r['metric']) for r in rows}), 169)
                for row in rows:
                    self.assertEqual(row['source_sha256'], SOURCE['sha256'])
                    if row['scope'].startswith('system'):
                        self.assertIsNone(row['nominal_percent'])
                        self.assertIsNone(row['low'])
                    if row['scope'] == 'paired_scene_contrast' and row['metric'] == 'ari':
                        self.assertEqual(row['nominal_percent'], 97.5 if split == 'deformed_family' else 95)

    def test_no_output_is_not_zero(self):
        for row in report.table_rows(summary(0), SOURCE):
            if row['scope'] != 'system_all_scenes':
                self.assertIsNone(row['mean'])
                self.assertIsNone(row['low'])
                self.assertEqual(row['count'], 0)

    def test_malformed_summaries_rejected(self):
        for case in ('nan', 'bool', 'missing_metric', 'support', 'order', 'interval', 'nominal', 'null', 'historical'):
            with self.subTest(case=case):
                s = copy.deepcopy(summary())
                p = s['learned']['primary']
                if case in ('nan', 'bool', 'null'):
                    p['arms']['local']['ari']['mean'] = {'nan': float('nan'), 'bool': True, 'null': None}[case]
                elif case == 'missing_metric':
                    del p['arms']['local']['ari']
                elif case == 'support':
                    s['systems']['output_count'] = 2
                elif case == 'order':
                    p['metric_order'].reverse()
                elif case == 'interval':
                    p['arms']['local']['ari']['interval'] = [0.2, 0.0]
                elif case == 'nominal':
                    p['arms']['local']['ari']['nominal_percent'] = 90
                else:
                    del s['systems']['historical']['2026090721']
                with self.assertRaises(ValueError):
                    report.table_rows(s, SOURCE)

    def test_reader_hash_bytes_and_path(self):
        with tempfile.TemporaryDirectory(dir=report.WORK) as directory:
            root = Path(directory)
            raw = b'{"value":1}'
            (root/'x.json').write_bytes(raw)
            ref = {'path': 'x.json', 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            reader = report.Reader(root)
            self.assertEqual(reader.read(ref)[0], {'value': 1})
            for changed in ({**ref, 'sha256': '0'*64}, {**ref, 'bytes': 999},
                            {**ref, 'path': '../x.json'}, {**ref, 'path': str(root/'x.json')}):
                with self.assertRaises(ValueError):
                    reader.read(changed)
            (root/'link.json').symlink_to(root/'x.json')
            with self.assertRaises(ValueError):
                reader.read({**ref, 'path': 'link.json'})

    def test_incomplete_campaign_writes_nothing(self):
        with tempfile.TemporaryDirectory(dir=report.WORK) as directory:
            root = Path(directory)
            with patch.object(report, 'FINAL', root/'missing.json'), patch.object(report, 'WORK', root):
                with self.assertRaisesRegex(RuntimeError, 'all four tests'):
                    report.main()
                self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless((report.ROOT/'data/atencion_armonica/generative_evidence_reader_v1/fresh/iid/evaluation/summary.json').is_file(), 'requires preserved local IID evaluation')
    def test_completed_iid_format_only(self):
        # IID was already evaluated/replayed before this test. No raw observations.
        path = report.ROOT/'data/atencion_armonica/generative_evidence_reader_v1/fresh/iid/evaluation/summary.json'
        ref = {'path': path.relative_to(report.ROOT).as_posix(),
               'sha256': '3a6345420b8485f486c66f61336a7c68c7b0904fcf09822f52f8e3e1128be053'}
        data, source = report.Reader(report.ROOT).read(ref)
        rows = report.table_rows(data, source)
        self.assertEqual(len(rows), 169)
        for row in rows:
            if row['scope'] == 'common_output_support':
                self.assertEqual(row['mean'], data['learned']['primary']['arms'][row['reader']][row['metric']]['mean'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
