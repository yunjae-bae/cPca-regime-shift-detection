import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


def definitions():
    notebook = json.loads(Path(__file__).with_name('PCA_economy.ipynb').read_text())
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(node for node in ast.parse(''.join(cell['source'])).body
                         if isinstance(node, ast.FunctionDef))
    scope = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook definitions>', 'exec'), scope)
    return scope


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.functions = definitions()
        self.dates = pd.date_range('2020-01-01', periods=5, freq='MS')

    def test_boundary_is_first_normal_month_after_last_breach(self):
        fn = self.functions['trailing_threshold_start']
        self.assertEqual(fn(pd.Series([0, 0, 3, 0, 0], index=self.dates), 2), self.dates[3])
        self.assertIsNone(fn(pd.Series([0, 0, 0, 0, 3], index=self.dates), 2))
        self.assertEqual(fn(pd.Series([np.nan, np.nan, 0, 0, 0], index=self.dates), 2), self.dates[2])
        self.assertEqual(fn(pd.Series([0] * 5, index=self.dates), 2), self.dates[0])
        self.assertIsNone(fn(pd.Series(dtype=float), 2))
        self.assertIsNone(fn(pd.Series([np.nan], index=self.dates[:1]), 2))

    def test_similarity_crossing_and_no_crossing(self):
        fn = self.functions['first_similarity_crossing']
        self.assertIsNone(fn(pd.Series([0.2, 0.5, 0.8], index=self.dates[:3])))
        self.assertIsNone(fn(pd.Series(dtype=float)))
        self.assertEqual(fn(pd.Series([np.nan, 0.9, 0.99], index=self.dates[:3])), self.dates[1])

    def test_cpca_eigenpairs_and_rejected_inputs(self):
        fn = self.functions['run_cpca']
        bg = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        target = bg * [2, 1]
        vectors, values, alpha = fn(bg, target)
        contrast = np.cov(target, rowvar=False) - alpha * np.cov(bg, rowvar=False)
        np.testing.assert_allclose(contrast @ vectors, vectors * values, atol=1e-12)
        self.assertTrue(np.isfinite(alpha))
        for invalid in (np.zeros((4, 2)), bg[:1], np.full((4, 2), np.nan)):
            with self.assertRaises(ValueError): fn(invalid, target)
            with self.assertRaises(ValueError): fn(bg, invalid)
        with self.assertRaisesRegex(ValueError, 'matching features'):
            fn(bg, np.column_stack([target, target[:, 0]]))

    def test_detector_rejects_one_observation_windows(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            self.functions['detect_regime_shift'](pd.DataFrame(), target_size=1)

    def test_detector_transitions_preserve_past_events(self):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.functions.update(PCA=PCA, StandardScaler=StandardScaler)
        rng = np.random.default_rng(42)
        x = rng.normal(size=240)
        noise = rng.normal(scale=0.05, size=240)
        y = x + noise
        y[100:124] = -5 * x[100:124] + noise[100:124]
        data = pd.DataFrame({'x': x, 'y': y},
                            index=pd.date_range('2000-01-01', periods=240, freq='MS'))
        detect = self.functions['detect_regime_shift']
        with redirect_stdout(StringIO()):
            full = detect(data)
            prefix = detect(data.iloc[:180])
        self.assertEqual(full['type'].iloc[:2].tolist(), ['SHOCK_START', 'RECOVERY'])
        self.assertGreaterEqual(full['date'].iloc[0], data.index[100])
        self.assertGreaterEqual(full['date'].iloc[1], data.index[124])
        self.assertTrue(np.isfinite(full['score']).all())
        past_events = full[full['date'] <= data.index[179]].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix, past_events)


if __name__ == '__main__':
    unittest.main()
