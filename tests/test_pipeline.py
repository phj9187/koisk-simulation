import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import warnings
from unittest.mock import patch
import pandas as pd
from kiosk_research.config import Config
from kiosk_research.design import screening_design,interior_validation_design
from kiosk_research.engine import Experiment
from kiosk_research.persistence import unique_records,read_json
from kiosk_research.analysis import aggregate,export
from kiosk_research.cross_evaluation import cross_evaluate
from kiosk_research.session import status,run_session
from test_pareto_research import small,deterministic


class PipelineTests(unittest.TestCase):
    def test_screening_design_and_holdout(self):
        d=screening_design();self.assertEqual(len(d),9)
        Config(scenarios=d).validate()
        self.assertEqual(len({tuple(v['customer']+v['store']) for v in d.values()}),9)
        self.assertFalse(set(d)&set(interior_validation_design()))
        d['customer_uniform']['customer'][0]=9
        self.assertAlmostEqual(screening_design()['store_profit']['customer'][0],1/3)
        with self.assertRaises(ValueError):small(checkpoint_keep=1).validate()

    def test_cross_evaluation_freezes_actions_and_uses_common_tradeoff(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);r.run();original=copy.deepcopy(r.state)
            points=aggregate(r.records())
            criteria={'price':{'customer':[0.,1.,0.],'store':[1.,0.]}}
            cross,robust=cross_evaluate(points,criteria)
            self.assertEqual(r.state,original)
            self.assertEqual(set(cross.evaluation_lambda),{.5})
            self.assertEqual(set(cross.policy_lambda),{0.,.5,1.})
            self.assertTrue((cross.regret>=0).all())
            self.assertTrue((robust.top_fraction.between(0,1)).all())
            for _,row in cross.iterrows():
                source=points[(points.model==row.model)&(points.lambda_value==row.policy_lambda)].iloc[0]
                self.assertAlmostEqual(row.customer,source.affordability_per_visitor)
                self.assertAlmostEqual(row.store,source.margin_component)
            dest=export(r);self.assertTrue((dest/'cross_evaluation.csv').is_file())

    def test_lossless_duplicate_handling(self):
        a=pd.DataFrame([{'day':1,'score':.2},{'day':1,'score':.2}])
        self.assertEqual(len(unique_records(a,['day'])),1)
        a.loc[1,'score']=.3
        with self.assertRaises(ValueError):unique_records(a,['day'])

    def test_cleanup_preserves_recovery_and_unknown_files(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(repetitions=2),root);r.run(day_budget=1)
            source=r.directory/r.state['parts'][0]['path']
            duplicate=source.parent/'duplicate.parquet';shutil.copyfile(source,duplicate)
            unknown=source.parent/'unknown.parquet';unknown.write_bytes(b'not a known duplicate')
            r.run(day_budget=3)
            self.assertFalse(duplicate.exists());self.assertTrue(unknown.exists())
            self.assertEqual(len(list(r.checkpoints.glob('day_*.json'))),2)
            self.assertFalse((r.directory/'daily_metrics.parquet').exists())
            expected=deterministic(r.records())
            latest=sorted(r.checkpoints.glob('day_*.json'))[-1];latest.write_text('corrupt')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');resumed=Experiment.resume(root,r.id)
            self.assertEqual(resumed.state['day'],3)
            resumed.run(day_budget=1)
            pd.testing.assert_frame_equal(expected,deterministic(resumed.records()))

    def test_cleanup_disabled_preserves_history(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(auto_cleanup=False),root);r.run()
            self.assertEqual(len(list(r.checkpoints.glob('day_*.json'))),4)
            self.assertEqual(len(list(r.checkpoints.glob('partial_*'))),3)

    def test_cleanup_checks_result_data_before_retiring_fallbacks(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);r.run(day_budget=1)
            part=r.directory/r.state['parts'][0]['path'];part.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):r._save()
            # day 0 still provides a clean restart; no invalid checkpoint is treated
            # as sufficient proof that older data can be discarded.
            self.assertTrue((r.checkpoints/'day_000000.json').exists())

    def test_session_progress_and_resume(self):
        with tempfile.TemporaryDirectory() as root,patch('builtins.print'):
            r=run_session(root,small(),day_budget=1)
            p=status(root);self.assertEqual(p['day'],1);self.assertEqual(p['status'],'paused')
            self.assertEqual(p['checkpoint_day'],1)
            r=run_session(root,resume_id=r.id)
            self.assertEqual(status(root,r.id)['status'],'completed')
            self.assertTrue((r.directory/'session_result.json').is_file())
            self.assertFalse((r.directory/'session_error.json').exists())

    def test_notebooks_compile_and_autorun_is_opt_in_for_upload(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'notebooks').glob('run*.ipynb'):
            n=json.loads(path.read_text())
            for i,cell in enumerate(n['cells']):
                if cell['cell_type']=='code':
                    compile(''.join(cell['source']),f'{path.name}:{i}','exec')
                    self.assertFalse(cell['outputs'])
        text=(root/'notebooks/run_automatic.ipynb').read_text()
        self.assertIn('UPLOAD_SUMMARY = False',text)

if __name__=='__main__':unittest.main()
