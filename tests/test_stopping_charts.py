import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from kiosk_research.config import Config
from kiosk_research.convergence import training_dates,stability_metrics,observe
from kiosk_research.engine import Experiment
from kiosk_research.linecharts import export_lines,load_snapshot
from kiosk_research.persistence import read_json,write_json
from test_pareto_research import small


class StoppingTests(unittest.TestCase):
    def test_absolute_limit_never_adds_another_budget_on_resume(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(continual=True,repetitions=None),root)
            run.run(target_day=4)
            self.assertEqual(run.state['day'],4)
            self.assertEqual(run.state['stop_reason'],'target_reached')
            run=Experiment.resume(root,run.id);run.run(target_day=4)
            self.assertEqual(run.state['day'],4)
            run.run(target_day=5);self.assertEqual(run.state['day'],5)
            with self.assertRaises(ValueError):run.run(target_day=0)

    def test_full_window_rejects_slow_drift_and_return_to_start(self):
        stable=np.full((101,2,3,5),.2)
        scores=np.full((21,2,3),.5)
        d,s=stability_metrics(stable,scores);self.assertTrue((d==0).all())
        drifting=stable.copy()
        for i in range(101):drifting[i,1,:,0]+=.0005*i;drifting[i,1,:,1]-=.0005*i
        d,_=stability_metrics(drifting,scores);self.assertAlmostEqual(d[1],.05)
        stable[50,1,:,0]+=.04;stable[50,1,:,1]-=.04
        d,_=stability_metrics(stable,scores);self.assertAlmostEqual(d[1],.04)
        scores[10,1,:]+=.02
        _,s=stability_metrics(stable,scores);self.assertAlmostEqual(s[1],.02)

    def test_training_window_ignores_frozen_dates(self):
        c=small(continual=True,repetitions=None)
        self.assertEqual([training_dates(c,d) for d in range(1,7)],[1,1,1,2,2,2])
        for change in ({'early_stopping':True},{'convergence_window':99},{'convergence_tolerance':0.}):
            with self.assertRaises(ValueError):small(**change).validate()

    def test_monitor_persists_window_across_recreation_and_requires_all_policies(self):
        with tempfile.TemporaryDirectory() as root:
            c=small(continual=True,early_stopping=True,train_days=1000,repetitions=None)
            def make(day):return SimpleNamespace(directory=Path(root),state={'day':day},config=c,identity={'id':'monitor'})
            unstable=False
            def probes(run,include_scores=False):
                weights=np.full((2,3,5),.2)
                if unstable:weights[1,0,0]+=.04;weights[1,0,1]-=.04
                return ['p0','p1'],weights,np.full((2,3),.5) if include_scores else np.array([])
            with patch('kiosk_research.convergence.fixed_probes',side_effect=probes):
                for day in range(1,105):
                    result=observe(make(day));self.assertFalse(result['converged'])
                result=observe(make(105));self.assertTrue(result['converged'])
                self.assertEqual(result['stable_policies'],2)
                self.assertTrue((Path(root)/'convergence/diagnostics/day_000105.json').is_file())
                unstable=True
                result=observe(make(105));self.assertFalse(result['converged'])
                self.assertEqual(result['stable_policies'],1)
                # Corruption cannot silently produce a stable window.
                (Path(root)/'convergence/day_000100.npz').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):observe(make(106))

    def test_converged_status_saves_and_stops_at_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(continual=True,early_stopping=True,repetitions=None),root)
            with patch('kiosk_research.convergence.observe',return_value={'converged':True}):run.run()
            self.assertEqual(run.state['day'],1);self.assertEqual(run.state['stop_reason'],'converged')
            resumed=Experiment.resume(root,run.id);resumed.run()
            self.assertEqual(resumed.state['day'],1)

    def test_tv_gives_a_unit_score_change_bound(self):
        rng=np.random.default_rng(42)
        for _ in range(100):
            a=rng.dirichlet(np.ones(5));b=rng.dirichlet(np.ones(5));x=rng.random(5)
            self.assertLessEqual(abs((a-b)@x),.5*np.abs(a-b).sum()+1e-12)

    def test_charts_are_readonly_and_keep_per_store_functions(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(),root);run.run();before=copy.deepcopy(run.state)
            write_json(run.directory/'convergence/diagnostics/day_000003.json',{
                'day':3,'training_dates':105,'tolerance':.01,'max_weight_tv_bound':.008,
                'max_probe_score_range':.005,'stable_policies':16,'total_policies':16})
            readonly=load_snapshot(root,run.id);target=export_lines(readonly)
            self.assertEqual(run.state,before)
            curves=pd.read_csv(target/'evaluation_functions.csv')
            self.assertEqual(curves.store_id.nunique(),8)
            np.testing.assert_allclose(curves.reward,curves.lambda_value*curves.customer+(1-curves.lambda_value)*curves.store)
            weights=pd.read_csv(target/'recommendation_weight_functions.csv')
            self.assertEqual(len(weights),8*2*101)
            self.assertTrue((target/'evaluation_balanced.png').is_file())
            self.assertTrue((target/'convergence.png').is_file())
            self.assertIn('chartdata',(target/'index.html').read_text(encoding='utf-8'))
            self.assertEqual(read_json(target/'complete.json')['day'],3)


if __name__=='__main__':unittest.main()
