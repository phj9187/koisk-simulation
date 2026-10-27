import copy
import tempfile
import unittest
import pandas as pd
from kiosk_research.engine import Experiment
from kiosk_research.analysis import aggregate, compare, export
from kiosk_research.persistence import read_json
from test_pareto_research import small, deterministic


class ContinualTests(unittest.TestCase):
    def test_keeps_optimizer_and_parameters_across_cycle_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(continual=True,repetitions=None),root)
            run.run(day_budget=3)
            before=copy.deepcopy(run.state['stores'])
            run=Experiment.resume(root,run.id)
            # With perturbation disabled, the exact learned policy must persist.
            from unittest.mock import patch
            from kiosk_research.policy import Policy
            with patch.object(Policy,'perturb',lambda self,*args:self):
                run.run(day_budget=1)
            for key,value in run.state['stores'].items():
                self.assertEqual(value['policies'],before[key]['policies'])
                self.assertEqual(set(value['optimizer_steps'].values()),{2})
            self.assertFalse(run.config.complete(1000000))
            self.assertEqual(read_json(run.directory/'progress.json')['learning_mode'],'continual')

    def test_resume_matches_full_run_across_cycles_and_freezes_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            config=small(continual=True,repetitions=2)
            full=Experiment(config,root,'full');full.run()
            paused=Experiment(config,root,'paused');paused.run(day_budget=4)
            trained=copy.deepcopy(paused.state['stores'])
            paused=Experiment.resume(root,paused.id);paused.run()
            pd.testing.assert_frame_equal(deterministic(full.records()),deterministic(paused.records()))
            for key,value in paused.state['stores'].items():
                self.assertEqual(value['policies'],trained[key]['policies'])
            target=export(paused)
            stats=read_json(target/'statistics.json')
            self.assertEqual(stats['completed_repetitions'],[])
            self.assertEqual(stats['completed_cycles'],[1,2])
            for item in compare(aggregate(full.records()),42,independent=False):
                self.assertEqual(item['independent_repetitions'],0)
                self.assertIsNone(item['ci95'])
            self.assertIn('Models carry forward',(target/'report.md').read_text())


if __name__=='__main__':unittest.main()
