from dataclasses import replace
import random
import tempfile
import unittest
import pandas as pd
from kiosk_research.engine import Experiment
from kiosk_research.policy import Policy
from kiosk_research.simulation import simulate
from kiosk_research.worker import restore_store,store_day
from kiosk_research.baseline import compare_baseline
from kiosk_research.analysis import export
from kiosk_research.convergence import fixed_probes
from kiosk_research.linecharts import export_lines
from kiosk_research.live import LiveDashboard
from kiosk_sim.generators import customers
from test_pareto_research import small,deterministic


class BaselineTests(unittest.TestCase):
    def test_inference_uses_independent_repetitions_and_wait_improvement_direction(self):
        rows=[]
        for rep in range(1,6):
            for model,reward,wait in [('no_recommendation',0.,5.),('fixed',.1,4.)]:
                rows.append(dict(day=rep,repetition=rep,store_id=0,scenario='balanced',lambda_value=.5,
                                 phase='evaluation',eligible=True,model=model,reward=reward,mean_total_wait=wait))
        frame=pd.DataFrame(rows)
        descriptive=compare_baseline(frame)
        self.assertTrue(descriptive.ci95_low.isna().all())
        independent=compare_baseline(frame,independent=True)
        reward=independent[independent.metric.eq('reward')].iloc[0]
        self.assertAlmostEqual(reward.ci95_low,.1)
        self.assertTrue(pd.isna(reward.relative_improvement))
        wait=independent[independent.metric.eq('mean_total_wait')].iloc[0]
        self.assertEqual(wait.delta_ai_minus_baseline,-1.)
        self.assertEqual(wait.improvement,1.)

    def config(self,**changes):
        return small(algorithm='actor_critic',decision_time_mode='equal',models=['no_recommendation','fixed','contextual'],auto_cleanup=False,**changes)

    def test_rejecting_all_recommendations_matches_unassisted_choice_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(self.config(),root)
            store=restore_store(run.state['cases'][0]['template'])
            people=[replace(c,openness=0.) for c in customers(10,random.Random(3),store)]
            scenario=next(iter(run.config.scenarios.values()))
            base=simulate(people,store,Policy.initial('no_recommendation'),.5,scenario,88,raw=True,decision_time_mode='equal')
            ai=simulate(people,store,Policy.initial('fixed'),.5,scenario,88,raw=True,decision_time_mode='equal')
            for key in base:
                if key not in ('logs','mean_weights'):self.assertEqual(base[key],ai[key],key)
            self.assertEqual([e['chosen'] for e in base['logs'] if e['served']],[e['chosen'] for e in ai['logs'] if e['served']])
            self.assertEqual([e['decision_minutes'] for e in base['logs'] if e['served']],[e['decision_minutes'] for e in ai['logs'] if e['served']])
            self.assertTrue(all(e['recommended'] is None and e['chosen']==e['own_choice'] and not e['accepted'] for e in base['logs'] if e['served']))

    def test_baseline_has_no_learning_and_lambda_changes_only_scoring(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(self.config(),root)
            result=store_day((run.config,run.state['cases'][0],None,1))
            rows=[r for r in result['rows'] if r['model']=='no_recommendation']
            self.assertEqual(len(rows),2)
            for metric in ('customer','store','profit','mean_total_wait','served','leftover_units'):
                self.assertEqual(rows[0][metric],rows[1][metric])
            self.assertTrue(all(r['optimizer_steps']==0 and r['training_simulator_calls']==0 for r in rows))
            self.assertEqual(result['policies']['balanced:no_recommendation'],{'kind':'no_recommendation','theta':[]})
            with self.assertRaises(ValueError):small(models=['no_recommendation']).validate()

    def test_resume_frozen_comparison_export_live_and_probes(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(self.config(),root);run.run(day_budget=1)
            trained={k:v['policies'] for k,v in run.state['stores'].items()}
            run=Experiment.resume(root,run.id);run.run()
            self.assertEqual(trained,{k:v['policies'] for k,v in run.state['stores'].items()})
            report=compare_baseline(run.export_metrics())
            self.assertFalse(report.empty)
            self.assertEqual(set(report.model),{'fixed','contextual'})
            self.assertTrue(report.ci95_low.isna().all())
            self.assertEqual(set(report.metric)&{'mean_total_wait','reward'},{'mean_total_wait','reward'})
            paired=run.export_metrics();missing=paired[~(paired.model.eq('no_recommendation')&paired.phase.eq('evaluation'))]
            with self.assertRaises(ValueError):compare_baseline(pd.concat([missing,paired[paired.model.eq('no_recommendation')&paired.phase.eq('evaluation')].iloc[1:]]))
            keys,weights,scores=fixed_probes(run,True)
            self.assertEqual(len(keys),16)
            self.assertEqual(weights.shape[0],16)
            target=export(run)
            self.assertTrue((target/'baseline_comparison.csv').exists())
            charts=export_lines(run)
            curves=pd.read_csv(charts/'evaluation_functions.csv')
            self.assertIn('no_recommendation',set(curves.model))
            weights=pd.read_csv(charts/'recommendation_weight_functions.csv')
            self.assertNotIn('no_recommendation',set(weights.model))
            dashboard=LiveDashboard(run,'2099-01-01T00:00:00Z',display=False)
            page=(dashboard.folder/'index.html').read_text(encoding='utf-8')
            self.assertIn('no_recommendation: 추천 없음',page)


if __name__=='__main__':unittest.main()
