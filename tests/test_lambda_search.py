import copy
import tempfile
import unittest
import pandas as pd
from kiosk_research.config import Config
from kiosk_research.engine import Experiment
from kiosk_research.lambda_search import knee_grid,select_plan,training_grid
from kiosk_research.convergence import training_dates
from kiosk_research.analysis import aggregate,export
from kiosk_research.persistence import read_json
from kiosk_research.worker import store_day
from test_pareto_research import small,deterministic


class LambdaSearchTests(unittest.TestCase):
    def config(self,**changes):
        return small(algorithm='actor_critic',decision_time_mode='equal',lambda_search='pareto_two_stage',
            train_days=2,train_lambdas=3,models=['no_recommendation','fixed','contextual'],auto_cleanup=False,**changes)

    def points(self,scenario='balanced',bend=.7):
        rows=[]
        for model in ('fixed','contextual','no_recommendation'):
            for lam,c,s in [(0.,.1,.9),(.5,bend,bend),(1.,.9,.1)]:
                rows.append(dict(day=2,phase='validation',search_stage='selection',scenario=scenario,model=model,
                                 lambda_value=lam,customer=c if model!='no_recommendation' else 1.,
                                 store=s if model!='no_recommendation' else 1.,store_id=0,eligible=True))
        return rows

    def test_geometric_knee_and_no_evidence_fallback(self):
        points=[dict(model='fixed',lambda_value=l,customer=c,store=s) for l,c,s in [(0,.1,.9),(.5,.7,.7),(1,.9,.1),(.25,.3,.3)]]
        result=knee_grid(points)
        self.assertEqual(result['grid'],[.25,.5,.75])
        self.assertEqual(result['knee_lambda'],.5)
        self.assertEqual(len(result['front']),3)
        flat=[dict(model='fixed',lambda_value=l,customer=l,store=1-l) for l in (0.,.5,1.)]
        self.assertEqual(knee_grid(flat)['reason'],'no_clear_knee_keep_global')
        self.assertEqual(knee_grid(points[:1])['grid'],[0.,.5,1.])

    def test_scenarios_separate_baseline_excluded_and_no_final_test_leakage(self):
        c=self.config();c.scenarios['second']={'customer':[0.,0.,1.],'store':[1.,0.]}
        frame=pd.DataFrame(self.points()+self.points('second',.5))
        plan=select_plan(frame,c,2)
        self.assertEqual(plan['choices']['balanced']['grid'],[.25,.5,.75])
        self.assertEqual(plan['choices']['second']['grid'],[0.,.5,1.])
        self.assertTrue(all(p['model']!='no_recommendation' for p in plan['choices']['balanced']['candidates']))
        with self.assertRaises(ValueError):select_plan(frame.assign(phase='evaluation'),c,2)
        with self.assertRaises(ValueError):select_plan(frame.assign(search_stage='validation'),c,2)

    def test_schedule_counts_and_global_coverage(self):
        c=self.config().validate()
        self.assertEqual(c.cycle_days,5)
        self.assertEqual([c.search_stage(d) for d in range(1,6)],['coarse','selection','refine','validation','evaluation'])
        self.assertEqual([training_dates(c,d) for d in range(7)],[0,1,1,2,2,2,3])
        c.train_days=10;c.validate()
        plan=select_plan(pd.DataFrame(self.points()).assign(day=6),c,6)
        self.assertEqual(training_grid(c,7,'balanced',plan),([.25,.5,.75],'refinement'))
        self.assertEqual(training_grid(c,11,'balanced',plan),([0.,.5,1.],'global_coverage'))
        with self.assertRaises(ValueError):training_grid(c,7,'balanced')
        with self.assertRaises(ValueError):Config(lambda_search='pareto_two_stage').validate()

    def test_persisted_plan_resume_freeze_baseline_and_final_validation(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            full=Experiment(c,a);full.run()
            paused=Experiment(c,b);paused.run(day_budget=1)
            trained={k:copy.deepcopy(v['policies']) for k,v in paused.state['stores'].items()}
            paused.run(day_budget=1)
            self.assertEqual(trained,{k:v['policies'] for k,v in paused.state['stores'].items()})
            plan=copy.deepcopy(paused.state['lambda_search'])
            self.assertEqual(plan['1']['selected_on_day'],2)
            self.assertEqual(read_json(paused.directory/'lambda_search/cycle_0001.json'),plan['1'])
            resumed=Experiment.resume(b,paused.id);resumed.run()
            self.assertEqual(resumed.state['day'],5)
            self.assertEqual(resumed.state['lambda_search'],plan)
            self.assertEqual(full.state['stores'],resumed.state['stores'])
            pd.testing.assert_frame_equal(deterministic(full.records()),deterministic(resumed.records()))
            frame=resumed.records()
            for (_,scenario),group in frame.groupby(['day','scenario']):
                grids=group.groupby('model').lambda_value.apply(lambda s:tuple(sorted(s.unique())))
                self.assertEqual(len(set(grids)),1)
            self.assertTrue(frame[frame.model.eq('no_recommendation')].optimizer_steps.eq(0).all())
            selected=aggregate(frame,True,'validation')
            expected=aggregate(frame[frame.day.eq(4)],True,'validation')
            pd.testing.assert_frame_equal(selected,expected)
            report=export(resumed)
            self.assertTrue((report/'baseline_comparison.csv').exists())
            self.assertTrue((report/'lambda_search.json').exists())
            self.assertTrue((report/'lambda_schedule.csv').exists())

    def test_refinement_worker_uses_selected_grid_for_all_three_arms(self):
        c=self.config(visitors=0)
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(c,root)
            previous=store_day((c,run.state['cases'][0],None,1))
            plan=select_plan(pd.DataFrame(self.points()),c,2)
            result=store_day((c,run.state['cases'][0],previous,3,plan))
            for model in c.models:
                rows=[r for r in result['rows'] if r['model']==model]
                self.assertEqual([r['lambda_value'] for r in rows],[.25,.5,.75])
                self.assertTrue(all(r['search_stage']=='refine' and r['lambda_schedule']=='refinement' for r in rows))


if __name__=='__main__':unittest.main()
