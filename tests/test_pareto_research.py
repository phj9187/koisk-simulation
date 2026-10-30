import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import pandas as pd
from kiosk_research.config import Config
from kiosk_research.policy import Policy
from kiosk_research.engine import Experiment
from kiosk_research.analysis import nondominated,hypervolume,aggregate,compare,recommend,export
from kiosk_research.persistence import save_checkpoint,load_checkpoint
from kiosk_research.worker import restore_store
from kiosk_sim.generators import customers
from kiosk_research.simulation import simulate

def small(**changes):
    values=dict(stores=8,visitors=3,train_days=1,validation_days=1,evaluation_days=1,repetitions=1,checkpoint_every=1,workers=1,plots=False,train_lambdas=2,
                validation_lambdas=[0.,.5,1.],evaluation_lambdas=[0.,.5,1.],scenarios={'balanced':{'customer':[.45,.25,.30],'store':[.75,.25]}})
    values.update(changes);return Config(**values)

def deterministic(frame):return frame.drop(columns=['elapsed_seconds']).sort_values(['day','store_id','scenario','model','lambda_value']).reset_index(drop=True)

class ResearchTests(unittest.TestCase):
    def test_store_frontier_analysis_does_not_force_shared_lambda(self):
        # Two stores need opposite lambdas. Averaging outcomes before computing
        # HV answers a different question than averaging store-specific HVs.
        rows=[]
        for store in (0,1):
            for lam in (0.,1.):
                for model in ('fixed','contextual'):
                    value=.8 if model=='contextual' and lam==store else .2
                    rows.append(dict(repetition=1,scenario='a',store_id=store,lambda_value=lam,model=model,customer=value,store=value))
        points=pd.DataFrame(rows)
        local=compare(points,42,per_store=True)[0]
        pooled=compare(points.groupby(['repetition','scenario','model','lambda_value'],as_index=False)[['customer','store']].mean(),42)[0]
        self.assertAlmostEqual(local['mean_hypervolume_contextual_minus_fixed'],.60)
        self.assertAlmostEqual(pooled['mean_hypervolume_contextual_minus_fixed'],.21)

    def test_evaluation_is_independent_monotone_and_reweightable(self):
        from kiosk_research.evaluation import score_outcomes,CUSTOMER_METRICS,STORE_METRICS
        components=dict.fromkeys(CUSTOMER_METRICS+STORE_METRICS,.4)
        scenario=small().scenarios['balanced']
        base=score_outcomes(components,scenario,.5)
        for key in components:
            improved=score_outcomes({**components,key:.8},scenario,.5)
            self.assertGreater(improved['reward'],base['reward'])
        self.assertEqual(score_outcomes(components,scenario,1.)['reward'],base['customer'])
        self.assertEqual(score_outcomes(components,scenario,0.)['reward'],base['store'])
        with self.assertRaises(ValueError):score_outcomes({**components,'margin_component':float('nan')},scenario,.5)

    def test_no_risk_has_no_artificial_inventory_reward(self):
        from dataclasses import replace
        from kiosk_sim.models import Customer,Menu,Store
        person=Customer((.5,.5,.5),.7,.6,None,1.,20.)
        menu=Menu('one',(.5,.5,.5),10.,5.,3.,'item',1,None,0.)
        scenario=small().scenarios['balanced']
        store=Store([menu],{'item':2},1)
        out=simulate([person],store,Policy.initial('fixed'),.5,scenario,1)
        self.assertEqual(out['waste_reduction_component'],0.)
        self.assertFalse(out['risk_reduction_applicable'])
        store.menus=[replace(menu,expiry_risk=.5)]
        out=simulate([person],store,Policy.initial('fixed'),.5,scenario,1)
        self.assertAlmostEqual(out['waste_reduction_component'],.5)
        store.stock={'item':0}
        out=simulate([person],store,Policy.initial('fixed'),.5,scenario,1)
        self.assertEqual(out['customer'],0.);self.assertEqual(out['served'],0)

    def test_recommendation_features_have_stable_scales_and_price_signal(self):
        from dataclasses import replace
        from kiosk_sim.models import Customer,Menu,Store
        from kiosk_research.policy import recommendation_features,COMPONENTS
        from kiosk_research.evaluation import ingredient_risks
        person=Customer((.5,.5,.5),.7,.6,None,1.,20.)
        menu=Menu('one',(.5,.5,.5),10.,5.,3.,'item',1,None,.5)
        cheap=replace(menu,name='cheap',price=5.,cost=0.)
        store=Store([menu,cheap],{'item':2},1)
        def features(m,now=0):return recommendation_features(person,store,m,0.,now,25.,ingredient_risks(store),.5,5.,1.)
        a,b=features(menu),features(cheap)
        self.assertEqual(len(a),len(COMPONENTS))
        self.assertTrue(all(0<=v<=1 for v in a+b))
        self.assertEqual(a[:4],b[:4]);self.assertGreater(b[4],a[4])
        self.assertLess(features(menu,100.)[3],a[3])
        # Policy cannot see or modify evaluator coefficients.
        before=simulate([person],store,Policy.initial('fixed'),.5,small().scenarios['balanced'],1,raw=True)
        after=simulate([person],store,Policy.initial('fixed'),.5,{'customer':[0.,1.,0.],'store':[0.,1.]},1,raw=True)
        self.assertEqual(before['logs'],after['logs'])

    def test_validation_guard_and_configuration_snapshot(self):
        points=pd.DataFrame([{'model':'fixed','lambda_value':.5,'customer':.6,'store':.7,'phase':'evaluation'}])
        with self.assertRaises(ValueError):recommend(points,min_customer=.5)
        points=points.drop(columns='phase')
        mixed=pd.concat([points.assign(scenario='a'),points.assign(scenario='b')])
        with self.assertRaises(ValueError):recommend(mixed,min_customer=.5)
        with tempfile.TemporaryDirectory() as root:
            config=small();run=Experiment(config,root)
            config.scenarios['balanced']['customer'][0]=1.
            self.assertEqual(run.config.scenarios['balanced']['customer'][0],.45)

    def test_bad_configuration(self):
        for values in ({'visitors':-1},{'workers':5},{'models':['fixed','fixed']},{'sigma':float('nan')},{'evaluation_lambdas':[0.,2.]},{'scenarios':{'bad':{'customer':[1,1,1],'store':[.5,.5]}}}):
            with self.subTest(values=values),self.assertRaises(ValueError):small(**values).validate()

    def test_lambda_function_and_common_environment(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(),root);store=restore_store(run.state['cases'][0]['template'])
            import random
            person=customers(1,random.Random(1),store)[0]
            policy=Policy.initial('fixed');policy.theta[0][1]=2.
            self.assertLess(policy.weights(person,store,0.,0.)[0],policy.weights(person,store,0.,1.)[0])
            self.assertAlmostEqual(sum(policy.weights(person,store,0.,.5)),1.)
            original=copy.deepcopy(store)
            a=simulate([person],store,Policy.initial('fixed'),.3,run.config.scenarios['balanced'],42)
            b=simulate([person],store,Policy.initial('contextual'),.3,run.config.scenarios['balanced'],42)
            self.assertEqual(a,b);self.assertEqual(store,original)
            self.assertAlmostEqual(a['reward'],.3*a['customer']+.7*a['store'])

    def test_pareto_hypervolume(self):
        p=np.array([[.2,.9],[.8,.3],[.1,.1],[.2,.9]])
        self.assertEqual(nondominated(p).tolist(),[True,True,False,True]);self.assertAlmostEqual(hypervolume(p),.36)
        with self.assertRaises(ValueError):hypervolume([[2.,1.]])

    def test_checksum(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'s.json';save_checkpoint(p,{'value':1});v=json.loads(p.read_text(encoding='utf-8'));v['state']['value']=2;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):load_checkpoint(p)

    def test_freeze_equal_budgets_and_no_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);r.run(day_budget=1);trained=copy.deepcopy(r.state['stores'])
            r=Experiment.resume(root,r.id);r.run()
            for key,v in trained.items():
                self.assertEqual(v['policies'],r.state['stores'][key]['policies']);self.assertEqual(v['optimizer_steps'],r.state['stores'][key]['optimizer_steps'])
            f=r.records();self.assertFalse(f.duplicated(['day','store_id','scenario','model','lambda_value']).any())
            counts=f.groupby(['phase','model']).simulator_calls.sum().unstack();self.assertTrue((counts.fixed==counts.contextual).all())
            target=export(r);self.assertTrue((target/'function_curves.csv').exists());self.assertEqual(target,export(r))

    def test_resume_identical_to_uninterrupted(self):
        with tempfile.TemporaryDirectory() as root:
            a=Experiment(small(),root,'full');a.run();b=Experiment(small(),root,'pause');b.run(day_budget=1);b=Experiment.resume(root,b.id);b.run()
            pd.testing.assert_frame_equal(deterministic(a.records()),deterministic(b.records()));self.assertEqual(a.state['stores'],b.state['stores'])

    def test_day_boundary_stop(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);r.run(progress=lambda day,done,total:r.request_stop() if done==1 else None)
            self.assertEqual(r.state['day'],1);self.assertEqual(r.state['stop_reason'],'paused');self.assertEqual(r.records().store_id.nunique(),8)
            r=Experiment.resume(root,r.id);r.run();self.assertEqual(r.state['day'],3)

    def test_partial_store_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root)
            def fail(day,done,total):
                if done==2:raise RuntimeError('simulated interruption')
            with self.assertRaises(RuntimeError):r.run(progress=fail)
            self.assertEqual(r.state['day'],0);paths=sorted(r.checkpoints.glob('partial_000001/store_*.json'));self.assertEqual(len(paths),2)
            saved=paths[0].read_bytes();r=Experiment.resume(root,r.id);r.run(day_budget=1);self.assertEqual(paths[0].read_bytes(),saved)

    def test_corrupt_latest_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);r.run(day_budget=1);expected=deterministic(r.records());(r.checkpoints/'day_000001.json').write_text('corrupt')
            with warnings.catch_warnings(record=True) as caught:r=Experiment.resume(root,r.id)
            self.assertTrue(caught);self.assertEqual(r.state['day'],0);r.run(day_budget=1);pd.testing.assert_frame_equal(expected,deterministic(r.records()))

    def test_stale_runner_and_changed_config(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);stale=Experiment.resume(root,r.id);r.run(day_budget=1)
            with self.assertRaises(RuntimeError):stale.run()
            with self.assertRaises(ValueError):Experiment(small(seed=123),root,r.id,True)
            lock=r.directory/'RUNNING.lock';lock.write_text('active')
            with self.assertRaises(RuntimeError):r.run()
            lock.unlink()

    def test_zero_visitors(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(visitors=0),root);r.run();self.assertFalse(r.records().eligible.any());self.assertTrue(aggregate(r.records()).empty)
            for s in r.state['stores'].values():self.assertEqual(set(s['optimizer_steps'].values()),{0})

    def test_fresh_independent_repetition(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(repetitions=2),root);r.run(day_budget=4);self.assertEqual(r.config.stage(r.state['day']),(2,'train',1))
            for s in r.state['stores'].values():self.assertEqual(set(s['optimizer_steps'].values()),{1})

    def test_validation_selection_needs_constraint(self):
        p=pd.DataFrame([{'model':'fixed','lambda_value':0.,'customer':.2,'store':.9},{'model':'contextual','lambda_value':1.,'customer':.8,'store':.4}])
        with self.assertRaises(ValueError):recommend(p)
        self.assertEqual(recommend(p,min_customer=.7)['model'],'contextual');self.assertEqual(recommend(p,min_customer=.99)['status'],'no_feasible_policy')

    def test_missing_upload_token(self):
        from kiosk_research.github import upload_summary
        self.assertEqual(upload_summary('/nonexistent','phj9187/koisk-simulation',None)['status'],'skipped')

    def test_auth_failure_is_not_success(self):
        from kiosk_research.github import upload_summary,GitHubError
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);r.run();target=export(r)
            response=unittest.mock.Mock(status_code=401)
            with patch('requests.Session.request',return_value=response):
                with self.assertRaises(GitHubError):upload_summary(target,'phj9187/koisk-simulation','test-only-not-a-real-token')
            self.assertFalse((target/'github_upload.json').exists())

    def test_parallel_matches_serial(self):
        with tempfile.TemporaryDirectory() as root:
            a=Experiment(small(),root,'serial');a.run(day_budget=1);b=Experiment(small(workers=2),root,'parallel');b.run(day_budget=1)
            pd.testing.assert_frame_equal(deterministic(a.records()),deterministic(b.records()))

    def test_repetition_bootstrap_not_store_day_pseudoreplication(self):
        points=pd.DataFrame([{'repetition':r,'scenario':'test','model':model,'lambda_value':.5,'customer':c,'store':.5} for r in range(1,6) for model,c in [('fixed',.4),('contextual',.6)]])
        result=compare(points,42)[0]
        self.assertEqual(result['independent_repetitions'],5)
        self.assertAlmostEqual(result['mean_hypervolume_contextual_minus_fixed'],.1)
        self.assertTrue(all(abs(v-.1)<1e-12 for v in result['ci95']))

    def test_raw_logs_and_stock_are_saved(self):
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(raw_logs=True),root);r.run(day_budget=1)
            self.assertEqual(len(r.state['raw_parts']),1)
            logs=pd.read_parquet(r.directory/r.state['raw_parts'][0]['path'])
            self.assertEqual(len(logs),8*2*2*3)
            for store in r.state['stores'].values():
                for inv in store['inventory'].values():
                    self.assertTrue(all(v>=0 for v in inv['terminal'].values()))
                    self.assertLessEqual(sum(inv['terminal'].values()),sum(inv['initial'].values()))

    def test_full_upload_api_flow_uses_new_branch(self):
        from kiosk_research.github import upload_summary
        with tempfile.TemporaryDirectory() as root:
            r=Experiment(small(),root);r.run();target=export(r)
            calls=[]
            def response(method,url,**kwargs):
                calls.append((method,url,kwargs.get('json')))
                if method=='GET' and url.endswith('koisk-simulation'):value={'default_branch':'main'}
                elif '/git/ref/heads/' in url:value={'object':{'sha':'parent'}}
                elif '/git/commits/parent' in url:value={'sha':'parent','tree':{'sha':'tree'}}
                elif method=='GET' and '/git/trees/' in url:value={'truncated':False,'tree':[]}
                else:value={'sha':'newsha'}
                out=unittest.mock.Mock(status_code=200);out.json.return_value=value;return out
            with patch('requests.Session.request',side_effect=response):result=upload_summary(target,'phj9187/koisk-simulation','test-not-real')
            self.assertEqual(result['status'],'uploaded')
            refs=[data for method,url,data in calls if url.endswith('/git/refs')]
            self.assertEqual(len(refs),1);self.assertTrue(refs[0]['ref'].startswith('refs/heads/results/'))
            self.assertTrue((target/'github_upload.json').exists())
            self.assertFalse(any('test-not-real' in str(v) for v in calls))

    def test_widget_start_returns_and_stop_callback_pauses_full_day(self):
        import threading
        from kiosk_research.ui import controls
        import kiosk_research.engine as engine
        original=engine.store_day;entered=threading.Event();release=threading.Event()
        def delayed(task):
            entered.set()
            if not release.wait(5):raise RuntimeError('test gate timed out')
            return original(task)
        with tempfile.TemporaryDirectory() as root,patch('IPython.display.display') as rendered:
            state=controls(root)
            container=rendered.call_args.args[0]
            fields=[]
            def walk(widget):
                fields.append(widget)
                for child in getattr(widget,'children',[]):walk(child)
            walk(container)
            by_description={getattr(w,'description',''):w for w in fields}
            by_description['방문자 수'].value=0
            by_description['그래프 저장'].value=False
            with patch('kiosk_research.engine.store_day',side_effect=delayed):
                by_description['새 실험 시작'].click()
                self.assertIsNotNone(state['future']);self.assertTrue(entered.wait(5))
                self.assertFalse(state['future'].done())
                by_description['오늘 완료 후 중지'].click();release.set();state['future'].result(timeout=60)
            self.assertEqual(state['experiment'].state['day'],1)
            self.assertEqual(state['experiment'].state['stop_reason'],'paused')
            self.assertEqual(state['experiment'].records().store_id.nunique(),8)

    def test_real_source_checksum_cannot_be_bypassed(self):
        from kiosk_cloud.data import ensure_survey
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'source.xlsx';path.write_bytes(b'wrong-file')
            with self.assertRaises(ValueError):ensure_survey(path)

    def test_notebook_and_original_integrity(self):
        root=Path(__file__).resolve().parents[1];n=json.loads((root/'notebooks/run_simulation.ipynb').read_text(encoding='utf-8'))
        for i,c in enumerate(n['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'cell-{i}','exec');self.assertFalse(c['outputs'])
        import hashlib
        for p,sha in json.loads((root/'docs/original_source_sha256.json').read_text(encoding='utf-8')).items():self.assertEqual(hashlib.sha256((root/p).read_bytes()).hexdigest(),sha)

if __name__=='__main__':unittest.main()
