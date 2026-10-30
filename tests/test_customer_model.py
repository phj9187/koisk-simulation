import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from kiosk_sim.models import Customer,Menu,Store
from kiosk_research.customer_model import CustomerDecision,feasibility
from kiosk_research.realism import Realism
from kiosk_research.policy import Policy
from kiosk_research.simulation import simulate
from kiosk_research.worker import restore_store
from kiosk_research.actor_critic import ActorCriticPolicy
from kiosk_research.analysis import aggregate
from kiosk_research.cross_evaluation import cross_evaluate
from test_pareto_research import small,deterministic
from kiosk_research.engine import Experiment
import pandas as pd

SCENARIO={'customer':[.45,.25,.30],'store':[.75,.25]}

def shop():
    return Store([Menu(str(i),(i/8,.4,.6),5.+i,.5+i/4,3.+i,'x',1,None,.2) for i in range(8)],{'x':100},2)

def person():return Customer((.5,.5,.5),.7,.6,None,.8,30.)

class CustomerModelTests(unittest.TestCase):
    def test_disabled_defaults_preserve_prior_numeric_and_action_trace(self):
        reference=json.loads((Path(__file__).parent/'fixtures/customer_legacy_trace.json').read_text(encoding='utf-8'))
        for case in reference['cases']:
            current=simulate([Customer(**c) for c in reference['people']],restore_store(reference['store']),Policy.initial(case['kind']),.5,
                             reference['scenario'],case['seed'],raw=True,decision_time_mode='equal')
            for key,value in case['expected'].items():
                if key=='logs':
                    self.assertEqual(len(value),len(current['logs']))
                    for old,new in zip(value,current['logs']):
                        for event_key,event_value in old.items():self.assertEqual(event_value,new[event_key],event_key)
                else:self.assertEqual(value,current[key],key)

    def test_consideration_has_no_duplicates_is_paired_and_exposes_new_menu(self):
        store=shop();c=person();settings=Realism(consideration_mean=0.)
        a=CustomerDecision(c,store,store.menus,settings,5,1,{m.name:1. for m in store.menus},0.)
        b=CustomerDecision(c,store,store.menus,settings,5,1,{m.name:1. for m in store.menus},0.)
        self.assertEqual(a.considered,b.considered);self.assertEqual(len(a.considered),1)
        unseen=next(m for m in store.menus if m not in a.considered)
        choice=a.choose([unseen],0.)
        self.assertTrue(choice['accepted']);self.assertEqual(choice['picked'],unseen)
        self.assertEqual(len(CustomerDecision(c,store,store.menus,Realism(),5,1,{m.name:1. for m in store.menus},0.).considered),8)

    def test_noise_changes_perception_not_realized_taste_and_ai_need_not_be_better(self):
        store=shop();store.menus=store.menus[:1]
        settings=Realism(taste_noise=.3,ai_taste_noise=.6)
        settings.validate()
        result=simulate([person()],store,Policy.initial('fixed'),.5,SCENARIO,7,raw=True,realism=settings)
        event=result['logs'][0]
        from kiosk_sim.recommenders import affinity
        self.assertEqual(event['taste_fit'],affinity(person(),store.menus[0]))
        self.assertNotEqual(event['taste_fit'],event['perceived_fit'])
        again=simulate([person()],store,Policy.initial('fixed'),.5,SCENARIO,7,raw=True,realism=settings)
        self.assertEqual(result,again)

    def test_stopping_search_does_not_compare_to_unseen_best(self):
        store=shop();c=person();settings=Realism(decision_rule='inspected')
        model=CustomerDecision(c,store,store.menus,settings,1,0,{m.name:1. for m in store.menus},0.)
        choice=model.choose([store.menus[-1]],0.);choice['suggestions']=[store.menus[-1]]
        self.assertEqual(choice['acceptance_reference'],model.considered[0].name)
        self.assertLessEqual(choice['inspected_count'],2)
        self.assertEqual(model.decision_minutes(choice,False,'equal'),.8+.25*choice['inspected_count'])

    def test_hidden_wait_ignores_true_cook_availability(self):
        store=shop();store.cook_free_at=[500.,600.];c=person()
        model=CustomerDecision(c,store,store.menus,Realism(wait_visibility='menu_only'),1,0,{m.name:1. for m in store.menus},10.)
        self.assertEqual(model.perceived_wait(store.menus[0]),store.menus[0].cook_minutes)
        complete=CustomerDecision(c,store,store.menus,Realism(),1,0,{m.name:1. for m in store.menus},10.)
        self.assertGreater(complete.perceived_wait(store.menus[0]),400.)

    def test_feasibility_reports_overlapping_constraints(self):
        store=shop();store.stock={'x':0};c=replace(person(),budget=0.)
        detail=feasibility(store,c)
        self.assertEqual(detail['menu_failure_masks'],{'3':8})
        result=simulate([c],store,Policy.initial('no_recommendation'),.5,SCENARIO,1,raw=True)
        self.assertEqual(result['infeasible_rescue_combined'],1)
        self.assertEqual(result['infeasible_rescue_budget'],0)

    def test_fatigue_reward_telescopes_and_frozen_reassessment_uses_daily_samples(self):
        p=ActorCriticPolicy('contextual',SCENARIO);p.training=True
        result=simulate([person()]*5,shop(),p,.5,SCENARIO,2,realism=Realism(overload_penalty=.02))
        self.assertAlmostEqual(p.episode_reward,result['reward'])
        # Clipping before aggregation differs from clipping the mean.
        row={'repetition':1,'model':'fixed','scenario':'balanced','lambda_value':.5,
             'taste_fit_per_visitor':.5,'affordability_per_visitor':0.,'wait_component_per_visitor':0.,'margin_component':.5,'waste_reduction_component':.5,
             'customer_component_samples':json.dumps([{'values':[0.,0.,0.,.2],'weight':.5},{'values':[1.,0.,0.,.2],'weight':.5}])}
        out,_=cross_evaluate(pd.DataFrame([row]),{'x':{'customer':[1.,0.,0.],'store':[1.,0.]}})
        self.assertAlmostEqual(out.customer.iloc[0],.4)

    def test_bounded_information_multiday_resume_is_exact(self):
        config=small(algorithm='actor_critic',models=['no_recommendation','fixed','contextual'],decision_time_mode='equal',raw_logs=True,
                     realism={'inventory_horizon_days':2,'consideration_mean':2.,'taste_noise':.25,'ai_taste_noise':.25,'wait_visibility':'menu_only',
                              'decision_rule':'inspected','overload_penalty':.02,'repeat_fraction':.5,'trust_learning':True})
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            full=Experiment(config,a);full.run()
            partial=Experiment(config,b);partial.run(day_budget=1);resumed=Experiment.resume(b,partial.id);resumed.run()
            pd.testing.assert_frame_equal(deterministic(full.records()),deterministic(resumed.records()))
            self.assertEqual(full.state['stores'],resumed.state['stores'])

if __name__=='__main__':unittest.main()
