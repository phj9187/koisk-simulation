import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import torch
from kiosk_sim.models import Customer,Menu,Store
from kiosk_research.realism import Realism,Inventory
from kiosk_research.config import Config
from kiosk_research.simulation import simulate
from kiosk_research.policy import Policy
from kiosk_research.actor_critic import ActorCriticPolicy
from kiosk_research.episodes import run_episode
from kiosk_research.engine import Experiment
from kiosk_research.convergence import fixed_probes
from kiosk_research.diagnostics import diagnose,sensitivity
from kiosk_research.worker import store_day
from test_pareto_research import small,deterministic

SCENARIO={'customer':[.45,.25,.30],'store':[.75,.25]}

def shop():
    return Store([Menu('one',(.5,.5,.5),5.,2.,5.,'a',1,None,.2),Menu('two',(.7,.3,.5),6.,2.,7.,'a',1,None,.3)],{'a':20},1)

class RealismTests(unittest.TestCase):
    def test_fifo_expiry_restock_and_accounting(self):
        s=shop();settings=Realism(inventory_horizon_days=3,shelf_life_days=2,holding_cost_per_unit=.1,disposal_cost_per_unit=.5)
        inventory=Inventory(s,settings)
        first,cost=inventory.morning(s,1);self.assertEqual(cost,0.)
        closed=inventory.close({'a':20},{'a':17})
        self.assertEqual((closed['carried_units'],closed['ordered_units']),(17,3))
        second,cost=inventory.morning(s,2)
        self.assertEqual((second.stock['a'],cost),(20,6.))
        closed=inventory.close({'a':20},{'a':18})
        self.assertEqual((closed['expired_units'],closed['carried_units']),(15,3))
        self.assertAlmostEqual(closed['inventory_loss'],37.5)
        self.assertAlmostEqual(closed['holding_cost'],.3)

    def test_abandonment_reduces_serving_and_wait_decomposes(self):
        people=[Customer((.5,.5,.5),.5,.5,None,.8,30.)]*20
        early={p:(1.,) for p in ('meal','dessert','mixed')}
        base=simulate(people,shop(),Policy.initial('no_recommendation'),.5,SCENARIO,4,hours=1,raw=True,decision_time_mode='equal',purpose_profiles=early)
        limited=simulate(people,shop(),Policy.initial('no_recommendation'),.5,SCENARIO,4,hours=1,raw=True,decision_time_mode='equal',purpose_profiles=early,realism=Realism(patience_minutes=1.,seats=1,dine_in_fraction=1.))
        self.assertGreater(limited['abandoned'],0);self.assertLess(limited['served'],base['served'])
        for result in (base,limited):
            self.assertAlmostEqual(result['mean_total_wait'],sum(result[k] for k in ('mean_seat_wait','mean_kiosk_wait','mean_decision_time','mean_cook_wait','mean_cook_time')))

    def test_top_three_and_common_service_randomness(self):
        people=[Customer((.5,.5,.5),.5,.5,None,1.,30.)]*5
        a=simulate(people,shop(),Policy.initial('fixed'),.5,SCENARIO,3,raw=True,realism=Realism(recommendation_count=3,cook_time_cv=.3))
        b=simulate(people,shop(),Policy.initial('fixed'),.5,SCENARIO,3,raw=True,realism=Realism(recommendation_count=3,cook_time_cv=.3))
        self.assertEqual(a,b)
        self.assertTrue(all(len(r['recommendations'])==2 for r in a['logs'] if r['served']))
        self.assertNotEqual(a['mean_cook_time'],5.)

    def test_fixed_return_to_go_and_contextual_terminal_boundary(self):
        policy=ActorCriticPolicy('fixed',SCENARIO);policy.training=True
        person=Customer((.5,.5,.5),.5,.5,None,.8,30.)
        with patch.object(policy,'_update') as update:
            for day,reward in enumerate((.2,.4)):
                policy.begin_episode(shop(),.5,SCENARIO,1,8,continuing=day>0)
                policy.weights(person,shop(),0.,.5)
                policy.end_episode(reward,terminal=day==1,discount=.9)
            policy.end_day()
            targets=[s[2] for s in update.call_args.args[0]]
            self.assertAlmostEqual(targets[0],.2+.9*.4);self.assertAlmostEqual(targets[1],.4)
        policy=ActorCriticPolicy('contextual',SCENARIO);policy.training=True
        policy.begin_episode(shop(),.5,SCENARIO,1,8);policy.weights(person,shop(),0.,.5);policy.feedback(.2)
        with patch.object(policy,'_update') as update:
            policy.end_episode(.2,terminal=False);self.assertIsNotNone(policy.pending);update.assert_not_called()
            policy.begin_episode(shop(),.5,SCENARIO,1,8,continuing=True);policy.weights(person,shop(),0.,.5)
            update.assert_called_once()

    def test_multiday_resume_pairing_and_disjoint_dates(self):
        config=small(algorithm='actor_critic',models=['no_recommendation','fixed','contextual'],decision_time_mode='equal',
                     realism={'inventory_horizon_days':3,'recommendation_count':2,'patience_minutes':20.,'cook_time_cv':.2,'demand_cv':.2,'repeat_fraction':.3},raw_logs=True,auto_cleanup=False)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            full=Experiment(config,a);full.run()
            paused=Experiment(config,b);paused.run(day_budget=1)
            resumed=Experiment.resume(b,paused.id);resumed.run()
            pd.testing.assert_frame_equal(deterministic(full.records()),deterministic(resumed.records()))
            self.assertEqual(full.state['stores'],resumed.state['stores'])
            f=full.records();self.assertTrue(f.horizon_days.eq(3).all())
            self.assertLess(f[f.day==1].calendar_end.max(),f[f.day==2].calendar_start.min())
            self.assertEqual(f.groupby(['day','store_id']).visitors.nunique().max(),1)

    def test_probes_use_shared_environment_and_are_fixed(self):
        config=small(algorithm='actor_critic',visitors=7,realism={'inventory_horizon_days':2})
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(config,root);run.run(day_budget=1)
            with patch('kiosk_research.convergence.run_episode',wraps=run_episode) as called:
                first=fixed_probes(run,True)
                self.assertEqual(called.call_args.args[0].visitors,7)
                self.assertEqual(called.call_args.kwargs['namespace'],'convergence')
            run.state['day']=999
            second=fixed_probes(run,True)
            for a,b in zip(first[1:],second[1:]):self.assertTrue((a==b).all())

    def test_repetition_initialization_and_frozen_diagnostics(self):
        config=small(algorithm='actor_critic')
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(config,root);case=run.state['cases'][0]
        first=store_day((config,case,None,1));second=store_day((config,case,first,4))
        self.assertNotEqual(first['policies']['balanced:fixed']['payload'],second['policies']['balanced:fixed']['payload'])
        policy=Policy.restore(first['policies']['balanced:fixed']);before=copy.deepcopy(policy.to_dict())
        rows=diagnose(config,case,policy,SCENARIO,3,'balanced',1)
        self.assertEqual(before,policy.to_dict());self.assertEqual(set(r['arm'] for r in rows),{'initial','trained','rule','no_recommendation'})
        self.assertTrue(all(r['order_change_rate']==0 for r in rows if r['arm']=='initial'))

    def test_invalid_hypotheses_and_input_mutation_fail_closed(self):
        for value in ({'recommendation_count':4},{'cook_time_cv':-1},{'weekday_multipliers':[1.]},{'seats':-1}):
            with self.assertRaises(ValueError):Config(realism=value).validate()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'sales.csv'
            from test_real_data_inputs import sales_csv
            path=sales_csv(root);config=small(sales_profile_path=str(path))
            run=Experiment(config,root);run.run(day_budget=1)
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'differ'):Experiment.resume(root,run.id)

    def test_empty_day_retains_prior_action_and_fresh_episode_resets_discount(self):
        policy=ActorCriticPolicy('contextual',SCENARIO);policy.training=True
        person=Customer((.5,.5,.5),.5,.5,None,.8,30.)
        policy.begin_episode(shop(),.5,SCENARIO,1,8);policy.weights(person,shop(),0.,.5);policy.feedback(.2)
        policy.end_episode(.2,terminal=False,discount=.9)
        policy.begin_episode(shop(),.5,SCENARIO,0,8,continuing=True);policy.feedback(-.1)
        self.assertIsNotNone(policy.pending[1]);self.assertAlmostEqual(policy.pending[2],.1)
        policy.end_episode(0.,terminal=False,discount=.9)
        self.assertAlmostEqual(policy.transition_discount,.81)
        policy.begin_episode(shop(),.5,SCENARIO,1,8)
        self.assertIsNone(policy.pending);self.assertEqual(policy.transition_discount,1.)

if __name__=='__main__':unittest.main()
