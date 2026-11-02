import copy,random,tempfile,unittest
from unittest.mock import patch
from kiosk_research.realism import Realism,Inventory
from kiosk_research.customer_model import CustomerDecision
from kiosk_research.simulation import simulate
from kiosk_research.policy import Policy
from kiosk_research.engine import Experiment
from test_realism import shop,SCENARIO
from test_customer_model import person
from test_pareto_research import small

class OperationalTests(unittest.TestCase):
 def test_abandoned_decision_still_occupies_kiosk(self):
  settings=Realism(operational_queue=True,decision_rule='inspected',decision_base=100.,decision_per_item=0.,patience_minutes=1.)
  r=simulate([person()]*12,shop(),Policy.initial('no_recommendation'),.5,SCENARIO,5,hours=1,raw=True,realism=settings)
  self.assertEqual(r['served'],0)
  self.assertEqual(r['kiosk_busy_minutes'],100.)
  self.assertEqual(sum(x['reason']=='predicted_wait_abandonment' for x in r['logs']),1)
  self.assertEqual(r['abandoned'],12)
 def test_outside_option_reserves_browsing_but_not_payment_or_stock(self):
  settings=Realism(operational_queue=True,decision_rule='inspected',decision_base=2.,decision_per_item=0.,outside_utility=100.,payment_minutes=1.)
  r=simulate([person()]*3,shop(),Policy.initial('no_recommendation'),.5,SCENARIO,5,raw=True,realism=settings)
  self.assertEqual(r['outside_option_customers'],3);self.assertEqual(r['served'],0)
  self.assertEqual(r['kiosk_busy_minutes'],6.);self.assertEqual(r['initial_stock'],r['terminal_stock'])
 def test_last_order_boundary_and_overtime(self):
  settings=Realism(operational_queue=True,enforce_closing=True,decision_rule='inspected',decision_base=61.,decision_per_item=0.)
  r=simulate([person()]*3,shop(),Policy.initial('no_recommendation'),.5,SCENARIO,5,hours=1,raw=True,realism=settings)
  self.assertEqual(r['served'],0);self.assertEqual(r['closed_customers'],3)
  settings.decision_base=.1;store=shop();store.menus=[__import__('dataclasses').replace(m,cook_minutes=100.) for m in store.menus]
  r=simulate([person()],store,Policy.initial('no_recommendation'),.5,SCENARIO,5,hours=1,raw=True,realism=settings)
  self.assertGreater(r['overtime_minutes'],0.)
 def test_pipeline_subtracts_in_transit_and_delivers_after_lead_time(self):
  store=shop();s=Realism(delivery_lead_days=2,shelf_life_days=30);inv=Inventory(store,s)
  inv.morning(store,1);a=inv.close({'a':20},{'a':15});self.assertEqual(a['ordered_units'],5)
  second,_=inv.morning(store,2);self.assertEqual(second.stock['a'],15)
  a=inv.close({'a':15},{'a':15});self.assertEqual(a['ordered_units'],0)
  third,cost=inv.morning(store,3);self.assertEqual(third.stock['a'],20);self.assertEqual(cost,10.)
 def test_staggered_inventory_ingredient_expiry_and_conservation(self):
  store=shop();s=Realism(initial_age_mode='staggered',ingredient_shelf_lives={'a':3},shelf_life_days=20)
  a=Inventory(store,s,7);b=Inventory(store,s,7);self.assertEqual(a.cohorts,b.cohorts)
  self.assertGreater(len(a.cohorts['a']),1)
  first,_=a.morning(store,1);r=a.close(first.stock,{'a':17})
  self.assertEqual(20,3+r['expired_units']+r['carried_units'])
  self.assertGreater(r['expired_units'],0)
 def test_gumbel_sign_and_legacy_separation(self):
  store=shop();shocks={m.name:2. for m in store.menus}
  legacy=CustomerDecision(person(),store,store.menus,Realism(),1,0,shocks,0)
  modern=CustomerDecision(person(),store,store.menus,Realism(choice_error='gumbel'),1,0,shocks,0)
  menu=store.menus[0];self.assertAlmostEqual(legacy.noisy_utility(menu)-legacy.utility(menu),-(modern.noisy_utility(menu)-modern.utility(menu)))
 def test_observed_validation_rejects_missing_store_dates(self):
  import sys
  from pathlib import Path
  import pandas as pd
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
  from validate_observed_operations import compare
  a=pd.DataFrame([dict(date='2026-11-01',store_id=1,visitors=10,orders=8,mean_total_wait_minutes=4.,expired_units=1,net_profit=20.)])
  self.assertEqual(compare(a,a)['metrics']['orders']['mae'],0.)
  with self.assertRaises(ValueError):compare(a,a.assign(store_id=2))
 def test_new_operations_resume_matches_full(self):
  c=small(algorithm='actor_critic',models=['fixed','contextual'],realism=dict(inventory_horizon_days=2,initial_age_mode='staggered',delivery_lead_days=2,operational_queue=True,enforce_closing=True,choice_error='gumbel',outside_utility=.5))
  with tempfile.TemporaryDirectory() as root:
   a=Experiment(c,root,'a');a.run();b=Experiment(c,root,'b');b.run(day_budget=1);b=Experiment.resume(root,b.id);b.run()
   for k in a.state['stores']:self.assertEqual(a.state['stores'][k]['policies'],b.state['stores'][k]['policies'])

if __name__=='__main__':unittest.main()
