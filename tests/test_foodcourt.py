import json, tempfile, unittest
from pathlib import Path
from kiosk_foodcourt.models import Settings,generate_court,POLICIES
from kiosk_foodcourt.environment import simulate
from kiosk_foodcourt.cli import run_job,signature

class FoodcourtTests(unittest.TestCase):
 def test_nested_menus_do_not_add_resources(self):
  s=Settings();a=generate_court(s,0,12,80);b=generate_court(s,0,64,80)
  self.assertEqual(a.dishes,b.dishes[:12]);self.assertEqual(a.initial_stock,b.initial_stock);self.assertEqual(a.servers,b.servers)
 def test_inventory_and_chronology_for_every_policy(self):
  s=Settings(kiosks=3,seats=3);c=generate_court(s,0,32,150)
  totals=[]
  for policy in POLICIES:
   row,raw=simulate(c,s,policy,visitors=150,raw=True)
   self.assertEqual(row['visitors'],sum(v for k,v in row.items() if k.startswith('count_')))
   totals.append(row['visitors'])
   for initial,used,left in zip(raw['initial_stock'],raw['consumed'],raw['leftover']):
    for k in initial:self.assertEqual(initial[k],used[k]+left[k]);self.assertGreaterEqual(left[k],0)
   self.assertEqual(len(raw['records']),row['visitors'])
   times=[e['time'] for e in raw['trace']];self.assertEqual(times,sorted(times))
   active=set()
   for event in raw['trace']:
    if event['event']=='cook_start':
     key=(event['counter'],event['worker']);self.assertNotIn(key,active);active.add(key)
    elif event['event']=='food_ready':active.remove((event['counter'],event['worker']))
   self.assertFalse(active)
  self.assertEqual(len(set(totals)),1)
 def test_no_forced_recommendation_gain(self):
  s=Settings(openness=0,recommendation_overhead=1);c=generate_court(s,0,32,80)
  baseline,_=simulate(c,s,'none',visitors=80)
  recommended,_=simulate(c,s,'customer_only',visitors=80)
  self.assertEqual(recommended['accepted_recommendations'],0)
  self.assertGreater(recommended['mean_decision_time'],baseline['mean_decision_time'])
 def test_resume_equal_and_training_budget(self):
  import time
  s=Settings(menu_counts=[12],attention_modes=['bounded'],visitors=[20],repetitions=1,generations=2,population=3,train_days=1,validation_days=1,evaluation_days=1,validation_every=1,workers=1)
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   task=(0,12,'bounded',20)
   from unittest.mock import patch
   with patch('kiosk_foodcourt.cli.time.monotonic',side_effect=[0,2]):
    run_job((s.to_dict(),a,task,1))
   saved=json.loads(next(Path(a).rglob('checkpoint.json')).read_text())
   self.assertEqual(saved['generation'],1)
   pa=Path(run_job((s.to_dict(),a,task,float('inf'))))/'result.json'
   pb=Path(run_job((s.to_dict(),b,task,float('inf'))))/'result.json'
   x=json.loads(pa.read_text(encoding='utf-8'));y=json.loads(pb.read_text(encoding='utf-8'))
   self.assertEqual(x,y)
   calls=[v['calls'] for v in x['training']['models'].values()];self.assertEqual(calls,[6,6])
 def test_signature_changes_with_scientific_setting(self):
  self.assertNotEqual(signature(Settings()),signature(Settings(openness=.1)))

if __name__=='__main__':unittest.main()
