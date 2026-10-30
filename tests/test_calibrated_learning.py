import copy,tempfile,unittest
from unittest.mock import patch
import torch
from kiosk_research.engine import Experiment
from kiosk_research.actor_critic import ActorCriticPolicy
from kiosk_research.episodes import run_episode
from kiosk_research.policy import Policy
from kiosk_research.benchmarks import candidate_benchmark
from kiosk_research.convergence import fixed_probes,observe
from test_pareto_research import small

class CalibratedTests(unittest.TestCase):
 def test_budget_inventory_are_store_relative_and_deterministic(self):
  config=small(realism=dict(budget_mode='price_relative',inventory_mode='demand',expected_purchase_rate=.7,inventory_cover_days=1.1))
  with tempfile.TemporaryDirectory() as root:
   run=Experiment(config,root);case=run.state['cases'][0];seen=[]
   import kiosk_research.episodes as ep
   actual=ep.simulate
   def capture(people,store,*a,**kw):
    seen.append(([c.budget for c in people],dict(store.stock)))
    return actual(people,store,*a,**kw)
   with patch.object(ep,'simulate',capture):
    for i in range(2):run_episode(config,case,Policy.initial('fixed'),.5,config.scenarios['balanced'],1)
   self.assertEqual(seen[0],seen[1])
   import statistics,math
   price=statistics.median(m['price'] for m in case['template']['menus'])
   self.assertTrue(all(.75*price<=b<=1.8*price for b in seen[0][0]))
   expected={k:math.ceil(3*.7*1.1*sum(m['units']/len(case['template']['menus']) for m in case['template']['menus'] if m['ingredient']==k)) for k in case['template']['stock']}
   self.assertEqual(seen[0][1],expected)
 def test_paired_multiday_training_resumes_exactly(self):
  config=small(algorithm='actor_critic',models=['fixed','contextual'],learning=dict(initial_alpha=10.,fixed_candidates=3,paired_baseline=True,normalize_advantage=True),realism=dict(inventory_horizon_days=2,inventory_mode='demand',budget_mode='price_relative'))
  with tempfile.TemporaryDirectory() as root:
   full=Experiment(config,root,'full');full.run()
   part=Experiment(config,root,'part');part.run(day_budget=1);part=Experiment.resume(root,part.id);part.run()
   for k in full.state['stores']:self.assertEqual(full.state['stores'][k]['policies'],part.state['stores'][k]['policies'])
   train=full.records().query("phase=='train' and model=='fixed'")
   self.assertEqual(set(train.training_simulator_calls),{6})
 def test_alpha_and_learning_state_roundtrip(self):
  s=small().scenarios['balanced'];p=ActorCriticPolicy('fixed',s,learning=dict(initial_alpha=10.,normalize_advantage=True))
  obs=torch.zeros(p.input_size);self.assertTrue(torch.allclose(p.distribution(obs).concentration,torch.full((5,),10.)))
  p.advantage_second_moment=.25;q=ActorCriticPolicy.restore(p.to_dict());self.assertEqual(q.advantage_second_moment,.25)
 def test_paired_equal_returns_do_not_move_actor(self):
  s=small().scenarios['balanced'];p=ActorCriticPolicy('fixed',s,learning=dict(fixed_candidates=2,paired_baseline=True))
  obs=torch.zeros(p.input_size);before=copy.deepcopy(p.actor.state_dict())
  p.actor_baselines=[.25,.25]
  p._update([(obs,torch.tensor([.2]*5),.25),(obs,torch.tensor([.2]*5),.25)])
  self.assertTrue(all(torch.equal(before[k],v) for k,v in p.actor.state_dict().items()))
 def test_candidate_selection_does_not_use_heldout(self):
  config=small()
  with tempfile.TemporaryDirectory() as root:
   run=Experiment(config,root);case=run.state['cases'][0]
   a,rows=candidate_benchmark(config,case,config.scenarios['balanced'],.5,4,1,1)
   b,more=candidate_benchmark(config,case,config.scenarios['balanced'],.5,4,1,2)
   self.assertEqual(a['selected_weights'],b['selected_weights']);self.assertEqual(len(rows),3);self.assertEqual(len(more),6)
 def test_shared_network_is_frozen_during_evaluation(self):
  import sys
  from pathlib import Path
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
  from shared_policy_comparison import compare
  config=small(algorithm='actor_critic',models=['contextual'])
  with tempfile.TemporaryDirectory() as root:
   run=Experiment(config,root)
   rows,policies=compare(config,run.state['cases'][:2],config.scenarios['balanced'],1,1)
   self.assertEqual(len(rows),18)
   self.assertEqual(policies['shared']['updates'],sum(p['updates'] for p in policies['independent'].values()))
 def test_stable_without_improvement_cannot_stop(self):
  import numpy as np
  config=small(algorithm='actor_critic',continual=True,convergence_window=100,convergence_probe_every=1)
  with tempfile.TemporaryDirectory() as root:
   run=Experiment(config,root);keys=['policy'];w=np.full((1,1,5),.2);scores=np.full((1,3),.5)
   with patch('kiosk_research.convergence.fixed_probes',return_value=(keys,w,scores)):
    for count in range(1,102):
     # Map training count to daily cycle with two validation/evaluation dates.
     run.state['day']=(count-1)*3+1
     result=observe(run)
   self.assertEqual(result['status'],'stable_without_improvement');self.assertFalse(result['converged']);self.assertTrue(result['functionally_stable'])

if __name__=='__main__':unittest.main()
