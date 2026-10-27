import copy
import random
import tempfile
import unittest
import pandas as pd
import torch
from kiosk_research.actor_critic import ActorCriticPolicy
from kiosk_research.policy import Policy
from kiosk_research.worker import store_day,restore_store
from kiosk_research.simulation import simulate
from kiosk_research.engine import Experiment
from kiosk_research.convergence import fixed_probes
from kiosk_research.linecharts import export_lines,load_snapshot
from kiosk_sim.generators import customers
from test_pareto_research import small,deterministic

SCENARIO={'customer':[.45,.25,.30],'store':[.75,.25]}


class ActorCriticTests(unittest.TestCase):
    def case(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(algorithm='actor_critic',auto_cleanup=False),root)
            return run.state['cases'][0]

    def test_customer_and_daily_cadence_same_interactions_and_frozen_evaluation(self):
        config=small(algorithm='actor_critic',raw_logs=True,visitors=5)
        case=self.case()
        result=store_day((config,case,None,1))
        self.assertEqual(result['optimizer_steps']['balanced:fixed'],1)
        for model in config.models:
            rows=[r for r in result['rows'] if r['model']==model]
            self.assertTrue(all(r['training_simulator_calls']==1 for r in rows))
            self.assertEqual(len(rows),2)
        served=sum(r['served'] for r in result['rows'] if r['model']=='contextual')
        self.assertEqual(result['optimizer_steps']['balanced:contextual'],served)
        for lam in (0.,1.):
            weights=[x['payload_json'] for x in result['logs'] if x['model']=='fixed' and x['lambda_value']==lam]
            import json
            weights=[json.loads(x)['weights'] for x in weights if json.loads(x)['served']]
            self.assertTrue(all(w==weights[0] for w in weights))
        validation=store_day((config,case,result,2))
        self.assertEqual(result['policies'],validation['policies'])
        evaluation=store_day((config,case,validation,3))
        self.assertEqual(result['policies'],evaluation['policies'])

    def test_reward_telescopes_both_networks_update_and_resume_exactly(self):
        case=self.case();shop=restore_store(case['template'])
        people=customers(6,random.Random(12),shop)
        for model in ('fixed','contextual'):
            p=ActorCriticPolicy(model,SCENARIO,123,6)
            before=copy.deepcopy(p.to_dict())
            p.training=True
            result=simulate(people,shop,p,.5,SCENARIO,84)
            self.assertAlmostEqual(p.episode_reward,result['reward'],places=12)
            p.end_day()
            restored=ActorCriticPolicy.restore(p.to_dict())
            self.assertNotEqual(before['payload'],p.to_dict()['payload'])
            original=ActorCriticPolicy.restore(before)
            for name in ('actor','critic'):
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(getattr(p,name).parameters(),getattr(original,name).parameters())))
            restored.training=True
            a=simulate(people,shop,p,.7,SCENARIO,90,raw=True);p.end_day()
            b=simulate(people,shop,restored,.7,SCENARIO,90,raw=True);restored.end_day()
            self.assertEqual(a,b)
            self.assertEqual(p.to_dict(),restored.to_dict())

    def test_save_resume_charts_probes_and_absolute_limit(self):
        config=small(algorithm='actor_critic',continual=True,repetitions=None,auto_cleanup=False)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            first=Experiment(config,a);first.run(target_day=3)
            paused=Experiment(config,b);paused.run(target_day=1)
            resumed=Experiment.resume(b,paused.id);resumed.run(target_day=3)
            pd.testing.assert_frame_equal(deterministic(first.export_metrics()),deterministic(resumed.export_metrics()))
            self.assertEqual(first.state['stores'],resumed.state['stores'])
            _,weights,scores=fixed_probes(resumed,True)
            self.assertEqual(weights.shape[-1],5)
            self.assertTrue(torch.isfinite(torch.tensor(scores)).all())
            export_lines(resumed)
            self.assertEqual(load_snapshot(b,resumed.id).state['day'],3)
            resumed.run(target_day=3);self.assertEqual(resumed.state['day'],3)

    def test_policy_gradient_learns_simple_reward_and_no_global_rng_leak(self):
        p=ActorCriticPolicy('fixed',SCENARIO,9)
        obs=torch.zeros(p.input_size)
        baseline=p.distribution(obs).mean[0].item()
        state=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            for _ in range(300):
                with torch.no_grad():actions=p.distribution(obs).sample((32,))
                p._update([(obs,a,10.*a[0].item()) for a in actions])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertGreater(p.distribution(obs).mean[0].item(),baseline+.05)

    def test_initial_policy_parity_and_unservable_day(self):
        shop=restore_store(self.case()['template'])
        people=customers(5,random.Random(8),shop)
        p=ActorCriticPolicy('fixed',SCENARIO)
        original=simulate(people,shop,Policy.initial('fixed'),.5,SCENARIO,91,raw=True)
        neural=simulate(people,shop,p,.5,SCENARIO,91,raw=True)
        self.assertEqual(original,neural)
        shop.stock={k:0 for k in shop.stock}
        for kind in ('fixed','contextual'):
            p=ActorCriticPolicy(kind,SCENARIO);p.training=True
            before=p.to_dict()
            result=simulate(people,shop,p,.5,SCENARIO,91)
            p.end_day()
            self.assertEqual(result['reward'],0.)
            self.assertEqual(before,p.to_dict())


if __name__=='__main__':unittest.main()
