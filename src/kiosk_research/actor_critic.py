"""Score-function actor and TD critic; no derivative through the simulator.

Checkpoint payloads contain JSON and compressed tensors, never executable pickle.
"""
import base64
import copy
import zlib
import torch
from torch import nn
from torch.distributions import Dirichlet
from kiosk_research.policy import context

torch.set_num_threads(1)


def pack(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        return {'tensor': base64.b64encode(zlib.compress(bytes(value.untyped_storage()))).decode(),
                'dtype': str(value.dtype).split('.')[-1], 'shape': list(value.shape)}
    if isinstance(value, dict):
        return {'mapping': [[k, pack(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value, dict) and 'tensor' in value:
        types = {'float32': torch.float32, 'float64': torch.float64, 'int64': torch.int64, 'uint8': torch.uint8}
        data = bytearray(zlib.decompress(base64.b64decode(value['tensor'])))
        return torch.frombuffer(data, dtype=types[value['dtype']]).clone().reshape(value['shape'])
    if isinstance(value, dict) and 'mapping' in value:
        return {k: unpack(v) for k, v in value['mapping']}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


class ActorCriticPolicy:
    algorithm = 'actor_critic'
    # Both variants have exactly the same parameter count. Daily inputs contain
    # expected customer features, never the actual future customer list.
    input_size = 25

    def __init__(self, kind, scenario, seed=42, expected_visitors=30, hours=8):
        if kind not in ('fixed', 'contextual'):
            raise ValueError('unknown model')
        self.kind, self.scenario = kind, copy.deepcopy(scenario)
        self.expected_visitors, self.hours = expected_visitors, hours
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed % (2**63-1))
            def network(outputs):
                return nn.Sequential(nn.Linear(self.input_size,16),nn.Tanh(),nn.Linear(16,16),nn.Tanh(),nn.Linear(16,outputs))
            self.actor, self.critic = network(5), network(1)
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
            self.rng = torch.get_rng_state().clone()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.updates = 0
        self.training = False
        self.pending = None
        self.daily_batch = []
        self.reference = None

    def observation(self, c, store, now, lam):
        from kiosk_sim.models import Customer
        if c is None:
            c = Customer((.5,.5,.5),.7,.6,None,.5,30.,'mixed')
        menus = store.menus
        n = max(1,len(menus))
        public = [len(menus)/50.,store.cooks/10.,self.expected_visitors/100.,self.hours/24.,
                  sum(m.price for m in menus)/n/50.,sum(m.price-m.cost for m in menus)/n/50.,
                  sum(m.expiry_risk for m in menus)/n]
        values = [lam,*self.scenario['customer'],*self.scenario['store'],*context(c,store,now),
                  min(now/max(1,self.hours*60),2.),c.openness,c.budget/100.,*public]
        return torch.tensor(values,dtype=torch.float32).clamp(-10,10)

    def begin_episode(self, store, lam, scenario, n, hours):
        self.reference = copy.deepcopy(store)
        self.pending = None
        self.action = None
        self.episode_reward = 0.

    def distribution(self, obs):
        return Dirichlet(torch.nn.functional.softplus(self.actor(obs)) + .5)

    def _update(self, samples):
        if not samples:
            return
        observations = torch.stack([s[0] for s in samples])
        actions = torch.stack([s[1] for s in samples]).detach()
        targets = torch.tensor([s[2] for s in samples],dtype=torch.float32)
        values = self.critic(observations).squeeze(-1)
        advantage = (targets-values).detach()
        actor_loss = -(self.distribution(observations).log_prob(actions)*advantage).mean()
        critic_loss = .5*(targets-values).square().mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),1.)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(),1.)
        self.critic_optimizer.step()
        self.updates += 1

    def weights(self, c, store, now, lam):
        if self.kind == 'fixed':
            obs = self.observation(None,self.reference or store,0.,lam)
        else:
            obs = self.observation(c,store,now,lam)
        if self.training and self.kind == 'contextual' and self.pending is not None:
            old_obs,old_action,reward = self.pending
            with torch.no_grad():
                target = reward + self.critic(obs).item()  # gamma=1, finite day
            self._update([(old_obs,old_action,target)])
            self.pending = None
        if self.kind == 'fixed' and self.training and self.action is not None:
            weights = self.action.double(); weights /= weights.sum()
            return tuple(float(x) for x in weights)
        with torch.no_grad():
            distribution = self.distribution(obs)
            if self.training:
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(self.rng)
                    action = distribution.sample()
                    self.rng = torch.get_rng_state().clone()
            else:
                action = distribution.mean
        # Convert to double and renormalize for the strict simplex probe checks.
        weights = action.double(); weights /= weights.sum()
        if self.training:
            self.last_obs, self.action = obs, action.detach()
        return tuple(float(x) for x in weights)

    def feedback(self, reward):
        self.episode_reward += reward
        if self.training and self.kind == 'contextual':
            self.pending = (self.last_obs,self.action,reward)

    def end_episode(self, reward):
        if not self.training:
            return
        if self.kind == 'contextual' and self.pending is not None:
            self._update([self.pending])  # terminal V=0
            self.pending = None
        elif self.kind == 'fixed' and self.action is not None:
            self.daily_batch.append((self.last_obs,self.action,reward))

    def end_day(self):
        if self.training and self.kind == 'fixed':
            self._update(self.daily_batch)
        self.daily_batch = []

    def to_dict(self):
        return {'algorithm':self.algorithm,'version':1,'kind':self.kind,'scenario':self.scenario,
                'expected_visitors':self.expected_visitors,'hours':self.hours,'updates':self.updates,
                'payload':pack({'actor':self.actor.state_dict(),'critic':self.critic.state_dict(),
                                'actor_optimizer':self.actor_optimizer.state_dict(),
                                'critic_optimizer':self.critic_optimizer.state_dict(),'rng':self.rng})}

    @classmethod
    def restore(cls, state):
        if state.get('algorithm') != cls.algorithm or state.get('version') != 1:
            raise ValueError('incompatible actor-critic checkpoint')
        policy = cls(state['kind'],state['scenario'],expected_visitors=state['expected_visitors'],hours=state['hours'])
        data = unpack(state['payload'])
        for name in ('actor','critic','actor_optimizer','critic_optimizer'):
            getattr(policy,name).load_state_dict(data[name])
        if any(not torch.isfinite(p).all() for net in (policy.actor,policy.critic) for p in net.parameters()):
            raise ValueError('non-finite network')
        policy.rng,policy.updates = data['rng'],state['updates']
        return policy
