from dataclasses import dataclass
import math
import random

COMPONENTS=('affinity','waste_risk','margin','wait_utility','affordability')
CONTEXT=('bias','taste_1','taste_2','taste_3','price_sensitivity','wait_sensitivity','dessert_visit','stock_pressure','congestion')

def context(c,store,now):
    return (1.,*(t-.5 for t in c.taste),c.price_sensitivity-.7,c.wait_sensitivity-.6,
            1. if c.purpose=='dessert' else -1. if c.purpose=='meal' else 0.,
            1.-min(1.,sum(store.stock.values())/max(1.,40.*len(store.stock))),
            min(1.,max(0.,min(store.cook_free_at)-now)/30.))

@dataclass
class Policy:
    kind: str
    theta: list

    @classmethod
    def initial(cls,kind):
        if kind=='no_recommendation':return cls(kind,[])
        if kind not in ('fixed','contextual'): raise ValueError('unknown model')
        return cls(kind,[[0.]*(3 if kind=='fixed' else 3*len(CONTEXT)) for _ in COMPONENTS])

    def weights(self,c,store,now,lam):
        if not 0<=lam<=1: raise ValueError('invalid lambda')
        basis=(1.,2*lam-1.,(2*lam-1.)**2)
        features=tuple(x*b for x in ((1.,) if self.kind=='fixed' else context(c,store,now)) for b in basis)
        logits=[sum(a*b for a,b in zip(row,features)) for row in self.theta]
        raw=[math.exp(v-max(logits)) for v in logits]
        return tuple(v/sum(raw) for v in raw)

    def perturb(self,seed,sigma):
        rng=random.Random(seed);step=sigma/math.sqrt(sum(map(len,self.theta)))
        return Policy(self.kind,[[max(-8.,min(8.,v+rng.gauss(0.,step))) for v in row] for row in self.theta])

    def to_dict(self):return {'kind':self.kind,'theta':self.theta}

    @classmethod
    def restore(cls,state):
        if state.get('kind')=='no_recommendation':
            if state.get('theta')!=[]:raise ValueError('baseline must have no learned parameters')
            return cls.initial('no_recommendation')
        if state.get('algorithm')=='actor_critic':
            from kiosk_research.actor_critic import ActorCriticPolicy
            return ActorCriticPolicy.restore(state)
        policy=cls(state['kind'],state['theta']);expected=cls.initial(policy.kind)
        if len(policy.theta)!=len(COMPONENTS) or any(len(a)!=len(b) or any(not math.isfinite(v) for v in a) for a,b in zip(policy.theta,expected.theta)):raise ValueError('invalid policy state')
        return policy


def recommendation_features(customer, store, menu, arrival, now, wait_scale, risks, risk_scale, margin_scale, decision_minutes):
    """Only information available before choice; fixed menu reference scales.

    Risk is the marginal weighted stock consumed, not the remaining stock level.
    Wait predicts accepting the suggestion, including the kiosk queue/decision.
    """
    from kiosk_sim.recommenders import affinity, predicted_wait
    from kiosk_research.evaluation import affordability, wait_utility
    predicted = now - arrival + decision_minutes + predicted_wait(store, menu, now + decision_minutes)
    return (affinity(customer, menu),
            risks[menu.ingredient] * menu.units / risk_scale if risk_scale > 0 else 0.,
            max(0., menu.price-menu.cost) / margin_scale if margin_scale > 0 else 0.,
            wait_utility(predicted, wait_scale), affordability(menu.price, customer.budget))
