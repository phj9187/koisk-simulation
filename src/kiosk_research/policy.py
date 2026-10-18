from dataclasses import dataclass
import math
import random

COMPONENTS=('affinity','waste_risk','margin','negative_wait')
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
        if kind not in ('fixed','contextual'): raise ValueError('unknown model')
        return cls(kind,[[0.]*(3 if kind=='fixed' else 3*len(CONTEXT)) for _ in range(4)])

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
        policy=cls(state['kind'],state['theta']);expected=cls.initial(policy.kind)
        if len(policy.theta)!=4 or any(len(a)!=len(b) or any(not math.isfinite(v) for v in a) for a,b in zip(policy.theta,expected.theta)):raise ValueError('invalid policy state')
        return policy
