"""Bounded customer information. All coefficients are hypotheses, not estimates.

Customer perceptions, AI predictions and realized satisfaction are separate.
Random streams are keyed by customer and menu, never by the recommending arm.
"""
import hashlib
import math
import random
import numpy as np
from kiosk_sim.recommenders import affinity,purpose_fit,predicted_wait


def keyed_rng(seed,person,channel,menu=''):
    value=int.from_bytes(hashlib.sha256(repr((seed,person,channel,menu)).encode()).digest()[:8],'big')
    return random.Random(value)


def feasibility(store,customer):
    """Failure masks retain overlapping constraints rather than assuming budget."""
    masks={}
    for menu in store.menus:
        mask=(1 if store.stock.get(menu.ingredient,0)<menu.units else 0)
        mask|=2 if menu.price>customer.budget else 0
        mask|=4 if customer.allergy and customer.allergy==menu.allergen else 0
        masks[str(mask)]=masks.get(str(mask),0)+1
    return {'menu_failure_masks':masks,'stock_bit':1,'budget_bit':2,'allergy_bit':4,
            'feasible_menus':masks.get('0',0),'budget':customer.budget,
            'affordable_menus':sum(n for mask,n in masks.items() if not int(mask)&2),
            'stock_available_menus':sum(n for mask,n in masks.items() if not int(mask)&1),
            'allergy_safe_menus':sum(n for mask,n in masks.items() if not int(mask)&4)}


class CustomerDecision:
    def __init__(self,customer,store,options,settings,seed,person,shocks,now,visible_queue=0.,regular=False,trust=0.):
        self.customer,self.store,self.options,self.settings=customer,store,options,settings
        self.seed,self.person,self.shocks,self.now=seed,person,shocks,now
        self.visible_queue,self.trust=visible_queue,trust
        self.sigma=settings.regular_taste_noise if regular and settings.regular_taste_noise is not None else settings.taste_noise
        self.considered=self._consideration()

    def _consideration(self):
        if self.settings.consideration_mean is None:return list(self.options)
        rng=keyed_rng(self.seed,self.person,'consideration_count')
        k=min(len(self.options),1+int(np.random.default_rng(rng.getrandbits(64)).poisson(self.settings.consideration_mean)))
        order={m.name:i for i,m in enumerate(self.store.menus)}
        def priority(m):
            position=1.-order[m.name]/max(1,len(order)-1)
            log_weight=(self.settings.salience_position*position+self.settings.salience_category*purpose_fit(self.customer,m)
                        +self.settings.salience_popularity.get(m.name,0.))
            weight=math.exp(max(-40.,min(40.,log_weight)))
            u=max(1e-15,keyed_rng(self.seed,self.person,'consideration_priority',m.name).random())
            return math.log(u)/weight
        return sorted(self.options,key=priority,reverse=True)[:k]

    def perceived_taste(self,menu,ai=False):
        sigma=self.settings.ai_taste_noise if ai else self.sigma
        value=affinity(self.customer,menu)
        if sigma:value+=keyed_rng(self.seed,self.person,'ai_taste' if ai else 'customer_taste',menu.name).gauss(0.,sigma)
        return max(0.,min(1.,value))

    def perceived_wait(self,menu,now=None):
        now=self.now if now is None else now
        if self.settings.wait_visibility=='full':return predicted_wait(self.store,menu,now)
        delay=0.
        if self.settings.wait_visibility=='visible_queue':
            average=sum(m.cook_minutes for m in self.store.menus)/max(1,len(self.store.menus))
            delay=self.visible_queue*average/max(1,self.store.cooks)
        estimate=menu.cook_minutes+delay
        if self.settings.wait_noise:
            sigma=math.sqrt(math.log1p(self.settings.wait_noise**2))
            estimate*=keyed_rng(self.seed,self.person,'customer_wait',menu.name).lognormvariate(-.5*sigma*sigma,sigma)
        return estimate

    def utility(self,menu):
        c=self.customer
        return 2.5*self.perceived_taste(menu)-c.price_sensitivity*menu.price/20.-c.wait_sensitivity*self.perceived_wait(menu)/30.

    def noisy_utility(self,menu):
        return self.utility(menu)+self.settings.choice_noise*math.log(max(1e-9,self.shocks[menu.name]))

    def choose(self,suggestions,u):
        own=max(self.considered,key=self.noisy_utility)
        suggestion=max(suggestions,key=self.noisy_utility) if suggestions else None
        # In a stopping search, acceptance cannot use the best unseen menu.
        reference=self.considered[0] if self.settings.decision_rule=='inspected' else own
        gain=self.utility(suggestion)-self.utility(reference) if suggestion else 0.
        probability=0. if suggestion is None else min(1.,self.settings.acceptance_multiplier*self.customer.openness/
            (1.+math.exp(-max(-50.,min(50.,3.*(gain+.25+self.trust))))))
        accepted=u<probability
        inspected={m.name for m in (self.considered[:1] if accepted and self.settings.decision_rule=='inspected' else self.considered)}
        inspected.update(m.name for m in suggestions)
        return {'own':own,'suggestion':suggestion,'picked':suggestion if accepted else own,'accepted':accepted,
                'probability':probability,'inspected_count':len(inspected),'acceptance_reference':reference.name}

    def decision_minutes(self,choice,no_recommendation,legacy_mode):
        s=self.settings
        if s.decision_rule=='inspected':value=(s.decision_base+s.decision_per_item*choice['inspected_count'])*s.decision_multiplier
        else:
            value=(1.3+.45*len(self.options))*s.decision_multiplier
            if legacy_mode!='equal' and choice['accepted']:value=(.8+.10*len(self.options))*s.decision_multiplier
        if not no_recommendation:value+=s.decision_penalty*len(choice.get('suggestions',[]))
        return value

    def fatigue(self,choice):return self.settings.overload_penalty*math.log1p(choice['inspected_count'])

    def update_trust(self,choice,realized_fit):
        if not self.settings.trust_learning or not choice['accepted']:return self.trust
        error=realized_fit-self.perceived_taste(choice['picked'])
        return max(-1.,min(1.,(1-self.settings.trust_rate)*self.trust+self.settings.trust_rate*error))
