"""Training-only candidate selection, followed by untouched paired evaluation."""
import copy
import random
from kiosk_research.episodes import run_episode,seed_for

class ConstantWeights:
    kind='fixed'
    def __init__(self,weights):self.value=tuple(weights)
    def weights(self,*args):return self.value


def candidate_benchmark(config,case,scenario,lam,candidates=32,train_samples=3,test_samples=3):
    from kiosk_research.policy import Policy
    if candidates<2 or min(train_samples,test_samples)<1:raise ValueError('invalid benchmark budget')
    rng=random.Random(seed_for(config.seed,'candidate_search',case['id'],lam))
    vectors=[(.2,)*5]+[tuple(float(i==j) for j in range(5)) for i in range(min(5,candidates-1))]
    while len(vectors)<candidates:
        values=[rng.expovariate(1.) for _ in range(5)];vectors.append(tuple(x/sum(values) for x in values))
    values=[sum(run_episode(config,case,ConstantWeights(w),lam,scenario,s,'candidate_train')['reward'] for s in range(1,train_samples+1))/train_samples for w in vectors]
    best=max(range(len(values)),key=values.__getitem__)
    rows=[]
    for sample in range(1,test_samples+1):
        for arm,policy in [('no_recommendation',Policy.initial('no_recommendation')),('uniform',ConstantWeights(vectors[0])),('selected_candidate',ConstantWeights(vectors[best]))]:
            result=run_episode(config,case,policy,lam,scenario,sample,'candidate_heldout')
            rows.append(dict(store_id=case['id'],lambda_value=lam,sample=sample,arm=arm,customer=result['customer'],store=result['store'],reward=result['reward']))
    return dict(selected_weights=vectors[best],training_objective=values[best],candidates=candidates,training_calls=candidates*train_samples,
                protocol='selection uses candidate_train only; candidate_heldout never selects weights; not a global upper bound'),rows
