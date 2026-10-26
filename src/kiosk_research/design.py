"""Declared preference designs; geometric coverage, not a population distribution."""
from copy import deepcopy

def screening_design():
    customer = {
        'uniform': [1/3, 1/3, 1/3], 'taste': [1.,0.,0.],
        'price': [0.,1.,0.], 'wait': [0.,0.,1.],
        'taste_price': [.5,.5,0.], 'taste_wait': [.5,0.,.5],
        'price_wait': [0.,.5,.5],
    }
    scenarios = {'customer_'+key: {'customer': value, 'store': [.5,.5]} for key,value in customer.items()}
    scenarios.update(store_profit={'customer': customer['uniform'], 'store':[1.,0.]},
                     store_inventory={'customer': customer['uniform'], 'store':[0.,1.]})
    return deepcopy(scenarios)

def interior_validation_design():
    # Reserved intermediate utility definitions, never passed to the optimizer.
    return {'interior_balanced': {'customer':[.4,.3,.3], 'store':[.6,.4]},
            'interior_price': {'customer':[.2,.6,.2], 'store':[.3,.7]},
            'interior_wait': {'customer':[.2,.2,.6], 'store':[.7,.3]}}
