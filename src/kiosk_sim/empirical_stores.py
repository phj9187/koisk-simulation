"""Weighted, industry-stratified whole-row donors; not independent marginal LHS."""
import hashlib
import math
import random
from pathlib import Path

SOURCE_URL = 'https://www.krei.re.kr/foodInfo/board/atchDown.do?no=113723'
SOURCE_SHA256 = '213e3e7bf26d0992a1f1d27f0aecc1d789bbc5560987fdaadd7f9daaff793449'
INDUSTRIES = {1: 'Korean', 2: 'Chinese', 3: 'Japanese', 4: 'Western',
              5: 'Other foreign', 8: 'Bakery', 9: 'Fast food', 10: 'Chicken',
              11: 'Other quick service', 12: 'Takeaway', 17: 'Coffee', 18: 'Other drinks'}


def number(value):
    if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value < 999999999:
        return float(value)
    return None


def load_survey(path):
    import openpyxl
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Survey file differs from the verified 2025 source. Do not silently mix versions.')
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = book['Data'].iter_rows(values_only=True)
        next(rows)
        columns = next(rows)
        donors, excluded = [], {}
        for row in rows:
            r = dict(zip(columns, row))
            industry, visitors, weight = r['a3'], number(r['b24']), number(r['WT'])
            reason = ('outside_target_industries' if industry not in INDUSTRIES else
                      'missing_visitors_or_weight' if visitors is None or not weight else None)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            donors.append(dict(source_row=str(r['rowid']), industry=int(industry),
                industry_label=INDUSTRIES[industry], survey_weight=weight,
                visitor_mean=visitors, seats=number(r['b12']),
                opening_hours=number(r['b9a1']), total_staff=number(r['d1a9']),
                spend_won=number(r['b26']), waste_liters_month=number(r['b28a1']),
                revenue_manwon=number(r['c1a1']), food_expense_manwon=number(r['c1a3']),
                dine_in_revenue_pct=number(r['b20a1']), delivery_revenue_pct=number(r['b20a2']),
                takeaway_revenue_pct=number(r['b20a3']),
                menu_prices_won=[number(r[f'b27b{i}']) for i in (1, 2, 3)
                                 if number(r[f'b27b{i}']) is not None and number(r[f'b27b{i}']) > 0]))
        return donors, excluded
    finally:
        book.close()


def sample_donors(donors, count, seed, strata=8):
    """Hamilton industry quotas + repeated quantile strata on weighted visitor CDF.

    The complete observed tuple is retained, including zeros and extreme values.
    Replacement is intentional for a discrete weighted empirical distribution.
    """
    if count < 1 or strata < 1 or not donors:
        raise ValueError('Positive count/strata and nonempty survey required.')
    rng = random.Random(seed)
    groups = {k: [d for d in donors if d['industry'] == k] for k in sorted({d['industry'] for d in donors})}
    masses = {k: sum(d['survey_weight'] for d in g) for k, g in groups.items()}
    total = sum(masses.values())
    exact = {k: count * v / total for k, v in masses.items()}
    quotas = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (exact[k] - quotas[k], -k), reverse=True)[:count-sum(quotas.values())]:
        quotas[k] += 1
    selected = []
    for k, group in groups.items():
        group = sorted(group, key=lambda d: (d['visitor_mean'], d['source_row']))
        n = quotas[k]
        # Full blocks use eight strata; a smaller last block uses its own n strata.
        for start in range(0, n, strata):
            size = min(strata, n-start)
            for slot in range(size):
                q = (slot + rng.random()) / size
                threshold, cumulative = q * masses[k], 0.
                chosen = group[-1]
                for donor in group:
                    cumulative += donor['survey_weight']
                    if cumulative > threshold:
                        chosen = donor
                        break
                selected.append({**chosen, 'sampling_quantile': q, 'sampling_block_size': size,
                                 'sampling_block': start // strata + 1,
                                 'analysis_weight': masses[k] / total / n})
    rng.shuffle(selected)
    return selected, {'industry_quotas': quotas, 'industry_population_share': {k: v/total for k,v in masses.items()},
                      'unrepresented_industries': [k for k,n in quotas.items() if not n],
                      'unique_donors': len({d['source_row'] for d in selected}),
                      'method': 'industry quotas; weighted visitor-quantile LHS blocks; whole-row donors',
                      'not_full_multivariate_lhs': True}


def diagnostics(rows, weight_key):
    fields = ('visitor_mean', 'seats', 'total_staff', 'spend_won', 'opening_hours')
    result = {'missing': {k: sum(r[k] is None for r in rows) for k in fields}}
    def mean(pairs):
        return sum(v*w for v,w in pairs)/sum(w for _,w in pairs) if pairs else None
    result['weighted_means'] = {k: mean([(r[k],r[weight_key]) for r in rows if r[k] is not None]) for k in fields}
    correlations = {}
    for k in fields[1:]:
        valid = [r for r in rows if r[k] is not None]
        if not valid:
            correlations[k] = None
            continue
        mx = mean([(r['visitor_mean'],r[weight_key]) for r in valid])
        my = mean([(r[k],r[weight_key]) for r in valid])
        xx = sum(r[weight_key]*(r['visitor_mean']-mx)**2 for r in valid)
        yy = sum(r[weight_key]*(r[k]-my)**2 for r in valid)
        xy = sum(r[weight_key]*(r['visitor_mean']-mx)*(r[k]-my) for r in valid)
        correlations[k] = xy/math.sqrt(xx*yy) if xx*yy else None
    result['weighted_pearson_with_visitors'] = correlations
    return result


def empirical_cases(config):
    from .lhs_study import StoreCase, _lhs_values, _make_store
    from .ai.fixed_bayesian import BayesianFixedOptimizer
    from .ai.contextual import OnlineContextualBandit
    from dataclasses import replace
    donors, excluded = load_survey(config.survey_path)
    selected, audit = sample_donors(donors, config.store_count, config.seed, config.lhs_strata)
    rng = random.Random(config.seed)
    # Unobserved mechanisms remain explicitly labelled sensitivity assumptions.
    specs = {'menu_count': (4,12), 'cooks': (1,3), 'price_level': (8,20),
             'cost_ratio': (.35,.65), 'cook_minutes': (5,18), 'expiry_risk': (.1,1),
             'stock_level': (28,70), 'menu_diversity': (.08,.34),
             'taste_center_1': (.2,.8), 'taste_center_2': (.2,.8), 'taste_center_3': (.2,.8)}
    cols = {k: _lhs_values(config.store_count, *v, config.lhs_strata, rng) for k,v in specs.items()}
    cases = []
    for i, donor in enumerate(selected):
        f = {k: v[i] for k,v in cols.items()}
        f.update(donor)
        f['purpose'] = 'dessert' if donor['industry'] in (8,17,18) else 'meal'
        f['zero_visitors_retained'] = donor['visitor_mean'] == 0
        f['cooks'] = min(round(f['cooks']), max(1, int(donor['total_staff'] or 1)))
        f['assumed_fields'] = 'cooks allocation,menu count,cost ratio,cook time,expiry,stock,tastes,customer behavior,arrival shape'
        f['observed_metadata_only'] = 'seats,total staff,spend,waste,expenses,sales channel shares'
        f['menu_price_mapping'] = 'cycle observed representative prices; synthetic menu identities' if donor['menu_prices_won'] else 'assumed price fallback: missing representative prices'
        template = _make_store(i+1, f, rng)
        if donor['menu_prices_won']:
            prices = donor['menu_prices_won']
            template.menus = [replace(m, price=prices[j % len(prices)]/1000,
                                      cost=prices[j % len(prices)]/1000*f['cost_ratio'])
                              for j,m in enumerate(template.menus)]
        cases.append(StoreCase(i+1, f, template, BayesianFixedOptimizer(seed=config.seed+i*37), OnlineContextualBandit()))
    audit.update(source=SOURCE_URL, sha256=SOURCE_SHA256, exclusions=excluded,
                 population_diagnostics=diagnostics(donors, 'survey_weight'),
                 sample_diagnostics=diagnostics(selected, 'analysis_weight'),
                 eligible_rows=len(donors), zero_visitor_rows=sum(d['visitor_mean']==0 for d in donors),
                 assumptions='No empirical daily time series: daily counts fixed to reported mean. Outcomes remain synthetic.',
                 scope='Individual food/drink shops; institutional/mobile catering and alcohol-oriented businesses excluded explicitly.')
    return cases, audit
