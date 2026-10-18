"""Grouped survey margins for store means; within-bin and daily shapes are assumptions."""
import math

SOURCE = 'https://www.mafra.go.kr/bbs/home/798/595246/download.do'
PROBABILITIES = (.057, .138, .351, .364, .090)
SURVEY_MEAN = 56.6
BIN_LABELS = ('under_20', '20_to_30', '30_to_50', '50_to_100', '100_plus')
# Uniform bounded bins plus shifted exponential tail, calibrated to the survey mean.
TAIL_SCALE = (SURVEY_MEAN - (.057 * 10 + .138 * 25 + .351 * 40 + .364 * 75)) / .09 - 100


def visitor_quantile(probability):
    if not 0 <= probability < 1:
        raise ValueError('Quantile probability must be in [0, 1).')
    cumulative = 0.
    for index, mass in enumerate(PROBABILITIES):
        if probability < cumulative + mass:
            fraction = (probability - cumulative) / mass
            if index == 4:
                return 100 - TAIL_SCALE * math.log1p(-fraction), BIN_LABELS[index]
            low, high = ((0, 20), (20, 30), (30, 50), (50, 100))[index]
            return max(1e-6, low + (high - low) * fraction), BIN_LABELS[index]
        cumulative += mass
    raise ValueError('Invalid cumulative probabilities')


def sample_store_means(count, strata, rng):
    # Independent replicated 8-point LHS in probability space, then inverse CDF.
    samples = []
    for _ in range(count // strata):
        replicate = [visitor_quantile((index + rng.random()) / strata) for index in range(strata)]
        rng.shuffle(replicate)
        samples.extend(replicate)
    return samples


def daily_visitors(mean, rng):
    """Exact Poisson count using exponential interarrival times; zero days are valid."""
    elapsed, count = rng.expovariate(1.), 0
    while elapsed <= mean:
        count += 1
        elapsed += rng.expovariate(1.)
    return count


def distribution_metadata():
    return {'source': SOURCE, 'survey': '2025 restaurant management survey, table 3-3-1, general restaurants',
            'store_mean_probabilities': dict(zip(BIN_LABELS, PROBABILITIES)),
            'published_mean': SURVEY_MEAN, 'within_bin_assumption': 'uniform, not measured',
            'tail_assumption': '100 + exponential; no factual maximum; scale calibrated to published mean',
            'tail_scale': TAIL_SCALE, 'daily_assumption': 'Poisson conditional on each store mean; not fitted to daily records',
            'sampling': 'independent replicated 8-point probability-space LHS with inverse CDF',
            'correlations': 'store mean sampled independently of other features; empirical joint distribution unavailable'}
