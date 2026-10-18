from dataclasses import dataclass


@dataclass
class Result:
    satisfaction: float
    profit: float
    avg_wait: float
    served: int
    accept_rate: float
    stockouts: int
    waste: float
    avg_decision: float = 0.
    avg_kiosk_queue: float = 0.
    avg_total_wait: float = 0.
    store_satisfaction: float = 0.
    score: float = 0.


def score(result: Result, customers: int) -> float:
    """Hypothesized objective. Report components and vary its weights in sensitivity runs."""
    return (.45 * result.satisfaction + .35 * result.store_satisfaction
            + .10 * result.served / max(1, customers)
            - .10 * result.avg_total_wait / (result.avg_total_wait + 25.))
