from .models import Customer, Menu, Store

POLICIES = ("none", "popular", "customer", "fixed", "gemini")
FIXED_WEIGHTS = (.25, .25, .25, .25)  # neutral starting point, not a calibrated optimum


def purpose_fit(customer: Customer, menu: Menu) -> float:
    return (menu.meal_fit if customer.purpose == "meal" else
            menu.dessert_fit if customer.purpose == "dessert" else
            (menu.meal_fit + menu.dessert_fit) / 2)


def affinity(customer: Customer, menu: Menu) -> float:
    taste_match = max(0., 1. - sum(abs(a - b) for a, b in zip(customer.taste, menu.taste)) / 3.)
    return .7 * taste_match + .3 * purpose_fit(customer, menu)


def available(store: Store, customer: Customer) -> list[Menu]:
    return [m for m in store.menus if store.stock[m.ingredient] >= m.units
            and m.price <= customer.budget
            and (not customer.allergy or customer.allergy != m.allergen)]


def predicted_wait(store: Store, menu: Menu, arrival: float) -> float:
    return max(0., min(store.cook_free_at) - arrival) + menu.cook_minutes


def rank(customer: Customer, store: Store, arrival: float, policy: str,
         weights: tuple[float, ...], sales: dict[str, int]) -> Menu | None:
    options = available(store, customer)
    if not options or policy == "none":
        return None
    if hasattr(weights, "choose"):
        return weights.choose(customer, store, arrival, options)
    if policy == "popular":
        return max(options, key=lambda m: (sales[m.name], m.name))
    if policy == "customer":
        return max(options, key=lambda m: affinity(customer, m) -
                   .12 * customer.price_sensitivity * m.price / 20. -
                   .10 * customer.wait_sensitivity * predicted_wait(store, m, arrival) / 30.)
    if policy not in ("fixed", "gemini"):
        raise ValueError(policy)
    if callable(weights):
        weights = weights(customer, store, arrival)
    a, b, c, d = weights
    return max(options, key=lambda m: a * affinity(customer, m) +
               b * m.expiry_risk * min(1., store.stock[m.ingredient] / 40.) +
               c * (m.price - m.cost) / 15. -
               d * min(1., predicted_wait(store, m, arrival) / 30.))
