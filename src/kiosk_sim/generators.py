"""Independent customer and store generators. Values are assumptions, not observations."""
import random
from .models import Customer, Menu, Store

INGREDIENTS = ("base_a", "base_b", "base_c", "base_d")
ALLERGENS = (None, None, None, "nuts", "dairy")


def customers(count: int, rng: random.Random, store: Store | None = None) -> list[Customer]:
    if count < 1:
        raise ValueError("count must be positive")
    purpose_mix = {"meal": (.72, .12, .16), "dessert": (.12, .72, .16),
                   "mixed": (.40, .40, .20)}[store.purpose if store else "mixed"]
    return [Customer(
        tuple(max(0., min(1., rng.gauss(.5, .24))) for _ in range(3)),
        rng.uniform(.2, 1.2), rng.uniform(.1, 1.1),
        rng.choice((None, None, None, "nuts", "dairy")),
        rng.uniform(.25, .9), rng.uniform(8., 22.),
        rng.choices(("meal", "dessert", "mixed"), weights=purpose_mix)[0]
    ) for _ in range(count)]


def stores(count: int, rng: random.Random, menu_count: int = 8) -> list[Store]:
    if count < 1 or menu_count < 2:
        raise ValueError("count must be positive and menu_count >= 2")
    result = []
    for store_id in range(count):
        purpose = rng.choice(("meal", "dessert", "mixed"))
        center = [rng.uniform(.25, .75) for _ in range(3)]
        menus = []
        for j in range(menu_count):
            taste = tuple(max(0., min(1., rng.gauss(x, .22))) for x in center)
            price = round(rng.uniform(8., 20.), 2)
            meal_fit = max(.05, min(.95, rng.gauss(.75 if purpose == "meal" else
                                                    .25 if purpose == "dessert" else .5, .18)))
            dessert_fit = max(.05, min(.95, rng.gauss(1. - meal_fit, .12)))
            menus.append(Menu(f"S{store_id + 1}_M{j + 1}", taste, price,
                              round(price * rng.uniform(.35, .65), 2),
                              rng.uniform(5., 18.), rng.choice(INGREDIENTS),
                              rng.randint(1, 2), rng.choice(ALLERGENS),
                              rng.uniform(.1, 1.), meal_fit, dessert_fit))
        result.append(Store(menus, {k: rng.randint(28, 60) for k in INGREDIENTS},
                            rng.randint(1, 3), purpose))
    return result
