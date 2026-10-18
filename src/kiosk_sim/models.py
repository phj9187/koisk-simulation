from dataclasses import dataclass, field


@dataclass(frozen=True)
class Customer:
    taste: tuple[float, float, float]
    price_sensitivity: float
    wait_sensitivity: float
    allergy: str | None
    openness: float
    budget: float
    purpose: str = "mixed"  # purpose of this visit, not a permanent customer type


@dataclass(frozen=True)
class Menu:
    name: str
    taste: tuple[float, float, float]
    price: float
    cost: float
    cook_minutes: float
    ingredient: str
    units: int
    allergen: str | None
    expiry_risk: float
    meal_fit: float = .5
    dessert_fit: float = .5


@dataclass
class Store:
    menus: list[Menu]
    stock: dict[str, int]
    cooks: int
    purpose: str = "mixed"
    cook_free_at: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.cook_free_at:
            self.cook_free_at = [0.0] * self.cooks
