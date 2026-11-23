"""CNN product recognition at checkout. NOT TRAINED YET (teammate, RPC dataset).

Stand-in: the demo enters by hand the items placed at checkout.

To plug in the trained CNN: set AVAILABLE = True and make `items_seen` take the checkout photo
and return the recognised item ids with their mean confidence. Keep the CheckoutView return type
and nothing else in the app has to change.
"""
from dataclasses import dataclass

AVAILABLE = False


@dataclass(frozen=True)
class CheckoutView:
    item_ids: tuple[int, ...]      # items placed at checkout
    confidence: float | None       # mean recognition confidence; None for manual entry
    source: str                    # "manual" until the CNN exists, then "cnn"


def items_seen(manual_item_ids: list[int]) -> CheckoutView:
    return CheckoutView(item_ids=tuple(manual_item_ids), confidence=None, source="manual")
