"""Turn a visit's raw events into facts people can read: items, totals, pauses, theft cues."""
from dataclasses import dataclass

ACTION_LABELS = {"LOOK": "Regarde", "PICK": "Prend", "PAY": "Paie"}


@dataclass(frozen=True)
class Item:
    """One store item touched during the visit (id, aisle, price)."""
    item_id: int
    aisle: int
    price: float

    @property
    def label(self) -> str:
        """French label shown to staff, e.g. ``Article 42 (rayon 3, 9.90 €)``."""
        aisle = f"rayon {self.aisle}" if self.aisle else "rayon inconnu"
        return f"Article {self.item_id} ({aisle}, {self.price:.2f} €)"


@dataclass(frozen=True)
class VisitSummary:
    """Readable summary of a visit: items picked and paid, duration, longest pause, theft cues."""
    items: dict[int, Item]          # every item touched, by id
    picked: tuple[int, ...]         # items put in the basket, in order, without duplicates
    paid: tuple[int, ...]           # one entry per payment line
    duration_min: float
    longest_pause_min: float
    cues: tuple[str, ...]           # theft cues found in the path, in plain French


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as French text, e.g. ``2 min 05 s``."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes} min {rest:02d} s" if rest else f"{minutes} min"


def _theft_cues(events: list[dict], thresholds: dict) -> list[str]:
    """The three behaviours the theft patterns were built on, when they occur in this path."""
    cues = []
    for prev, cur in zip(events, events[1:]):
        if (prev["action"] == "PICK" and prev["price"] > thresholds["price_expensive"]
                and cur["dt_sec"] >= thresholds["dt_long_lo"]):
            cues.append(f"Arrêt de {format_duration(cur['dt_sec'])} juste après avoir pris "
                        f"un article cher ({prev['price']:.2f} €)")

    run = best = 1
    for prev, cur in zip(events, events[1:]):
        fast_pick = prev["action"] == "PICK" and cur["action"] == "PICK" and cur["dt_sec"] <= thresholds["dt_fast"]
        run = run + 1 if fast_pick else 1
        best = max(best, run)
    if best >= 3:
        cues.append(f"{best} articles pris en quelques secondes")

    expensive = [e for e in events if e["action"] == "PICK" and e["price"] > thresholds["price_expensive"]]
    paid = [e["price"] for e in events if e["action"] == "PAY"]
    if expensive and paid and max(paid) <= thresholds["price_cheap"]:
        cues.append(f"Article cher pris ({expensive[0]['price']:.2f} €) mais seuls de petits articles payés "
                    f"({sum(paid):.2f} €)")
    return cues


def summarize(events: list[dict], thresholds: dict) -> VisitSummary:
    """Build the ``VisitSummary`` of a visit from its raw LOOK / PICK / PAY events."""
    items = {e["itemid"]: Item(e["itemid"], e["aisle"], e["price"]) for e in events}
    picked = tuple(dict.fromkeys(e["itemid"] for e in events if e["action"] == "PICK"))
    paid = tuple(e["itemid"] for e in events if e["action"] == "PAY")
    pauses = [e["dt_sec"] for e in events[1:]] or [0.0]
    return VisitSummary(items=items, picked=picked, paid=paid,
                        duration_min=sum(e["dt_sec"] for e in events) / 60,
                        longest_pause_min=max(pauses) / 60,
                        cues=tuple(_theft_cues(events, thresholds)))


def timeline(events: list[dict]) -> list[str]:
    """One readable line per event, for the agent."""
    lines, elapsed = [], 0.0
    for e in events:
        elapsed += e["dt_sec"]
        item = Item(e["itemid"], e["aisle"], e["price"])
        lines.append(f"{format_duration(elapsed)} : {ACTION_LABELS[e['action']]} {item.label}")
    return lines
