"""CNN product recognition at checkout (MobileNetV2, 17 retail supercategories).

The CNN is a factual optical sensor: it reports which product categories it sees and how
confident it is. It never decides whether a customer is trustworthy; that is the job of the
deterministic decision model in ``decision.py``.

How the CNN drives the checkout (camera mode, the default):
  * every store item has a product category in the simulated catalogue (``catalogue_category``);
  * the camera photographs each product placed on the tray (``camera_photos``; the demo uses the
    verified test crop of the item's category, and the operator can swap a photo to simulate fraud);
  * the CNN classifies every photo, and ``verify_basket`` keeps a declared item as *presented* only
    if a confident photo of its category was seen. Only those CNN-verified items feed the decision
    model (``items_picked_not_presented`` and ``cart_total``).

Manual entry (item ids only, no photo) remains as a fallback when the CNN is unavailable.

Important implementation detail: the exported Keras model already contains the
``mobilenet_v2.preprocess_input`` layers (``true_divide`` / ``subtract``) as part of its graph,
because it was built with ``x = preprocess_input(inputs)`` in the training notebook. We therefore
feed raw RGB pixels in ``[0, 255]`` and must **not** apply ``preprocess_input`` a second time.
"""
import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # hide TensorFlow start-up logs

import numpy as np  # noqa: E402
from tensorflow import keras  # noqa: E402

from .config import CNN_DIR, ROOT  # noqa: E402

AVAILABLE = True
MIN_CONFIDENCE = 0.60       # below this softmax score a photo does not validate any item

_MODEL_LOCK = threading.Lock()      # Keras inference is serialised: agents may run in parallel


@dataclass(frozen=True)
class Prediction:
    """One product recognised in the checkout image, with the CNN's calibrated softmax score."""
    label: str              # one of the 17 retail supercategories
    confidence: float       # max softmax probability, in [0, 1]
    source: str             # image path (or manual category name)


@dataclass(frozen=True)
class CheckoutView:
    item_ids: tuple[int, ...]          # items placed at checkout (basket ids, used by the decision)
    confidence: float | None           # mean recognition confidence; None for manual entry
    source: str                        # "cnn" or "manual"
    labels: tuple[str, ...] = ()       # recognised (or manually supplied) retail categories
    predictions: tuple[Prediction, ...] = ()   # per-image detail, for the CNN agent / UI
    declared_ids: tuple[int, ...] = ()  # items the customer placed on the tray (before CNN check)
    rejected_ids: tuple[int, ...] = ()  # declared items the CNN could not confirm


@lru_cache(maxsize=1)
def metadata() -> dict:
    with open(CNN_DIR / "cnn_meta.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def categories() -> tuple[str, ...]:
    """The 17 supercategories, in the exact class-index order used by the model."""
    return tuple(metadata()["supercategories"])


@lru_cache(maxsize=1)
def sample_index() -> tuple[dict, ...]:
    """Verified test crops shipped with the model, one per supercategory."""
    with open(CNN_DIR / "sample_crops" / "index.json") as f:
        return tuple(json.load(f))


@lru_cache(maxsize=1)
def load_model():
    """Load the trained Keras model once per process."""
    return keras.models.load_model(CNN_DIR / "cnn_model.keras", compile=False)


def sample_crops() -> tuple[dict, ...]:
    """Sample crops with absolute paths, ready for the Streamlit selector."""
    return tuple({**entry, "path": str(CNN_DIR / "sample_crops" / entry["file"])}
                 for entry in sample_index())


def _resolve_image(path_or_category: str | Path) -> Path:
    """Accept an absolute path, a path relative to the project, a crop filename or a category name."""
    raw = Path(path_or_category)
    candidates = [raw]
    if not raw.is_absolute():
        candidates += [ROOT / raw, CNN_DIR / "sample_crops" / raw]
    if raw.suffix == "":
        candidates += [CNN_DIR / "sample_crops" / f"{raw.name}.jpg", ROOT / f"{raw}.jpg"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image de caisse introuvable : {path_or_category}")


def _load_image(path: str | Path) -> np.ndarray:
    """RGB image resized to the model input size, raw [0, 255] values (preprocessing is in-graph)."""
    resolved = _resolve_image(path)
    size = int(metadata()["img_size"])
    image = keras.utils.load_img(resolved, target_size=(size, size), color_mode="rgb")
    array = keras.utils.img_to_array(image, dtype="float32")
    if array.shape != (size, size, 3):
        raise ValueError(f"Forme d'image inattendue : {array.shape}")
    return array


def predict_image(path_or_category: str | Path) -> Prediction:
    """Classify one checkout crop. Returns the top supercategory and its softmax confidence."""
    array = _load_image(path_or_category)
    with _MODEL_LOCK:
        probabilities = np.asarray(
            load_model()(np.expand_dims(array, axis=0), training=False), dtype="float32"
        ).ravel()
    class_id = int(probabilities.argmax())
    return Prediction(label=categories()[class_id], confidence=float(probabilities[class_id]),
                      source=str(path_or_category))


def predict_images(paths: list[str | Path]) -> tuple[Prediction, ...]:
    """Classify several checkout crops in one batch, preserving their order."""
    if not paths:
        return ()
    batch = np.stack([_load_image(path) for path in paths])
    with _MODEL_LOCK:
        probabilities = np.asarray(load_model()(batch, training=False), dtype="float32")
    labels = categories()
    return tuple(Prediction(label=labels[int(row.argmax())], confidence=float(row.max()),
                            source=str(path))
                 for path, row in zip(paths, probabilities))


def catalogue_category(item_id: int) -> str:
    """Product category of a store item in the simulated catalogue.

    RetailRocket item ids carry no product category, so the demo store assigns each item one of
    the CNN's 17 supercategories with a fixed, documented rule (id modulo 17).
    """
    labels = categories()
    return labels[int(item_id) % len(labels)]


def camera_photos(item_ids: list[int] | tuple[int, ...],
                  swaps: dict[int, str] | None = None) -> list[str]:
    """Simulated camera: one photo per product on the tray (the crop of its catalogue category).

    ``swaps`` maps an item id to another category, to simulate a customer who declares one product
    but places a different one on the tray (substitution fraud).
    """
    swaps = swaps or {}
    return [swaps.get(int(i), catalogue_category(i)) for i in item_ids]


def verify_basket(predictions: tuple[Prediction, ...], declared_ids: tuple[int, ...],
                  min_confidence: float = MIN_CONFIDENCE) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Match each confident photo to one declared item of the same category.

    Returns ``(verified_ids, rejected_ids)``. A declared item is verified only when the CNN saw,
    with at least ``min_confidence``, a product of its catalogue category; each photo validates at
    most one item.
    """
    remaining = [p.label for p in predictions if p.confidence >= min_confidence]
    verified, rejected = [], []
    for item_id in declared_ids:
        category = catalogue_category(item_id)
        if category in remaining:
            remaining.remove(category)
            verified.append(item_id)
        else:
            rejected.append(item_id)
    return tuple(verified), tuple(rejected)


def items_seen(manual_item_ids: list[int] | tuple[int, ...],
               image_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
               manual_categories: list[str] | tuple[str, ...] | None = None) -> CheckoutView:
    """Build the checkout view consumed by the decision model.

    Args:
        manual_item_ids: item ids the customer placed on the checkout tray (declared).
        image_paths: checkout photos of the tray. When supplied, the CNN classifies them and only
            the declared items it confirms (``verify_basket``) are kept in ``item_ids``: the CNN
            then decides which items count as presented for the decision model.
        manual_categories: optional category names supplied by the operator when no image is
            available (manual fallback). They are informational only.
    """
    item_ids = tuple(int(i) for i in manual_item_ids)

    if image_paths:
        predictions = predict_images(list(image_paths))
        mean_confidence = float(np.mean([p.confidence for p in predictions])) if predictions else None
        verified, rejected = verify_basket(predictions, item_ids)
        return CheckoutView(item_ids=verified, confidence=mean_confidence, source="cnn",
                            labels=tuple(p.label for p in predictions), predictions=predictions,
                            declared_ids=item_ids, rejected_ids=rejected)

    labels = tuple(str(c) for c in manual_categories) if manual_categories else ()
    return CheckoutView(item_ids=item_ids, confidence=None, source="manual", labels=labels,
                        declared_ids=item_ids)


def vision_status(view: CheckoutView) -> str:
    """Short French status line for the staff UI and the agents' facts."""
    if view.source == "cnn" and view.confidence is not None:
        seen = ", ".join(dict.fromkeys(view.labels)) or "aucune catégorie"
        text = (f"CNN — {len(view.item_ids)}/{len(view.declared_ids)} article(s) confirmé(s) par la "
                f"caméra, confiance moyenne {view.confidence:.0%} ; catégories vues : {seen}")
        if view.rejected_ids:
            text += f" ; {len(view.rejected_ids)} article(s) non reconnu(s) sur la photo"
        return text
    return ("Saisie manuelle : aucune photo de caisse analysée"
            + (f" — catégories déclarées : {', '.join(view.labels)}" if view.labels else ""))
