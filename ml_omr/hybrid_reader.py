
from ml_omr.inference import classify_batch

def crop_bubble(gray, x, y, radius=16):
    h, w = gray.shape[:2]

    x1 = max(0, int(x - radius))
    y1 = max(0, int(y - radius))
    x2 = min(w, int(x + radius + 1))
    y2 = min(h, int(y + radius + 1))

    return gray[y1:y2, x1:x2]

def scan_answers_ml(
    gray,
    coordinates,
    crop_radius=16,
    filled_confidence=0.70,
    ambiguous_confidence=0.60,
):
    """
    coordinates:
    {
        1: {"A": (x,y), "B": (x,y), "C": (x,y), "D": (x,y)},
        ...
    }

    Returns:
        answers: {1: "A", 2: None, 3: "MULTIPLE", ...}
        debug: per-question probabilities
    """

    answers = {}
    debug = {}

    # Batch ALL bubbles at once for speed.
    batch_crops = []
    batch_map = []

    for q, option_map in coordinates.items():
        for option, (x, y) in option_map.items():
            batch_crops.append(
                crop_bubble(gray, x, y, crop_radius)
            )
            batch_map.append((q, option))

    predictions = classify_batch(batch_crops)

    grouped = {}

    for (q, option), pred in zip(batch_map, predictions):
        grouped.setdefault(q, {})[option] = pred

    for q, option_preds in grouped.items():
        filled = []
        ambiguous = []

        for option, pred in option_preds.items():
            if pred["label"] == "filled" and pred["confidence"] >= filled_confidence:
                filled.append(option)
            elif pred["label"] == "ambiguous" and pred["confidence"] >= ambiguous_confidence:
                ambiguous.append(option)

        if len(filled) == 1:
            answer = filled[0]
            status = "answered"
        elif len(filled) > 1:
            answer = "MULTIPLE"
            status = "multiple"
        elif ambiguous:
            answer = None
            status = "ambiguous"
        else:
            answer = None
            status = "blank"

        answers[q] = answer
        debug[q] = {
            "status": status,
            "filled_options": filled,
            "ambiguous_options": ambiguous,
            "options": option_preds,
        }

    return answers, debug
