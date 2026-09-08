"""
Farmer-facing text for each class the CNN can predict.

Written for smallholder conditions in Ghana: cultural and mechanical control
first, extension officer for confirmation, no pesticide brand names.
"""

from __future__ import annotations

from model import UNCLEAR_LABEL

_ANALYSIS: dict[str, str] = {
    "Cassava bacterial blight": (
        "The leaf shows the pattern the model associates with cassava bacterial blight: "
        "angular water-soaked spots between the veins that dry into brown patches, often "
        "with leaf wilting and gum on the stem. It spreads fastest in the rainy season and "
        "through cutting tools and infected planting material."
    ),
    "Cassava brown streak disease": (
        "The leaf matches cassava brown streak disease: yellow blotches or chlorosis along "
        "the secondary veins while the rest of the leaf stays green. Leaves can look mild "
        "while the roots below develop dry brown rot, so check a root before deciding the "
        "plant is fine."
    ),
    "Cassava green mottle": (
        "The leaf matches cassava green mottle: pale green mottling, puckered or distorted "
        "leaf edges, and stunted young shoots. Plants often grow out of the visible symptoms, "
        "but the affected stems should not be used as planting material."
    ),
    "Cassava mosaic disease": (
        "The leaf matches cassava mosaic disease: a yellow-and-green mosaic pattern with "
        "twisted, reduced leaflets. It is carried by whiteflies and, more commonly, by "
        "planting cuttings taken from an already infected plant."
    ),
    "Healthy": (
        "No disease pattern was detected. The leaf reads as uniformly green with a normal "
        "shape and no mosaic, streaking, or blight lesions."
    ),
    UNCLEAR_LABEL: (
        "The model could not settle on a class with enough confidence. The photo may not be "
        "a cassava leaf, or it may be blurred, too dark, or too far away."
    ),
}

_SUGGESTIONS: dict[str, str] = {
    "Cassava bacterial blight": (
        "Remove and burn affected plants — do not compost them. Take cuttings only from "
        "clean fields, and disinfect your cutlass between plants. Avoid working the field "
        "while the leaves are wet, practise crop rotation for at least one season, and ask "
        "your extension officer about blight-tolerant varieties."
    ),
    "Cassava brown streak disease": (
        "Uproot and destroy infected plants and never take cuttings from them. Source "
        "planting material that is certified or from a visibly clean field, control "
        "whiteflies by keeping the field weed-free, and harvest early if roots are already "
        "affected. Report an outbreak to your local extension officer — this disease is "
        "reportable in many districts."
    ),
    "Cassava green mottle": (
        "Rogue out badly affected plants and take cuttings only from symptom-free stems. "
        "Keep the field weeded, avoid moving planting material between farms, and monitor "
        "new shoots — many plants recover as they mature."
    ),
    "Cassava mosaic disease": (
        "Remove and destroy infected plants early in the season, and replant using cuttings "
        "from healthy fields or improved resistant varieties. Control whiteflies by clearing "
        "weeds around the plot and avoid planting new cassava next to an infected field."
    ),
    "Healthy": (
        "Keep up current practice: weed regularly, scout the field weekly for mosaic or "
        "streaking on young leaves, and continue to take cuttings only from clean plants."
    ),
    UNCLEAR_LABEL: (
        "Retake the photo: one cassava leaf filling the frame, held flat, in good daylight "
        "and in focus. If the symptoms are on the stem or roots, take a second photo of "
        "those and show it to your extension officer."
    ),
}


def analysis_for(label: str, confidence: float) -> str:
    base = _ANALYSIS.get(label, _ANALYSIS[UNCLEAR_LABEL])
    return f"{base} Model confidence: {confidence * 100:.1f}%."


def suggestions_for(label: str) -> str:
    return _SUGGESTIONS.get(label, _SUGGESTIONS[UNCLEAR_LABEL])
