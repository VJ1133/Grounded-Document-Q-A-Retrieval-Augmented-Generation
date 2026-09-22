"""Generates a small synthetic insurance PDF for local demo/testing.

Not part of the app itself — run once during setup:
    python scripts/make_sample_pdf.py
"""

import pymupdf as fitz

PAGES = [
    (
        "Synthetic Catastrophe Model Overview",
        [
            "This document is a synthetic, illustrative overview of a catastrophe "
            "risk model created for demonstration purposes only. It does not "
            "describe any real vendor model or real insurance product.",
            "The model estimates potential losses from natural catastrophe events "
            "such as hurricanes, earthquakes, and severe convective storms by "
            "combining hazard, exposure, and vulnerability modules.",
            "Exposure characteristics used by the model include building "
            "construction type, occupancy class, year built, replacement cost "
            "value, and geographic location at the address level.",
        ],
    ),
    (
        "Hazard Module Assumptions",
        [
            "The hazard module simulates thousands of stochastic event years to "
            "represent the full range of possible catastrophe activity.",
            "For hurricane risk, the model assumes wind speed decay based on "
            "distance from landfall, terrain roughness, and storm size.",
            "For earthquake risk, the model assumes ground motion attenuation "
            "based on fault distance, soil type, and moment magnitude.",
        ],
    ),
    (
        "Vulnerability and Loss Estimation",
        [
            "Vulnerability functions translate hazard intensity, such as peak "
            "wind gust or peak ground acceleration, into expected damage ratios "
            "for a given construction type.",
            "The model applies these damage ratios to the insured replacement "
            "cost value to estimate gross loss before policy terms are applied.",
            "Policy conditions such as deductibles, limits, and coinsurance are "
            "then applied to estimate the net loss to the insurer.",
        ],
    ),
    (
        "Exposure Guidelines",
        [
            "Exposure data quality directly affects model accuracy. Address-level "
            "geocoding is strongly preferred over ZIP-code-level geocoding.",
            "Construction type should be verified against underwriting "
            "documentation rather than assumed from occupancy class alone.",
            "Replacement cost values should be updated at each renewal to reflect "
            "current construction costs and avoid systematic underinsurance.",
        ],
    ),
    (
        "Model Validation Notes",
        [
            "This synthetic model has not been validated against real historical "
            "loss experience and should not be used for any actual underwriting, "
            "pricing, or reserving decision.",
            "In a real production setting, model validation would compare "
            "modeled average annual loss and exceedance probability curves "
            "against actual historical loss experience where available.",
            "Known limitations of this illustrative model include the absence of "
            "storm surge, demand surge, and post-event loss amplification.",
        ],
    ),
]


def build_pdf(output_path: str = "data/sample_documents/catastrophe_model_overview.pdf"):
    doc = fitz.open()
    for title, paragraphs in PAGES:
        page = doc.new_page(width=612, height=792)  # US Letter
        y = 72
        page.insert_text((72, y), title, fontsize=16, fontname="helv")
        y += 36
        for paragraph in paragraphs:
            rect = fitz.Rect(72, y, 540, 760)
            page.insert_textbox(rect, paragraph, fontsize=11, fontname="helv", lineheight=1.4)
            y += 90
    doc.save(output_path)
    doc.close()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    build_pdf()
