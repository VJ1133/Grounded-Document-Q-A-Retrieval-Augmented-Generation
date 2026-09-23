"""Generates small synthetic insurance PDFs for local demo/testing.

Not part of the app itself — run once during setup:
    python scripts/make_sample_pdf.py

Produces two documents on different topics/types, useful for exercising V2's
multi-document search and filtering without needing real documents:
  - catastrophe_model_overview.pdf (document type: "Model Documentation")
  - sample_auto_policy_summary.pdf (document type: "Policy")
"""

import pymupdf as fitz

CATASTROPHE_MODEL_PAGES = [
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

AUTO_POLICY_PAGES = [
    (
        "Sample Auto Policy Summary",
        [
            "This document is a synthetic, illustrative auto insurance policy "
            "summary created for demonstration purposes only. It does not "
            "describe any real policyholder, vehicle, or in-force coverage.",
            "The policy provides liability, collision, and comprehensive "
            "coverage for one private passenger vehicle, effective for a "
            "twelve-month term.",
            "The collision deductible on this sample policy is 500 dollars. "
            "The comprehensive deductible on this sample policy is 250 dollars.",
        ],
    ),
    (
        "Coverage Details",
        [
            "Bodily injury liability limits are 100,000 dollars per person and "
            "300,000 dollars per accident. Property damage liability limit is "
            "50,000 dollars per accident.",
            "Uninsured motorist bodily injury coverage matches the policy's "
            "liability limits unless otherwise specified in the declarations.",
            "Medical payments coverage of 5,000 dollars per person is included "
            "for the named insured and passengers.",
        ],
    ),
    (
        "Exclusions",
        [
            "This sample policy does not cover intentional damage caused by the "
            "policyholder, use of the vehicle for commercial ride-sharing "
            "without an appropriate endorsement, or racing.",
            "Wear and tear, mechanical breakdown, and damage from driving on a "
            "route known to be flooded are also excluded under this sample "
            "policy.",
        ],
    ),
]


def build_pdf(pages, output_path: str):
    doc = fitz.open()
    for title, paragraphs in pages:
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
    build_pdf(CATASTROPHE_MODEL_PAGES, "data/sample_documents/catastrophe_model_overview.pdf")
    build_pdf(AUTO_POLICY_PAGES, "data/sample_documents/sample_auto_policy_summary.pdf")
