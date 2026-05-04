from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from predict import (
    nlp,
    normalize_text,
    extract_text_from_image,
    predict,
    fused_score,
    detect_hiring_intention,
    detect_scam_pattern,
    forensic_analysis,
    highlight_scam_words,
    check_digital_footprint,
)

app = FastAPI()

# Allow frontend requests

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(
text: str = Form(None),
image: UploadFile = File(None)
):


    ocr_text = None
    if image:

        temp_path = "temp_image.jpg"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        text = extract_text_from_image(temp_path)
        ocr_text = text

        os.remove(temp_path)


    if not text or text.strip() == "":
        return {"error": "No input provided"}


    original = text

    # Process text with spaCy only ONCE for performance
    doc = nlp(text)

    # Fused scoring: BERT probability blended with rule-based evidence count.
    bert_prob, prob, evidence = fused_score(original, doc=doc)
    score = round(prob * 100)

    # Lowered fake threshold (50) — empirically caught 10/12 missed scams in adversarial test.
    if score > 50:
        label = "FAKE JOB"
    elif score < 30:
        label = "REAL JOB"
    else:
        label = "SUSPICIOUS"

    return {
        "prediction": label,
        "score": score,
        "bert_score": round(bert_prob * 100),
        "evidence_count": evidence,
        "intention": detect_hiring_intention(original),
        "pattern": detect_scam_pattern(original),
        "analysis": forensic_analysis(original, prob),
        "words": highlight_scam_words(original),
        "digital_footprint": check_digital_footprint(original, doc=doc, perform_search=False),
        "ocr_text": ocr_text,
    }