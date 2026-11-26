
import os
import re
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Users\8000392\source\repos\Python\codebert-7.0"
)

PRECISION_THRESHOLD = float(
    os.getenv("PRECISION_THRESHOLD", "0.7")
)

CHUNK_STRIDE = int(
    os.getenv("CHUNK_STRIDE", "128")
)

MAX_SEQ_LENGTH = int(
    os.getenv("MAX_SEQ_LENGTH", "256")
)

# Label mapping from training dataset
# 0 = Human, 1 = AI

LABELS = {
    0: "HUMAN",
    1: "AI"
}

# --------------------------------------------------
# Device configuration
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = None
tokenizer = None

# --------------------------------------------------
# FastAPI initialization
# --------------------------------------------------

app = FastAPI(
    title="CodeBERT AI Code Detection API",
    description="Detect AI-generated and Human-written code",
    version="1.0.0"
)

# --------------------------------------------------
# Text sanitization
# --------------------------------------------------

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

_CTRL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def sanitize_text(text: str) -> str:
    """
    Remove invalid control characters and
    lone UTF-16 surrogate characters.
    """

    text = _CTRL_CHAR_RE.sub("", text)
    text = _SURROGATE_RE.sub("\ufffd", text)

    return text


# --------------------------------------------------
# Model initialization
# --------------------------------------------------

def load_model():

    global model, tokenizer

    print("-------------------------------------")
    print("Loading CodeBERT model")
    print("-------------------------------------")

    print("Model path:", MODEL_PATH)
    print("Device:", device)
    print("Precision threshold:", PRECISION_THRESHOLD)
    print("Max sequence length:", MAX_SEQ_LENGTH)
    print("Chunk stride:", CHUNK_STRIDE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model path not found: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    if model.config.num_labels != 2:
        raise ValueError(
            "Expected a binary classification model with 2 labels."
        )

    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    print("-------------------------------------")


@app.on_event("startup")
def startup_event():
    load_model()


# --------------------------------------------------
# Chunk-based inference
# --------------------------------------------------

def predict_with_chunks(
    code_input,
    max_length=MAX_SEQ_LENGTH,
    stride=CHUNK_STRIDE,
    threshold=PRECISION_THRESHOLD
):
    """
    Tokenize complete source code.
    Split into overlapping chunks.
    Run inference on each chunk.
    Calculate mean AI probability.
    """

    token_ids = tokenizer(
        code_input,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True
    )["input_ids"][0]

    total_tokens = len(token_ids)

    if total_tokens <= max_length:

        chunks = [token_ids]

    else:

        chunks = []

        for start in range(
            0,
            total_tokens - max_length + 1,
            stride
        ):

            chunks.append(
                token_ids[start:start + max_length]
            )

        # Include final chunk to cover remaining code
        if token_ids[-max_length:].tolist() != chunks[-1].tolist():

            chunks.append(
                token_ids[-max_length:]
            )

    probabilities = []

    # Run inference for each chunk
    for chunk in chunks:

        input_ids = chunk.unsqueeze(0).to(device)

        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

            # LABEL_1 = AI
            prob_ai = probs[0][1].item()

            probabilities.append(prob_ai)

    # Mean aggregation
    mean_prob_ai = sum(probabilities) / len(probabilities)

    return mean_prob_ai, len(chunks)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class CodeRequest(BaseModel):

    code: str = Field(
        ...,
        min_length=1,
        description="Source code to classify"
    )


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(request: CodeRequest):

    try:

        if model is None or tokenizer is None:

            raise HTTPException(
                status_code=503,
                detail="Model is not loaded"
            )

        # Sanitize incoming source code
        code_input = sanitize_text(request.code)

        if not code_input.strip():

            raise HTTPException(
                status_code=400,
                detail="Code input cannot be empty"
            )

        # Chunk-based prediction
        prob_ai, num_chunks = predict_with_chunks(
            code_input,
            max_length=MAX_SEQ_LENGTH,
            stride=CHUNK_STRIDE,
            threshold=PRECISION_THRESHOLD
        )

        # Classification
        predicted_class = (
            1 if prob_ai >= PRECISION_THRESHOLD else 0
        )

        # Confidence of predicted class
        confidence = (
            prob_ai
            if predicted_class == 1
            else 1.0 - prob_ai
        )

        label = LABELS[predicted_class]

        return {
            "prediction": int(predicted_class),
            "label": label,
            "confidence": round(confidence, 4),
            "prob_ai": round(prob_ai, 4),
            "prob_human": round(1.0 - prob_ai, 4),
            "chunks_evaluated": num_chunks,
            "threshold": PRECISION_THRESHOLD,
            "device": str(device)
        }

    except HTTPException:
        raise

    except Exception as error:

        print("Prediction error:", str(error))

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy" if model is not None else "not_ready",
        "model": "CodeBERT-v7.0",
        "model_path": MODEL_PATH,
        "device": str(device),
        "threshold": PRECISION_THRESHOLD
    }


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
async def root():

    return {
        "message": "CodeBERT AI Detection API is running",
        "endpoints": [
            "/predict",
            "/health",
            "/docs"
        ]
    }
