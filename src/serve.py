from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Google Cloud SDK tu dong
    dung Application Default Credentials. Tren GCE, credentials den tu service
    account gan truc tiep vao VM, khong can luu private key tren may.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client = storage.Client()

    bucket = client.bucket(ARTIFACT_BUCKET)
    blob = bucket.blob(MODEL_KEY)

    if not blob.exists(client):
        raise RuntimeError(
            f"Model gs://{ARTIFACT_BUCKET}/{MODEL_KEY} chua ton tai. "
            "Hay chay pipeline huan luyen truoc khi khoi dong service."
        )

    blob.download_to_filename(MODEL_PATH)

    print(f"Model da duoc tai tu gs://{ARTIFACT_BUCKET}/{MODEL_KEY}.")


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail="Expected 10 features (adult income)",
        )

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}
    return {"prediction": prediction, "label": labels[prediction]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
