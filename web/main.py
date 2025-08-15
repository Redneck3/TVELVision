import os
import io
import base64
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import tf_keras


from pathlib import Path


APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
MODEL_PATH = APP_DIR / "h5" / "tablet_detector.h5" 
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory="E:/Code/TVELVision/web/templates")

# Загружаем модель один раз при старте
try:
    model = tf_keras.models.load_model(str(MODEL_PATH))
    print("Model loaded:", MODEL_PATH)
except Exception as e:
    model = None
    print("Failed to load model:", e)


def save_data_url(data_url: str, prefix="capture") -> str:
    # data_url
    header, encoded = data_url.split(',', 1) if ',' in data_url else (None, data_url)
    data = base64.b64decode(encoded)
    filename = f"{prefix}_{datetime.utcnow()}_{uuid.uuid4().hex[:6]}.png"
    path = UPLOAD_DIR / filename
    with open(path, "wb") as f:
        f.write(data)
    return str(path)

def prepare_for_model(img_bgr) -> np.ndarray:
    try:
        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized, axis=(0, -1))
        return input_tensor
    except Exception as e:
        raise RuntimeError("Error preparing image for model: " + str(e))
    
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# route: upload file (form)
@app.post("/upload-file")
async def upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    #save
    filename = f"upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    path = UPLOAD_DIR / filename
    with open(path, "wb") as f:
        f.write(contents)
    
    # decode to cv2
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    try:
        inp = prepare_for_model(img)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    
    if model is None:
        return JSONResponse({"error": "Model not loaded on server"}, status_code=500)
    
    pred = float(model.predict(inp, verbose=0)[0][0])
    detected = pred > 0.5
    return JSONResponse({"detected": bool(detected), "probability": float(pred), "saved_path": str(path)})

# route: detect from camera (dataURL posted as form field "image_data")
@app.post("/detect")
async def detect(image_data: str = Form(...)):
    # save incoming dataURL to file
    try:
        saved_path = save_data_url(image_data, prefix="camera")
    except Exception as e:
        return JSONResponse({"error": "Bad image data: " + str(e)}, status_code=400)

    # read saved file via cv2
    img = cv2.imread(saved_path)
    if img is None:
        return JSONResponse({"error": "Could not decode image."}, status_code=400)
    try:
        inp = prepare_for_model(img)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if model is None:
        return JSONResponse({"error": "Model not loaded on server."}, status_code=500)

    # predict
    pred = float(model.predict(inp, verbose=0)[0][0])
    detected = pred > 0.5

    return JSONResponse({
        "detected": bool(detected),
        "probability": float(pred),
        "saved_path": saved_path
    })