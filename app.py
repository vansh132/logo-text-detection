from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import easyocr

app = FastAPI()
reader = easyocr.Reader(['en'])

@app.get('/')
def res():
    return {'message': "Text Detection: Server running on PORT 127.0.0.1"}

@app.post("/detect-text")
async def detect_text(
    image: UploadFile = File(...),
    xmin: str = Form(...),
    ymin: str = Form(...),
    xmax: str = Form(...),
    ymax: str = Form(...)
):
    try:
        # Convert file to an OpenCV image
        contents = await image.read()
        in_memory_file = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1280, 1280))

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        # Crop the image to the specified bounding box
        cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        # Perform text recognition on the cropped image
        text_recognition_results = reader.readtext(cropped_img)
        recognized_text = " ".join([text for bbox, text, score in text_recognition_results])

        return JSONResponse(content={"status": True, "recognized_text": recognized_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))