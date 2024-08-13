from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import cv2
import numpy as np

app = FastAPI()

# Initialize the PaddleOCR with DBNet for text detection
ocr = PaddleOCR(det_model_dir='path/to/dbnet/model', use_angle_cls=True, lang='en')  # Modify with the actual path if needed

@app.get("/")
async def home():
    return {"message": "Text Detection API: Running on Render"}

@app.post("/detect-text-db")
async def detect_text_db(
    image: UploadFile = File(...),
    xmin: str = Form(...),
    ymin: str = Form(...),
    xmax: str = Form(...),
    ymax: str = Form(...)
):
    # Convert coordinates to integers
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)

    # Read image using OpenCV
    contents = await image.read()
    in_memory_file = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1280, 1280))

    # if img is None:
    #     continue
        # raise HTTPException(status_code=400, detail="Failed to load image")

    # Crop the image to the specified region
    cropped_image = img[ymin:ymax, xmin:xmax]

    # Convert the image to RGB
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Use PaddleOCR for text detection
    result = ocr.ocr(cropped_image_rgb, cls=True)
    arr = []
    # Prepare the response with bounding boxes
    detections = []
    for line in result:
        for word_info in line:
            bbox = word_info[0]
            # detections.append({
            #     "bounding_box": {
            #         "top_left": bbox[0],
            #         "top_right": bbox[1],
            #         "bottom_right": bbox[2],
            #         "bottom_left": bbox[3]
            #     },
            #     "text": word_info[1][0],
            #     "confidence": word_info[1][1]
            # })
            detections.append({
                "text": word_info[1][0],
                "confidence": word_info[1][1]
            })
            arr.append(word_info[1][0])
    # print(arr)
    result = find_deemed(arr)
    if result['found']:
        response = JSONResponse(content={"status": True ,"detections": detections, "message": "valid poster",})
    else:
        response = JSONResponse(content={"staus": False, "detections": detections, "message": "poster not valid",})

    return response

def find_deemed(strings):
    
    for string in strings:
        result = {
            "found": False,
            "start_index": -1,
            "end_index": -1,
        }
        lower_string = string.lower()
        if "deemed" in lower_string:
            start_index = lower_string.find("deemed")
            end_index = start_index + len("deemed")
            results = f"'Deemed' found at position {start_index} to {end_index - 1} in string: {string}"
            print(results)
            result = {
                "found": True,
                "start_index": start_index,
                "end_index": end_index,
            }
            break
    # print(result)
    return result