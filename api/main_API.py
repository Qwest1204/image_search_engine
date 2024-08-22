from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import base64

app = FastAPI()

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):

    image_data = await image.read()

    image_base64 = base64.b64encode(image_data).decode('utf-8')

    with open("image.png", "wb") as f:
        f.write(base64.b64decode(image_base64))

    return JSONResponse({"message": "Image received successfully"})