from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from main_BACK import add_data_base64
import base64

app = FastAPI()

@app.post("/add_data_to_system")
async def upload_image(image: UploadFile = File(...)):
    image_data = await image.read()
    image_base64 = [base64.b64encode(image_data).decode('utf-8')]
    add_data_base64(image_base64)
    return JSONResponse({"message": "Image received successfully"})