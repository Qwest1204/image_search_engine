from typing import Annotated

from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from main_BACK import add_data_base64, get_data_text
from pydantic import BaseModel
import base64

app = FastAPI()

class Item(BaseModel):
    query: str
    n_num: int


@app.post("/add_data_to_system")
async def upload_image(image: UploadFile = File(...)):
    image_data = await image.read()
    image_base64 = [base64.b64encode(image_data).decode('utf-8')]
    add_data_base64(image_base64)
    return JSONResponse({"message": "Image received successfully"})


@app.get("/get_data_from_system_from_text")
async def get_data_from_system_from_text(query: Annotated[Item, Depends()]):
    result = get_data_text(query.query, query.n_num)
    return JSONResponse({"message": result['ids']})