from typing import Annotated
import json
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from utils import get_data_text, get_data_images, get_total_row, add_data_to_system
from pydantic import BaseModel
import base64

app = FastAPI()


class QueryByText(BaseModel):
    query: str
    n_num: int


class QueryByImage(BaseModel):
    image: UploadFile = File(...)
    n_num: int


@app.post("/add_data_to_system")
async def upload_image(image: UploadFile = File(...)):
    image_data = await image.read()
    add_data_to_system(image_data)
    return JSONResponse({"message": "OK"})


@app.get("/get_data_from_system_by_text")
async def get_data_from_system_by_text(query: Annotated[QueryByText, Depends()]):
    result = get_data_text(query.query, query.n_num)
    return JSONResponse({"response": {"ids": result['ids'],
                                      'L2_distances': result['distances'],
                                      "numpy_array": json.dumps(result['image_array'])
                                      }})


@app.post("/get_data_from_system_by_image")
async def get_data_from_system_by_image(query: Annotated[QueryByImage, Depends()]):
    result = get_data_images(query.image, query.n_num)
    return JSONResponse({"response": {"ids": result['ids'],
                                      'L2_distances': result['distances'],
                                      "numpy_array": json.dumps(result['image_array'])
                                      }})


@app.get("/get_row")
async def get_total_items():
    return JSONResponse({"total_row": get_total_row()})
