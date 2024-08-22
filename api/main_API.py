from fastapi import FastAPI, Request, Response

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}