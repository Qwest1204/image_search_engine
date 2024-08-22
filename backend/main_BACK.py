import hashlib
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

client = chromadb.PersistentClient(path="data/")
imageloader = ImageLoader()
multimodal_ef = OpenCLIPEmbeddingFunction()

multimodal_db = client.get_or_create_collection(name="multimodal",
                                                embedding_function=multimodal_ef,
                                                data_loader=imageloader,
                                                )


def base64_to_narray(base64_str):
    narray_img = []
    for j in range(len(base64_str)):
        image_bytes = base64.b64decode(base64_str[j])
        image = Image.open(BytesIO(image_bytes))
        narray_img.append(np.array(image))
    return narray_img


def get_hash_of_base64(base64_str):
    array_hash = []
    for k in range(len(base64_str)):
        array_hash.append(hashlib.sha256(base64_str[k].encode('utf-8')).hexdigest()[:10])
    return array_hash


def get_uid(data) -> str:
    hash_img = []
    for i in range(len(data)):
        with open(data[i], "rb") as f:
            hash_img.append(hashlib.sha256(f.read()).hexdigest()[:10])
    return hash_img


def add_data_base64(base64_string) -> int:
    ids = get_hash_of_base64(base64_string)
    multimodal_db.add(
        ids=ids,
        images=base64_to_narray(base64_string),
    )
    return multimodal_db.count()


def add_data_image(paths) -> int:
    ids = get_uid(paths)
    multimodal_db.add(
        ids=ids,
        uris=paths,
    )
    return multimodal_db.count()


def get_data_text(text, n_results):
    return multimodal_db.query(
        query_texts=text,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas', 'data', 'uris'],
    )


def get_data_images(image, n_results):
    return multimodal_db.query(
        query_images=image,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas', 'data', 'uris'],
    )

#print(add_data(["test_img/drill_1.jpg", "test_img/scissors_2.jpg"]))
#print(get_data_text(['tiger'], 1))
#print(get_data_images([cv2.imread("test_img/tiger.jpeg")], n_results=1))
#print(add_data_base64([""]))
