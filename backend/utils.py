import os

import hashlib
import cv2
import base64
import numpy as np
from numpy import asarray
from io import BytesIO
from PIL import Image
from CONFIG import *
import pickle

import psycopg2 as psy
import chromadb
from chromadb import Settings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

##---------chromadb------------------------------
client = chromadb.PersistentClient(path="/backend/data/", settings=Settings(allow_reset=True))
imageloader = ImageLoader()
multimodal_ef = OpenCLIPEmbeddingFunction()

multimodal_db = client.get_or_create_collection(name="multimodal",
                                                embedding_function=multimodal_ef,
                                                data_loader=imageloader,
                                                )

##----------postgresql----------------------------
DB_CONNECT_KWARGS['host'] = os.environ.get("DATABASE_HOST")
connection = psy.connect(**DB_CONNECT_KWARGS)
connection.set_session(autocommit=True)
cursor = connection.cursor()


def load_image_into_numpy_array(data):
    """

    """
    return np.array(Image.open(BytesIO(data)))


def base64_to_narray_list(base64_str) -> list:
    narray_img = []
    for j in range(len(base64_str)):
        image_bytes = base64.b64decode(base64_str[j])
        image = Image.open(BytesIO(image_bytes))
        narray_img.append(np.array(image))
    return narray_img


def base64_to_narray_ones(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes))
    return np.array(image)


def get_hash_of_base64(base64_str) -> list:
    array_hash = []
    for k in range(len(base64_str)):
        array_hash.append(hashlib.sha256(base64_str[k].encode('utf-8')).hexdigest()[:10])
    return array_hash


def get_uid(data) -> list:
    hash_img = []
    for i in range(len(data)):
        with open(data[i], "rb") as f:
            hash_img.append(hashlib.sha256(f.read()).hexdigest()[:10])
    return hash_img


def get_blobs(path):
    drawing = open(path, 'rb').read()
    return psy.Binary(drawing)


def get_image_array_from_postgres(uid) -> list:
    some_array = []
    for ids in range(len(uid)):
        cursor.execute(
            """
            SELECT np_array_bytes
            FROM chroma_images
            WHERE uniq_id=%s
            """,
            (uid[0][ids],)
        )
        some_array.append(pickle.loads(cursor.fetchone()[0]).tolist())
    return some_array


def add_data_to_postgres(data, ids, d_type):
    if d_type == 'base64':
        for i in range(len(data)):
            cursor.execute(
                """
                INSERT INTO chroma_images(uniq_id, np_array_bytes, image_bytes)
                VALUES (%s, %s, %s)
                """,
                (ids[i], pickle.dumps(base64_to_narray_ones(data[i])), psy.Binary(base64.b64decode(data[i])))
            )
    if d_type == 'image':
        for i in range(len(data)):
            cursor.execute(
                """
                INSERT INTO chroma_images(uniq_id, np_array_bytes, image_bytes)
                VALUES (%s, %s, %s)
                """,
                (ids[i], pickle.dumps(asarray(Image.open(data[i]))), get_blobs(data[i]))
            )


def add_data_base64(base64_string) -> int:
    ids = get_hash_of_base64(base64_string)
    image_data = base64_to_narray_list(base64_string)
    multimodal_db.add(
        ids=ids,
        images=image_data,
    )
    add_data_to_postgres(base64_string, ids, d_type='base64')
    return multimodal_db.count()


def add_data_image(paths) -> int:
    ids = get_uid(paths)
    multimodal_db.add(
        ids=ids,
        uris=paths,
    )
    add_data_to_postgres(paths, ids, d_type='image')
    return multimodal_db.count()


def get_data_text(text, n_results):
    query = multimodal_db.query(
        query_texts=text,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas', 'data', 'uris'],
    )
    query_dict = {
        'ids': query['ids'],
        'image_array': get_image_array_from_postgres(query['ids']),
    }
    return query_dict


def get_data_images(image, n_results):
    list_temp = []
    list_temp.append(load_image_into_numpy_array(image.file.read()))
    query = multimodal_db.query(
        query_images=list_temp,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas', 'data', 'uris'],
    )
    query_dict = {
        'ids': query['ids'],
        'image_array': get_image_array_from_postgres(query['ids']),
    }
    return query_dict

def get_total_row():
    return multimodal_db.count()