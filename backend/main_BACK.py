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
connection = psy.connect(**DB_CONNECT_KWARGS)
connection.set_session(autocommit=True)
cursor = connection.cursor()

def load_image_into_numpy_array(data):
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
        some_array.append(pickle.loads(cursor.fetchone()[0]))
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
    list_temp=[]
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


#base54 = ["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"]

#add_data_base64(base54)

#print(add_data_image(["test_img/banana_1.jpg", "test_img/scissors_2.jpg", "test_img/scissors_3.jpg", "test_img/tiger.jpeg"]))
#print(get_data_text(['tiger'], 2))
#print(get_data_images([cv2.imread("test_img/drill_3.jpg")], n_results=1))
#print(add_data_base64([""]))
#add_data_to_postgres(data=['test_img/drill_1.jpg'], ids=['7899'], d_type='image')
#add_data_to_postgres(data=["test_img/drill_1.jpg", "test_img/scissors_2.jpg"], ids=get_uid(["test_img/banana_1.jpg", "test_img/drill_2.jpg"]), d_type="np_array")
#print(base64_to_narray_ones(base54))
