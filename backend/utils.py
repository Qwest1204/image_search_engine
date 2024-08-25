import base64
import hashlib
import os
import pickle
from io import BytesIO

import chromadb
import numpy as np
import psycopg2 as psy
from PIL import Image
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from numpy import asarray

from CONFIG import DB_CONNECT_KWARGS

##---------chromadb------------------------------
client = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST"), port=8000)
imageloader = ImageLoader()
multimodal_ef = OpenCLIPEmbeddingFunction()

multimodal_db = client.get_or_create_collection(name="multimodal",
                                                embedding_function=multimodal_ef,
                                                data_loader=imageloader,
                                                )


def connect_to_postgres(DB_CONNECT_KWARGS):
    """ Function for connecting to a database, implemented via psycopg2

    Args:
        DB_CONNECT_KWARGS: Dictionary of database connection parameters. look in CONFIG.py

    Returns:
        connection: database connection
    """

    try:
        DB_CONNECT_KWARGS['host'] = os.environ.get("DATABASE_HOST")
        connection = psy.connect(**DB_CONNECT_KWARGS)
        connection.set_session(autocommit=True)
        return connection
    except (Exception, psy.Error) as error:
        print("Error of connection to database: ", error)
        return None


def load_image_into_numpy_array(data):
    """Loads an image from bytes data into a NumPy array.

    Args:
        data: Bytes data representing the image.

    Returns:
        A NumPy array representing the image.
    """

    return np.array(Image.open(BytesIO(data)))


def base64_to_narray(base64_str) -> list:
    """Converts a list of base64 encoded images to NumPy arrays.

    Args:
       base64_str: A list of base64 encoded strings.

    Returns:
        narray_img: A list of NumPy arrays representing the images.
    """

    narray_img = []
    for j in range(len(base64_str)):
        image_bytes = base64.b64decode(base64_str[j])
        image = Image.open(BytesIO(image_bytes))
        narray_img.append(np.array(image))
    return narray_img


# def base64_to_narray_ones(base64_str):
#     image_bytes = base64.b64decode(base64_str)
#     image = Image.open(BytesIO(image_bytes))
#     return np.array(image)


def get_hash_of_base64(base64_str) -> list:
    """Generates a SHA256 hash for each base64 encoded string.

    Args:
        base64_str: A list of base64 encoded strings.

    Returns:
        A list of SHA256 hashes (truncated to 10 characters).
    """

    array_hash = []
    for k in range(len(base64_str)):
        array_hash.append(hashlib.sha256(base64_str[k].encode('utf-8')).hexdigest()[:10])
    return array_hash

def get_blobs(path):
    """Reads image data from a file as binary.

    Args:
        path: The path to the image file.

    Returns:
        Binary data representing the image.
    """

    drawing = open(path, 'rb').read()
    return psy.Binary(drawing)


def get_image_array_from_postgres(uid) -> list:
    some_array = []
    cursor = connect_to_postgres(DB_CONNECT_KWARGS).cursor()
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
    cursor = connect_to_postgres(DB_CONNECT_KWARGS).cursor()
    if d_type == 'base64':
        for i in range(len(data)):
            cursor.execute(
                """
                INSERT INTO chroma_images(uniq_id, np_array_bytes, image_bytes)
                VALUES (%s, %s, %s)
                """,
                (ids[i], pickle.dumps(base64_to_narray([data[i]])[0]), psy.Binary(base64.b64decode(data[i])))
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
    image_data = base64_to_narray(base64_string)
    multimodal_db.add(
        ids=ids,
        images=image_data,
    )
    add_data_to_postgres(base64_string, ids, d_type='base64')
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
    """ Return the total number of rows in the database

    Returns:
        total number of rows in the database

    """
    return multimodal_db.count()