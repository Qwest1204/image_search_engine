import hashlib
import cv2
import os
import pickle
from io import BytesIO

import chromadb
import numpy as np
import psycopg2 as psy
from PIL import Image
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


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
    """ Loads an image from bytes data into a NumPy array.

    Args:
        data: Bytes data representing the image.

    Returns:
        A NumPy array representing the image.
    """

    return np.array(Image.open(BytesIO(data)))


def get_hash(data):
    """ Function for getting hash from binary data.

    Args:
        data: bytes data representing the image.

    Return:
        sha256 hash of data.
    """
    return hashlib.sha256(data).hexdigest()[:10]


def get_image_array_from_postgres(uid) -> list:
    """ Function for getting numpy array from database by unique id

    Args:
        uid: unique id of the image.

    Returns:
        list of numpy arrays representing the images.
    """
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


def add_data_to_postgres(data, ids):
    """ Function for getting numpy array from database by unique id

    Args:
        data: unique id of the image.
        ids: list of unique ids of the images.

    """
    cursor = connect_to_postgres(DB_CONNECT_KWARGS).cursor()
    for i in range(len(data)):
        cursor.execute(
            """
            INSERT INTO chroma_images(uniq_id, np_array_bytes, image_bytes)
            VALUES (%s, %s, %s)
            """,
            (ids[i], pickle.dumps(cv2.imdecode(np.frombuffer(data[i], np.uint8), -1)), data[i])
        )


def get_data_text(text, n_results) -> dict:
    """ Function for getting data from database by test query

    Args:
        text: text query to get data from system.
        n_results: number of results to return.

    Return:
        query_dict: Dictionary of query dictionaries.
    """
    query = multimodal_db.query(
        query_texts=text,
        n_results=n_results,
        include=['distances'],
    )
    query_dict = {
        'ids': query['ids'],
        'distances': query['distances'],
        'image_array': get_image_array_from_postgres(query['ids']),
    }
    return query_dict


def get_data_images(image, n_results) -> dict:
    """ Function for getting data from database by numpy array

    Args:
        image: numpy array representing the image.
        n_results: number of results to return.

    Return:
        query_dict: Dictionary of query dictionaries.
    """
    list_temp = [load_image_into_numpy_array(image.file.read())]
    query = multimodal_db.query(
        query_images=list_temp,
        n_results=n_results,
        include=['distances'],
    )
    query_dict = {
        'ids': query['ids'],
        'distances': query['distances'],
        'image_array': get_image_array_from_postgres(query['ids']),
    }
    return query_dict


def get_total_row():
    """ Return the total number of rows in the database

    Returns:
        total number of rows in the database

    """
    return multimodal_db.count()


def add_data_to_system(data) -> int:
    """ Function for adding data to system

    Args:
        data: numpy array representing the image.

    Returns:
        integer number count of row in the database.
    """
    ids = [get_hash(data)]
    image_data = [cv2.imdecode(np.frombuffer(data, np.uint8), -1)]
    multimodal_db.add(
        ids=ids,
        images=image_data,
    )
    add_data_to_postgres([data], ids)
    return multimodal_db.count()
