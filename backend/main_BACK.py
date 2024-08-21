import hashlib
import cv2

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


def get_uid(path) -> str:
    hash = []
    for i in range(len(path)):
        with open(path[i], "rb") as f:
            hash.append(hashlib.sha256(f.read()).hexdigest()[:10])
    return hash

def add_data(paths) -> int:
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
#print(get_data_text(['drill'], 2))
#print(get_data_images([cv2.imread("test_img/scissors_3.jpg")], n_results=1))