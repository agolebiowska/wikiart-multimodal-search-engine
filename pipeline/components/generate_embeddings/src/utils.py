import os
import logging
import json
import google.cloud.logging
from functools import partial
from google.cloud import storage
from PIL import Image as PILImage
from vertexai.preview.vision_models import MultiModalEmbeddingModel, Image


def download_from_gcs(image_uri, data_bucket):
    try:
        client = storage.Client()
        bucket = client.bucket(data_bucket)
        blob = bucket.blob(image_uri)
        blob.download_to_filename(image_uri)
    except Exception as ex:
        logging.error(f"Error downloading {image_uri} from GCS: {ex}")


def list_gcs_files(data_bucket, prefix, allowed_extensions=['.jpg']):
    try:
        client = storage.Client()
        bucket = client.bucket(data_bucket)
        files_list = [blob.name for blob in bucket.list_blobs(prefix=f"{prefix}/") 
        if blob.name.lower().endswith(tuple(allowed_extensions))]
        return files_list
    except Exception as ex:
        logging.error(f"Error listing files in GCS: {ex}")
        raise


def list_gcs_directories(data_bucket):
    try:
        client = storage.Client()
        bucket = client.bucket(data_bucket)
        directories = set()
        blobs = bucket.list_blobs()
        for blob in blobs:
            if blob.name == 'all/':
                continue
            directory = '/'.join(blob.name.split('/')[:-1])
            directories.add(directory)
        return list(directories)
    except Exception as ex:
        logging.error(f"Error listing directories in GCS: {ex}")
        raise


def upload_to_gcs(bucket_name, gcs_path, local_path):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
    except Exception as ex:
        logging.error(f"Error uploading {local_path} to GCS: {ex}")
        raise


def resize_image(image_path):
    try:
        with PILImage.open(image_path) as im:
            im.thumbnail((1024, 1024), PILImage.LANCZOS)
            im.save(image_path)
    except Exception as ex:
        logging.error(f"Error resizing image {image_path}: {ex}")
        raise


def is_file_empty(file_path):
    file_size_bytes = os.path.getsize(file_path)
    return file_size_bytes == 0