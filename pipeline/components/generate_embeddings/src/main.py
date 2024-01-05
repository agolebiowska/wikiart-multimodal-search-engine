import os
import logging
import datetime
import json
import utils
import google.cloud.logging
from utils import list_gcs_directories, list_gcs_files, download_from_gcs, upload_to_gcs, is_file_empty, resize_image
from google.cloud import storage, aiplatform
from vertexai.preview.vision_models import MultiModalEmbeddingModel, Image
from functools import partial
import argparse

client = google.cloud.logging.Client()
client.setup_logging()


def process_image_batch(batch, all_prefix, fail_prefix, data_bucket):
    embeddings = []
    for search_image in batch:
        try:
            resize_image(search_image)
            image = Image.load_from_file(search_image)
            emb = multimodalembedding.get_embeddings(
                image=image,
            )
            embeddings.append(emb.image_embedding)
        except Exception as ex:
            logging.error(f"Error processing image {search_image}: {ex}")
            upload_to_gcs(data_bucket,
                        f"{fail_prefix}/{os.path.relpath(search_image, all_prefix)}",
                        search_image)
    return embeddings


def process(batch, category, file_name, all_prefix, fail_prefix, vertex_bucket, data_bucket):
    embeddings = process_image_batch(batch, all_prefix, fail_prefix, data_bucket)
    logging.info(f'Successfully processed {len(embeddings)} images')

    with open(file_name, 'w') as f:
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                continue

            f.write(json.dumps({
                "id": os.path.basename(batch[i])[:-4],
                "embedding": embedding,
                "restricts": [
                    {
                        "namespace": "category",
                        "allow": [category],
                    }
                ]
            }) + '\n')

    if is_file_empty(file_name):
        raise ValueError(f"{file_name} is empty")

    upload_to_gcs(vertex_bucket, f"{idx_prefix}/{file_name}", file_name)
    logging.info(f'Successfully uploaded embeddings to GCS: {file_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--project_id', type=str, required=True, help='GCP Project ID.')
    parser.add_argument('--location', type=str, required=True, help='GCP location (e.g., us-central1).')
    parser.add_argument('--data_bucket', type=str, required=True, help='GCS bucket containing input images.')
    parser.add_argument('--vertex_bucket', type=str, required=True, help='GCS bucket for storing embeddings.')
    parser.add_argument('--all_prefix', type=str, required=True, help='Prefix for all images.')
    parser.add_argument('--idx_prefix', type=str, required=True, help='Prefix for index files.')
    parser.add_argument('--fail_prefix', type=str, required=True, help='Prefix for failed images.')
    args = parser.parse_args()

    project_id = args.project_id
    location = args.location
    data_bucket = args.data_bucket
    vertex_bucket = args.vertex_bucket
    all_prefix = args.all_prefix
    idx_prefix = args.idx_prefix
    fail_prefix = args.fail_prefix

    aiplatform.init(project=project_id, location=location)
    multimodalembedding = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    try:
        batch_size = 100

        # List all directories in the data bucket
        directories = list_gcs_directories(data_bucket)

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            logging.info(f"Processing images in directory: {directory}")

            # List all files in the current directory
            files_list = list_gcs_files(data_bucket, directory)

            # Download and process images in batches
            for i in range(0, len(files_list), batch_size):
                batch = files_list[i:i + batch_size]

                # Download images
                for image_path in batch:
                    download_from_gcs(image_path, data_bucket)
                logging.info(f'Successfully downloaded {len(batch)} images')

                # Process and upload embeddings
                current_datetime = datetime.datetime.now()
                formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
                category = os.path.basename(directory)  # Use the directory name as the category
                file_name = f'{formatted_datetime}_{category}_batch_{i // batch_size}.json'
                process(batch, category, file_name, all_prefix, fail_prefix, vertex_bucket, data_bucket)

                # Remove downloaded images to free up memory
                for image_path in batch:
                    os.remove(image_path)

    except Exception as ex:
        logging.error(f"Error in main process: {ex}")
        raise