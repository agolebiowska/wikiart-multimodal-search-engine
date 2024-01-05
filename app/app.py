import os
import logging

import numpy as np
import gradio as gr
from PIL import Image as PILImage

from google.cloud import aiplatform, storage, bigquery
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from vertexai.preview.vision_models import MultiModalEmbeddingModel, Image

from dotenv import load_dotenv
load_dotenv()

handler = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[handler])
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
INDEX_ID = os.getenv("INDEX_ID")
INDEX_ENDPOINT_ID = os.getenv("INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID")
ALL_PREFIX = os.getenv("ALL_PREFIX")
DATA_BUCKET = os.getenv("DATA_BUCKET")
DATASET = os.getenv("DATASET")
TABLE = os.getenv("TABLE")

multimodalembedding = None
index = None
index_endpoint = None


def get_path(name):
    return f"{DATA_BUCKET}/{ALL_PREFIX}/{name}.jpg"


def get_url(name):
    return f"https://storage.cloud.google.com/{get_path(name)}"


def get_image_metadata(image_id):
    try:
        filename = get_path(image_id)
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT artist, genre, description
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE filename = @filename
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("filename", "STRING", filename)]
        )

        query_job = client.query(query, job_config=job_config)
        result = query_job.result(max_results=1)

        for row in result:
            return artist, description, genre

        return "", "", ""
    except Exception as ex:
        logger.error(f"Error querying metadata for {image_id}: {ex}")
        return "", "", ""


def get_label(image_id):
    artist, description, genre = get_image_metadata(image_id)
    return f"{artist} - {description} - {genre}"


def get_matches(query_emb):
    result = index_endpoint.match(
        DEPLOYED_INDEX_ID,
        queries=[query_emb],
        num_neighbors=20
    )

    matches = []
    for match in result[0]:
        matches.append(
            (get_path(match.id), get_label(match.id))
        )

    return matches


def image_query(image_data):
    try:
        image = PILImage.fromarray(np.uint8(image_data))
        image_path = "image.jpg"
        image.save(image_path)
        query_image = Image.load_from_file(image_path)
        query_emb = multimodalembedding.get_embeddings(
            image=query_image).image_embedding
            
        return get_matches(query_emb)

    except Exception as ex:
        logger.error(f"Error: {ex}")
        return []


def text_query(text):
    try:
        query_emb = multimodalembedding.get_embeddings(
            contextual_text=text).text_embedding

        return get_matches(query_emb)

    except Exception as ex:
        logger.error(f"Error: {ex}")
        return []


def create_ui():
    with gr.Blocks() as iface:
        gr.Markdown("Wikiart Search engine")

        with gr.Tab("Image-to-image search"):
            image = gr.Image()
            find_by_image_btn = gr.Button("Get similar images")

        with gr.Tab("Text-to-image search"):
            text = gr.Text(label="Text query")
            find_by_text_btn = gr.Button("Get images with similar semantic value")

        images = gr.Gallery(label="Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=[4],
                            object_fit="contain",
                            scale=0)

        find_by_image_btn.click(image_query, inputs=[image], outputs=[images])
        find_by_text_btn.click(text_query, inputs=[text], outputs=[images])
        # gr.Examples([], [])

    return iface


if __name__ == '__main__':
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    logger.info(f"Initialized AI Platform for project {PROJECT_ID}")

    multimodalembedding = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    try:
        index = aiplatform.MatchingEngineIndex(index_name=INDEX_ID)
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=INDEX_ENDPOINT_ID
        )
    except Exception as ex:
        logger.error(f"Could not get Vertex AI Endpoint: {ex}")

    ui = create_ui()
    ui.queue().launch(inline=False,
                      server_name="0.0.0.0",
                      server_port=7860)