import os
import logging

import numpy as np
import gradio as gr
from PIL import Image as PILImage

from google.cloud import aiplatform, storage, bigquery
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint, matching_engine
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


def get_path(name, category):
    return f"{ALL_PREFIX}/{category}/{name}.jpg"


def get_url(name, category):
    return f"https://storage.cloud.google.com/{DATA_BUCKET}/{get_path(name, category)}"


def extract_author_title(image_id):
    parts = image_id.split('_')
    author = parts[0].replace('-', ' ')
    title = '_'.join(parts[1:])
    return author, title


def format_genre(genre):
    cleaned_string = genre.strip("[]").replace("'", "")
    cleaned_string = cleaned_string.split(",")[0]
    result = cleaned_string.lower().replace(" ", "-")
    return result


def format_folder(genre):
    cleaned_string = genre.replace(" ", "-")
    result = cleaned_string.lower()
    return result


def download_from_gcs(image_uri, data_bucket):
    try:
        client = storage.Client()
        bucket = client.bucket(data_bucket)
        blob = bucket.blob(image_uri)
        blob.download_to_filename(image_uri)
    except Exception as ex:
        logging.error(f"Error downloading {image_uri} from GCS: {ex}")


def get_image_metadata(image_id):
    try:
        author, title = extract_author_title(image_id)
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT artist, genre, description
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE artist = @author AND description = @title
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("author", "STRING", author),
                bigquery.ScalarQueryParameter("title", "STRING", title)]
        )

        query_job = client.query(query, job_config=job_config)
        result = query_job.result(max_results=1)

        for row in result:
            return row.artist, row.description, format_genre(row.genre)

        return "", "", ""
    except Exception as ex:
        logger.error(f"Error querying metadata for {image_id}: {ex}")
        return "", "", ""


def get_label(artist, description, genre):
    return f"{artist.title()} - {description.replace('-', ' ').title()} - {genre.replace('-', ' ').title()}"


def get_matches(query_emb, num_results, filter):
    result = index_endpoint.match(
        DEPLOYED_INDEX_ID,
        queries=[query_emb],
        num_neighbors=int(num_results),
        filter=[matching_engine.matching_engine_index_endpoint.Namespace(
            "category", filter, [])]
    )

    matches = []
    for match in result[0]:
        artist, description, genre = get_image_metadata(match.id)
        
        matches.append(
            (get_url(match.id, genre), get_label(artist, description, genre))
        )

    return matches


def image_query(image_data, num_results, genres_filter):
    if image_data is None:
        raise gr.Error("Image cannot be empty")
    
    try:
        image = PILImage.fromarray(np.uint8(image_data))
        image_path = "image.jpg"
        image.save(image_path)
        query_image = Image.load_from_file(image_path)
        query_emb = multimodalembedding.get_embeddings(
            image=query_image).image_embedding

        filter = [format_folder(genre) for genre in genres_filter]
            
        return get_matches(query_emb, num_results, filter)

    except Exception as ex:
        logger.error(f"Error: {ex}")
        return []


def text_query(text, num_results, genres_filter):
    if len(text) <= 0:
        raise gr.Error("Query cannot be empty")

    try:
        query_emb = multimodalembedding.get_embeddings(
            contextual_text=text).text_embedding

        filter = [format_folder(genre) for genre in genres_filter]

        return get_matches(query_emb, num_results, filter)

    except Exception as ex:
        logger.error(f"Error: {ex}")
        return []


def create_ui():
    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.emerald,
                                           secondary_hue=gr.themes.colors.emerald)) as iface:

        gr.Markdown("# Multimodal Search Engine for Art")

        genres = [
            "Expressionism",
            "Abstract expressionism",
            "Action painting",
            "Analytical cubism",
            "Art Nouveau modern",
            "Baroque",
            "Color field painting",
            "Realism",
            "Contemporary realism",
            "New Realism",
            "Cubism",
            "Synthetic cubism",
            "Early Renaissance",
            "High Renaissance",
            "Mannerism late Renaissance",
            "Northern Renaissance",
            "Fauvism",
            "Impressionism",
            "Post Impressionism",
            "Minimalism",
            "Naive art Primitivism",
            "Pointillism",
            "Pop art",
            "Rococo",
            "Romanticism",
            "Symbolism",
            "Ukiyo-e"
        ]

        with gr.Tab("Image-to-image search"):
            image = gr.Image()
            genres_filter = gr.CheckboxGroup(genres,
                                            label="Genres",
                                            info="Optional. Select genres to filter.")
            
            with gr.Row():
                num_results = gr.Dropdown([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                                        label="Number of results",
                                        value=20,
                                        info="How many results to show.")
                clear = gr.ClearButton(value="Clear input", components=[image, genres_filter])
                find_by_image_btn = gr.Button("Get images", variant="primary", icon="icon_search.svg")

            images = gr.Gallery(label="Images",
                                show_label=True,
                                columns=[5],
                                object_fit="cover")

            find_by_image_btn.click(image_query, inputs=[image, num_results, genres_filter], outputs=[images])


        with gr.Tab("Text-to-image search"):
            text = gr.Textbox(label="Text query")
            genres_filter = gr.CheckboxGroup(genres,
                                            label="Genres",
                                            info="Optional. Select genres to filter.")
            
            with gr.Row():
                num_results = gr.Dropdown([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                                        label="Number of results",
                                        value=20,
                                        info="How many results to show.")
                clear = gr.ClearButton(value="Clear input", components=[text, genres_filter])
                find_by_text_btn = gr.Button("Get images", variant="primary", icon="icon_search.svg")

            images = gr.Gallery(label="Images",
                                show_label=True,
                                columns=[5],
                                object_fit="cover")
        
            find_by_text_btn.click(text_query, inputs=[text, num_results, genres_filter], outputs=[images])

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
    ui.launch(inline=False,
              server_name="0.0.0.0",
              server_port=7860)