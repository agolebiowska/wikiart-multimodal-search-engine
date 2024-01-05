import os
import kfp.dsl as dsl

@dsl.component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9-slim"
)
def update_index(project_id: str,
                 location: str,
                 vertex_bucket: str,
                 idx_prefix: str,
                 index_name: str,
                 dimensions: int) -> str:

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    index = aiplatform.MatchingEngineIndex.list(
        location=location,
        project=project_id,
        filter=f'display_name="{index_name}"'
    )
    if len(index) <= 0:
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_name,
            project=project_id,
            location=location,
            contents_delta_uri=f"gs://{vertex_bucket}/{idx_prefix}/",
            dimensions=dimensions,
            approximate_neighbors_count=150,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=7
        )
        index_id = index.name
    else:
        index_id = index[0].name
        index = aiplatform.MatchingEngineIndex(index_name=index_id)
        index.update_embeddings(f"gs://{vertex_bucket}/{idx_prefix}/")

    return index_id