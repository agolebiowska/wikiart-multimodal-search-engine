import os
import kfp.dsl as dsl

@dsl.component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9-slim"
)
def deploy_index(project_id: str,
                 project_number: str,
                 location: str,
                 network: str,
                 vertex_bucket: str,
                 idx_prefix: str,
                 index_endpoint_name: str,
                 index_id: str):

    import uuid
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    endpoint = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f"display_name={index_endpoint_name}"
    )
    if len(endpoint) <= 0:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=index_endpoint_name,
            public_endpoint_enabled=False,
            project=project_id,
            location=location,
            network=f"projects/{project_number}/global/networks/{network}"
        )
    else:
        endpoint_name = endpoint[0].name
        endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_name
        )

    index = aiplatform.MatchingEngineIndex(index_name=index_id)
    endpoint.deploy_index(
        index=index, deployed_index_id=f"idx_{uuid.uuid4().hex}")