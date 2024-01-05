import os
import kfp
from kfp import compiler, dsl, components
from typing import NamedTuple, List
from google.cloud import aiplatform

from components.update_index import update_index
from components.deploy_index import deploy_index

from dotenv import load_dotenv
load_dotenv()

@dsl.container_component
def generate_embeddings(
    project_id: str,
    location: str,
    data_bucket: str,
    vertex_bucket: str,
    all_prefix: str,
    idx_prefix: str,
    fail_prefix: str):

  return dsl.ContainerSpec(
      image=os.getenv('DOCKER_IMAGE'),
      command=['python3', '/generate_embeddings/src/main.py'],
      args=[
        '--project_id', project_id,
        '--location', location,
        '--data_bucket', data_bucket,
        '--vertex_bucket', vertex_bucket,
        '--all_prefix', all_prefix,
        '--idx_prefix', idx_prefix,
        '--fail_prefix', fail_prefix]
  )

@dsl.pipeline(
    pipeline_root=os.getenv('PIPELINE_ROOT'),
    name="pipeline-wikiart-search-engine",
    description="Pipeline for deploying Wikiart search engine"
)
def pipeline(
    project_id: str,
    project_number: str,
    location: str,
    network: str,
    data_bucket: str,
    all_prefix: str,
    idx_prefix: str,
    fail_prefix: str,
    vertex_bucket: str,
    index_name: str,
    index_endpoint_name: str,
    dimensions: int
):

    generate_embeddings_op = generate_embeddings(
        project_id=project_id,
        location=location,
        data_bucket=data_bucket,
        vertex_bucket=vertex_bucket,
        all_prefix=all_prefix,
        idx_prefix=idx_prefix,
        fail_prefix=fail_prefix)

    update_index_op = update_index(
        project_id=project_id,
        location=location,
        vertex_bucket=vertex_bucket,
        idx_prefix=idx_prefix,
        index_name=index_name,
        dimensions=dimensions).after(generate_embeddings_op)

    deploy_inedx_op = deploy_index(
        project_id=project_id,
        project_number=project_number,
        location=location,
        network=network,
        vertex_bucket=vertex_bucket,
        idx_prefix=idx_prefix,
        index_endpoint_name=index_endpoint_name,
        index_id=update_index_op.output).after(update_index_op)


if __name__ == '__main__': 
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml"
    )

    job = aiplatform.PipelineJob(
        display_name="pipeline-wikiart-search-engine",
        template_path="pipeline.yaml",
        pipeline_root=os.getenv('PIPELINE_ROOT'),
        enable_caching=False,
        failure_policy="fast",
        parameter_values={
            "project_id": os.getenv('PROJECT_ID'),
            "project_number": os.getenv('PROJECT_NUMBER'),
            "location": os.getenv('LOCATION'),
            "network": os.getenv('NETWORK'),
            "data_bucket": os.getenv('DATA_BUCKET'),
            "vertex_bucket": os.getenv('BUCKET'),
            "all_prefix": os.getenv('ALL_PREFIX'),
            "idx_prefix": os.getenv('IDX_PREFIX'),
            "fail_prefix": os.getenv('FAIL_PREFIX'),
            "index_name": os.getenv('INDEX_NAME'),
            "index_endpoint_name": os.getenv('INDEX_ENDPOINT_NAME'),
            "dimensions": int(os.getenv('DIMENSIONS'))}
    )

    job.run(service_account=os.getenv('SERVICE_ACCOUNT'))