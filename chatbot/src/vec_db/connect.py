import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, connections, FieldSchema, DataType, CollectionSchema, Collection
from constants import COLLECTION_NAME_CHUNKS
# load_dotenv()
COLLECTION_NAME_CHUNKS = "data_gov_gsa_hacks"


client = MilvusClient(
    # Cluster endpoint obtained from the console
    uri="http://localhost:19530",
)

# Connect to the Milvus server
connections.connect(uri="http://localhost:19530", token=os.environ.get("MILVUS_TOKEN"))


if COLLECTION_NAME_CHUNKS not in client.list_collections():

    id_field = FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    
    dataset_id = FieldSchema(name="dataset_id", dtype=DataType.VARCHAR, max_length=264)

    # Define field for document IDs
    title = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=264)
    
    # description
    text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=264)

    # Sparse Vector Field
    sparse_vector_field = FieldSchema(name="sparse_vector_embeddings", dtype=DataType.SPARSE_FLOAT_VECTOR)

    # Define field for vector embeddings, 384 dimensions
    dense_vector_field = FieldSchema(name="dense_vector_embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)

    # Create a collection schema
    schema = CollectionSchema(fields=[id_field, dataset_id, title, text, sparse_vector_field, dense_vector_field], description="GSA Hackathon", auto_id=True, enable_dynamic_field=True)

    # Create the collection in Milvus
    col_chunks = Collection(COLLECTION_NAME_CHUNKS, schema)
    # Define the index parameters
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    dense_index = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 50}}
    # Create the index
    col_chunks.create_index("sparse_vector_embeddings", sparse_index)
    col_chunks.create_index("dense_vector_embeddings", dense_index)
else:
    print("THERE")