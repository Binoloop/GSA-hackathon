from src.vec_db.connect import client
from constants import (
    COLLECTION_NAME_CHUNKS,
    COLLECTION_NAME_EVALUATION,
    DENSE_WEIGHT_HYBRID_SEARCH,
)
from pymilvus import AnnSearchRequest, WeightedRanker, Collection
from pymilvus.exceptions import MilvusException

# Loading Collections
col_chunks = Collection(COLLECTION_NAME_CHUNKS)
col_chunks.load()


def insert_chunks_to_milvus(
    doc_id: str,
    sparse_embedded_text: str,
    dense_embedded_text: list,
    page_numbers_list: list,
):
    """
    Insert the vector embeddings associated with a document into Milvus.

    This function prepares the data by associating document IDs with their respective embeddings
    and then inserts this data into a Milvus collection. It flushes the collection to ensure data
    persistence and prints the number of entities added along with the total entities in the collection.

    Args:
        doc_id (str): The unique identifier for the document.
        embeded_text (list): A list of vector embeddings corresponding to segments of the document.

    Returns:
        list: The response from the Milvus insert operation, typically containing IDs of the inserted vectors.
    """
    # Prepare data for insertion into Milvus by associating document IDs with embeddings
    document_data = [
        {
            "doc_id": doc_id,
            "sparse_vector_embeddings": sparse_embedded_text[i],
            "dense_vector_embeddings": dense_embedded_text[i],
            "pp_num": page_numbers_list[i],
        }
        for i in range(len(dense_embedded_text))
    ]

    res = client.insert(collection_name=COLLECTION_NAME_CHUNKS, data=document_data)

    # Count number of entities in vec_db
    num_entities = client.query(
        COLLECTION_NAME_CHUNKS, filter="", output_fields=["count(*)"]
    )

    # Print the outcome of the insert operation
    print(
        f"Number of entities added to the db: {len(res['ids'])}, Total Entities in DB: {num_entities[0]['count(*)']}"
    )

    return res["ids"]


def search_through_chunks_collection(
    query_sparse_embeddings: str,
    query_dense_embeddings: str,
    max_chunks: int = None,
    hybrid_search: bool = True,
):
    """
    Search through the Milvus collection for the given text

    Args:
        search_text (str): The text input by the user to search for
        top_k (int): The number of results to return

    Returns:
        list: The top k results from the Milvus collection
    """
    # Use step function for calculating max_chunks
    max_chunks = 5

    # If hybrid search is enabled, perform a hybrid search
    if hybrid_search:
        return query_hybrid_search(
            query_sparse_embeddings,
            query_dense_embeddings,
            max_chunks,
            collection_name=COLLECTION_NAME_CHUNKS,
        )
    else:
        pass


# Count the number of chunks for a given doc_i


def search_through_chunks_collection(
    sparse_embeddings,
    dense_embeddings: str,
    top_results: int = 3,
    hybrid_search: bool = True,
):
    """
    Search through the Milvus collection for the given text

    A"rgs:
        search_text (str): The text input by the user to search for
        top_k (int): The number of results to return

    Returns:
        list: The top k results from the Milvus collection
    """

    if hybrid_search:
        results = query_hybrid_search(
            query_sparse_embeddings=sparse_embeddings,
            query_dense_embeddings=dense_embeddings,
            max_chunks=top_results,
            collection_name=COLLECTION_NAME_CHUNKS,
            output_fields=["title,text"],
        )
    else:
        results = client.search(
            collection_name=COLLECTION_NAME_CHUNKS,
            data=dense_embeddings,
            anns_field="dense_vector_embeddings",
            limit=top_results,
            output_fields=["title", "text"],
        )

    return results


def query_hybrid_search(
    query_sparse_embeddings,
    query_dense_embeddings,
    max_chunks,
    collection_name=COLLECTION_NAME_CHUNKS,
    output_fields=["title", "text"],
    filter=None,
):
    """
    Perform a hybrid search through the Milvus collection for the given text.

    Args:
        doc_id (str): The unique identifier for the document.
        query_sparse_embeddings (str): The query sparse embeddings.
        query_dense_embeddings (str): The query dense embeddings.
        max_chunks (int): The maximum number of chunks to return.
        page_number_filter (list): A list of page numbers to filter the search results.
        collection_name (str): The name of the Milvus collection to search.

    Returns:
        pymilvus HIT object: The top k results from the Milvus collection.
    """

    # Load the collection for hybrid search
    col = Collection(collection_name)
    col.load()

    # Check if a page number filter is provided, if yes then filter based on it
    sparse_req = AnnSearchRequest(
        query_sparse_embeddings,
        "sparse_vector_embeddings",
        {"metric_type": "IP"},
        limit=max_chunks,
    )
    # Create a request for searching on dense embeddings using page number filter
    dense_req = AnnSearchRequest(
        query_dense_embeddings,
        "dense_vector_embeddings",
        {"metric_type": "L2"},
        limit=max_chunks,
    )
    # Perform a hybrid search using both sparse and dense embeddings
    results = col.hybrid_search(
        [sparse_req, dense_req],
        rerank=WeightedRanker(
            1 - DENSE_WEIGHT_HYBRID_SEARCH, DENSE_WEIGHT_HYBRID_SEARCH
        ),
        limit=max_chunks,
        output_fields=["text"],
    )

    return results
