
from pymilvus import model
from constants import DENSE_EMBEDDING_MODEL_PATH, TOKENIZER_MODEL_PATH, SPARSE_EMBEDDING_MODEL_PATH
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
import scipy.sparse as sp
from pymilvus.model.sparse import SpladeEmbeddingFunction
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

class Embeddings:
    def __init__(self):
        # Default = all-MiniLM-L6-v2, dimensions 384
        self.dense_ef = SentenceTransformerEmbeddingFunction(model_name=DENSE_EMBEDDING_MODEL_PATH)
        self.sparse_ef = SpladeEmbeddingFunction(model_name=SPARSE_EMBEDDING_MODEL_PATH)
        self.token_splitter = SentenceTransformersTokenTextSplitter(model_name=TOKENIZER_MODEL_PATH)

    def encode_for_search(self, query: str) -> np.ndarray:
        """
        Encode the given text for search in the vector DB, with tokenization to improve accuracy.

        Args:
            search_text (str): The text input by the user to encode.

        Returns:
            np.ndarray: The normalized vector embedding of the tokenized input text.
        """
        # Tokenize the search text
        tokenized_texts = self.token_splitter.split_text(query)

        # If the tokenizer splits the text into multiple tokens, encode them separately
        # encoded_texts = self.embedding_function.encode(tokenized_texts)

        # # Normalize the embeddings
        # normalized_encoded_texts = encoded_texts / \
        #     np.linalg.norm(encoded_texts, axis=1, keepdims=True)

        normalized_dense_encoded_texts = self.dense_ef.encode_queries(tokenized_texts)
        normalized_sparse_encoded_texts = self.sparse_ef.encode_queries(tokenized_texts)

        # Aggregate the embeddings if there are multiple, this step depends on your application's needs
        # For simplicity, we'll average the embeddings to get a single vector representation
        if len(normalized_dense_encoded_texts) > 1:
            aggregated_dense_embedding = [np.mean(normalized_dense_encoded_texts, axis=0)]
        else:
            aggregated_dense_embedding = normalized_dense_encoded_texts

        if normalized_sparse_encoded_texts.shape[0] > 1:
            # Take mean of the sparse embeddings
            aggregated_sparse_embedding = sp.csr_matrix(np.mean(normalized_sparse_encoded_texts, axis=0))
        else:
            aggregated_sparse_embedding = normalized_sparse_encoded_texts

        return aggregated_dense_embedding, aggregated_sparse_embedding
