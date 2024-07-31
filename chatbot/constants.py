import os


DENSE_WEIGHT_HYBRID_SEARCH = 0.8

LLM_TEMPERATURE = 0.2
LLM_MODEL_GCP = "chat-bison"
LLM_MAX_OUTPUT_TOKENS = 256
LLM_NAME_OPENAI = "gpt-3.5-turbo-0125"
COLLECTION_NAME_CHUNKS = 'data_gov_gsa_hacks'
COLLECTION_NAME_EVALUATION = os.getenv("COLLECTION_NAME_EVALUATION")
LLM_MAX_NAME_EXTRACT_LIMIT = 10000 # Max allowed limit for text length to be sent to llm for name extraction
LLM_MAX_NAME_EXTRACT_INPUT_TOKEN = 2000 # Max characters to send to llm

# Vectorization
VECTOR_DIMENSIONS = 768  # arctic-embed-m

DENSE_EMBEDDING_MODEL = 'Snowflake/arctic-embed-m'
SPARSE_EMBEDDING_MODEL = 'naver/splade-cocondenser-ensembledistil'
TOKENIZER_MODEL = 'Snowflake/arctic-embed-m'

DENSE_EMBEDDING_MODEL_PATH = '/app/models/sentence_transformers/{}/'.format(
    DENSE_EMBEDDING_MODEL)
TOKENIZER_MODEL_PATH = '/app/models/sentence_transformers/{}/'.format(
    TOKENIZER_MODEL)
SPARSE_EMBEDDING_MODEL_PATH = '/app/models/sentence_transformers/{}/'.format(
    SPARSE_EMBEDDING_MODEL)


# Minimum character length of extracted text to ensure that the text is extracted properly.
# This value was chosen as a 20% of the current average value of input text length, which is 1500 characters.
# This was done to accomodate smaller documents like a passport for example which would not have as much text.
MINIMUM_LENGTH_EXTRACTED_TEXT = 300


CHUNK_OVERLAP = 0
TOKENS_PER_CHUNK = 300
MAX_CHAR_CHUNK_SIZE = 980

# Sentry Config
SENTRY_DNS=os.getenv('SENTRY_DNS') # Sentry for reporting errors
SENTRY_TRACE_SAMPLE_RATE = 1.0

PROJECT_ID = os.getenv('PROJECT_ID')
GCP_ACCESS_TOKEN = os.getenv('GCP_ACCESS_TOKEN')

# Define constants for step function and searching
CHUNK_THRESHOLD_1 = 6
CHUNK_THRESHOLD_2 = 40
MAX_CHUNKS_SMALL = 2
MAX_CHUNKS_MEDIUM = 4
MAX_CHUNKS_LARGE = 6
DEFAULT_TOP_RESULTS = 3

# Set weight for hybrid search
DENSE_WEIGHT_HYBRID_SEARCH = 0.8

# Vectorization
VECTOR_DIMENSIONS = 768  # arctic-embed-m

DENSE_EMBEDDING_MODEL_PATH = 'Snowflake/arctic-embed-m'
SPARSE_EMBEDDING_MODEL_PATH = 'naver/splade-cocondenser-ensembledistil'
TOKENIZER_MODEL_PATH = 'Snowflake/arctic-embed-m'


LLM_NAME_OPENAI = "gpt-3.5-turbo-0125"
LLM_PARAMETER_TEMPERATURE = 0.5

# Minimum character length of extracted text to ensure that the text is extracted properly.
# This value was chosen as a 20% of the current average value of input text length, which is 1500 characters.
# This was done to accomodate smaller documents like a passport for example which would not have as much text.
MINIMUM_LENGTH_EXTRACTED_TEXT = 300

# Maximum number of words for evaluation feedback
MAX_WORDS_EVALUATION_FEEDBACK = 200
# Maximum number of words for documents summary
MAX_WORDS_DOCUMENTS_SUMMARY = 200
# Maximum number of words for justification
LIMIT_NUM_OF_WORDS_FOR_JUSTIFICATION = 200
# DNS for Sentry
SENTRY_DOCS_DNS = os.getenv('SENTRY_DOCS_DNS')

# Chunking
CHUNK_OVERLAP = 0
TOKENS_PER_CHUNK = 280  # 20 tokens reserved for page number metadata

# Summary
MAX_SHORT_TEXT_LENGTH = 1000  # Maximum length for short texts
MAX_MEDIUM_TEXT_LENGTH = 750000  # Maximum length for medium texts, used for large text condition
MAX_LARGE_TEXT_LENGTH = 3000000  # Minimum length for large texts
MEDIUM_TEXT_DIVISOR = 5000  # Divisor for calculating sentence count in medium texts
LARGE_TEXT_DIVISOR = 10000  # Divisor for calculating sentence count in large texts
VERY_LARGE_TEXT_DIVISOR = 30000  # Divisor for calculating sentence count in Very large texts
MAX_SENTENCES_MEDIUM = 50  # Maximum sentences count for medium texts
MAX_SENTENCES_LARGE = 100  # Maximum sentences count for large texts, ensures at least 100 sentences for very long texts
MAX_SENTENCES_VERY_LARGE = 200  # Maximum sentences count for large texts, ensures at least 100 sentences for very long texts


# Feedback
MAX_SHORT_FEEDBACK_LENGTH = 1000  # Maximum length for short texts
MAX_MEDIUM_FEEDBACK_LENGTH = 750000  # Maximum length for medium texts, used for large text condition
MAX_LARGE_FEEDBACK_LENGTH = 3000000  # Minimum length for large texts
MEDIUM_FEEDBACK_TEXT_DIVISOR = 5000  # Divisor for calculating sentence count in medium texts
LARGE_FEEDBACK_TEXT_DIVISOR = 10000  # Divisor for calculating sentence count in large texts
VERY_LARGE_FEEDBACK_TEXT_DIVISOR = 30000  # Divisor for calculating sentence count in Very large texts
MAX_FEEDBACK_SENTENCES_MEDIUM = 100  # Maximum sentences count for medium texts
MAX_FEEDBACK_SENTENCES_LARGE = 150  # Maximum sentences count for large texts, ensures at least 100 sentences for very long texts
MAX_FEEDBACK_SENTENCES_VERY_LARGE = 200  # Maximum sentences count for large texts, ensures at least 100 sentences for very long texts



# Default Input Parameters
DEFAULT_INFERENCE_MODEL = 'gpt-3.5-turbo'
DEFAULT_INFERENCE_API = 'openai'
DEFAULT_ALIGNMENT_MODEL = 'text-bison'
DEFAULT_ALIGNMENT_API = 'google'
DEFAULT_FEEDBACK_MODEL = 'text-bison'
DEFAULT_FEEDBACK_API = 'google'
DEFAULT_QUERY_DECOMPOSITION_API = 'google'
DEFAULT_QUERY_DECOMPOSITION_MODEL = 'text-bison'
DEFAULT_EVALUATION_TYPE = 'NE'

# BACKOFF VALUES
INITIAL_BACKOFF = 1
BACKOFF_MULTIPLIER = 2
NON_CONSTANT_BACKOFF_VALUES = 5
CONSTANT_BACKOFF = 60
MAX_RETRIES = 15
BACKOFF_JITTER = True

# Hybrid Search Reranker weights
DENSE_WEIGHT_HYBRID_SEARCH = 0.8


# Constants for logger

os.environ['GCP_MILVUS_URI'] = 'http://localhost:19530'
# Set it into env variables so logger file can acess


