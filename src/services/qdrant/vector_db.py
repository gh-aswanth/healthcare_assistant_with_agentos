import io
import json
from pathlib import Path

import aiofiles
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import ApiException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random,
)

from src.core.config import get_settings

settings = get_settings()


class Qdrant:
    def __init__(
        self,
        index_name: str,
        disable_indexing: bool = False,
    ):
        self.collection_name = index_name
        self.disable_indexing = disable_indexing
        self.dense_embedding = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL, dimensions=settings.OPENAI_EMBEDDING_DIMENSIONS
        )
        self.dimensions = settings.OPENAI_EMBEDDING_DIMENSIONS
        self.new_collection = False

        # Initialize Qdrant client configuration
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            port=None,
            prefer_grpc=False,
        )

        # Initialize Qdrant asynchronous client configuration
        self.async_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            port=None,
            prefer_grpc=False,
        )
        self.middle_ware = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        if self.disable_indexing:
            await self.set_indexing_threshold(indexing_threshold=0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.disable_indexing:
            await self.set_indexing_threshold(indexing_threshold=20000)
        return not exc_type

    def validate(self):
        if not self.dense_embedding:
            raise ValueError("Invalid Embedding Models")

    async def load_vector_data(self):
        async with aiofiles.open(
            Path(__file__).parent / "data/qdrant_data.json",
            "rb",
        ) as json_file:
            file_data = await json_file.read()
            json_data = json.load(io.BytesIO(file_data))
            vector_data = [Document(page_content=data.pop("full_summary"), metadata=data) for data in json_data]
            await self.middle_ware.aadd_documents(vector_data)
            self.new_collection = False

    async def load_index(self):
        """
        Asynchronously loads the index from the Qdrant vector database.

        This function validates the specified embedding models and creates the collection if it does not exist.

        Raises:
            ValueError: If the specified embedding models are invalid.
        """

        self.validate()
        index_data = [("department", False)]
        # Check if the collection exists
        if not await self.collection_exists(self.collection_name):
            # Create the collection if it does not exist
            await self.create_collection(index_data)
            self.new_collection = True

    def dense_vector_config(
        self,
    ) -> dict[str, models.VectorParams] | models.VectorParams:
        """
        Returns a configuration dictionary for dense vectors.

        The configuration includes the following settings:
        - The dimension size for dense vectors is set based on the model.
        - The distance metric used is cosine similarity.
        - Vectors are stored on disk to optimize RAM usage.

        Returns:
            A dictionary with a single key-value pair, where the key is "dense"
            and the value is an instance of VectorParams.
        """
        return {
            "dense": models.VectorParams(
                size=self.dimensions,  # Dimension size for dense vectors
                distance=models.Distance.COSINE,  # Distance metric for dense vectors
                on_disk=True,  # Store vectors on disk
            ),
        }

    @retry(
        retry=retry_if_exception_type(ApiException),
        stop=(stop_after_delay(10) | stop_after_attempt(5)),
        wait=wait_random(1, 10),
    )
    async def create_payload_index(self, index_datas: list[tuple[str, bool]]):
        """
        Asynchronously creates a payload index for tenant support in the Qdrant vector database.

        This function creates two payload indexes: one for the tenant_id field and one for the doc_id field.
        The tenant_id field is used to store the tenant ID associated with each document,
        and the doc_id field is used to store the document ID.
        """
        for field_name_pair in index_datas:
            field_name = field_name_pair[0]
            is_tenant = field_name_pair[1]
            await self.async_client.create_payload_index(
                collection_name=self.collection_name,
                field_name=f"metadata.{field_name}",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=is_tenant,
                ),
            )

    async def collection_exists(self, collection_name: str, **kwargs) -> bool:
        return await self.async_client.collection_exists(collection_name=collection_name, **kwargs)

    @retry(
        retry=retry_if_exception_type(ApiException),
        stop=(stop_after_delay(10) | stop_after_attempt(5)),
        wait=wait_random(1, 10),
    )
    async def create_collection(self, index_datas: list[tuple[str, bool]]):
        """
        Asynchronously creates a collection in the Qdrant vector database with specified configurations.

        This function sets up both dense and sparse vector indices, quantization, and optimization configurations.
        It also configures the HNSW graph parameters for approximate nearest neighbor search.

        Raises:
            ApiException: If the collection creation fails.
        """
        self.validate()

        is_collection_exists = await self.async_client.collection_exists(self.collection_name)
        if is_collection_exists:
            await self.create_payload_index(index_datas=index_datas)
            return

        quantization_config = models.BinaryQuantization(
            # Quantization configuration for binary quantization
            binary=models.BinaryQuantizationConfig(always_ram=True),  # Always store quantized data in RAM
        )
        await self.async_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.dense_vector_config(),
            quantization_config=quantization_config,
            optimizers_config=models.OptimizersConfigDiff(
                # Optimizer configuration for collection
                default_segment_number=settings.DEFAULT_SEGMENT_NUM,  # Default number of segments
                indexing_threshold=0,  # Threshold for indexing
            ),
            hnsw_config=models.HnswConfigDiff(
                m=settings.EDGES_PER_NODE,  # Number of edges per node, 0 disable global index creation
                ef_construct=settings.NEIGHBORS_NUM,  # Construction parameter for neighbors
                max_indexing_threads=settings.MAX_INDEX_THREAD,  # Max threads for indexing
                on_disk=True,
            ),
            on_disk_payload=True,
        )
        # search index creation
        await self.create_payload_index(index_datas=index_datas)

    @retry(
        retry=retry_if_exception_type(ApiException),
        stop=(stop_after_delay(10) | stop_after_attempt(5)),
        wait=wait_random(1, 10),
    )
    async def set_indexing_threshold(self, indexing_threshold: int):
        """
        Update the collection with the given configuration.

        This method updates the collection with the given configuration. It
        is used to update the collection configuration after it has been
        created.

        The current configuration is overridden with the new configuration.
        """
        await self.load_index()
        # Update the collection configuration
        await self.async_client.update_collection(
            collection_name=self.collection_name,
            # The configuration to update
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=settings.DEFAULT_SEGMENT_NUM,  # Default number of segments
                indexing_threshold=indexing_threshold,  # Threshold for indexing
            ),
        )

    @retry(
        retry=retry_if_exception_type(ApiException),
        stop=(stop_after_delay(10) | stop_after_attempt(5)),
        wait=wait_random(1, 10),
    )
    async def initialize_vectorstore(self) -> None:
        """
        Initialize the Qdrant VectorStore with the specified embedding models and
        retrieval mode.

        This method loads the index and creates a new Qdrant VectorStore with the
        specified embedding models and retrieval mode. The VectorStore is used
        to store and query the text embeddings in the Qdrant cluster.

        Returns:
            None
        """
        # Load the index before creating the VectorStore
        await self.load_index()

        # Create a new VectorStore with the specified embedding models and retrieval mode

        self.middle_ware = QdrantVectorStore(
            client=self.client,
            embedding=self.dense_embedding,
            collection_name=self.collection_name,
            validate_embeddings=False,  # skip embedding check call
            validate_collection_config=False,  # skip index validation call
            vector_name="dense",
        )

        if self.new_collection:
            await self.load_vector_data()
