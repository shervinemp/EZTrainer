import faiss
import torch
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Protocol, Callable
import uuid
import threading
import math
import pickle
import logging
import time

import weaviate
from weaviate.util import generate_uuid5
import weaviate.exceptions

# --- Configuration ---
# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "YourDataClass")
WEAVIATE_BATCH_SIZE = 100  # Optimal batch size for Weaviate writes
WEAVIATE_GET_BATCH_SIZE = 100  # Batch size for getting objects by ID

# FAISS Configuration
FAISS_VECTOR_DIM = 128
FAISS_INDEX_TYPE = (
    "IndexFlatL2"  # Consider IndexIVFFlat or IndexHNSWFlat for scalability
)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "faiss_metadata.pkl")
FAISS_TRAINING_DATA_SIZE = (
    1000  # Minimum data points for training some FAISS index types
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Protocol for Vector Database Interface ---
class VectorDatabase(Protocol):
    """
    Protocol defining the interface for a vector database backend.
    Implementations can be in-memory (short-term) or persistent (long-term).
    """

    def connect(self):
        """Establishes connection to the database."""
        ...

    def disconnect(self):
        """Disconnects from the database and performs cleanup (e.g., saving)."""
        ...

    def is_connected(self) -> bool:
        """Checks if the database connection is active."""
        ...

    def batch_search_vectors(
        self,
        query_vectors: List[np.ndarray],
        k: int,
        properties: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, Optional[np.ndarray], Dict, float]]]:
        """
        Performs a batch k-NN search.

        Returns:
            A list of lists of tuples. Each tuple is:
            (object_id: str, vector: Optional[np.ndarray], properties: Dict, distance: float)
            Note: vector might be None if the database backend doesn't store/return it in search.
        """
        ...

    def write_vectors(
        self,
        data_objects: List[Tuple[str, np.ndarray, Optional[Dict]]],
        is_create: bool = False,
    ):
        """
        Performs a batch write operation (create or update) for data objects
        with vectors.

        Args:
            data_objects: A list of tuples, where each tuple is
                          (object_id: str, vector: np.ndarray, properties: Optional[Dict]).
                          If is_create is True, object_id should be the desired UUID
                          or a base string for UUID generation.
                          If is_create is False, object_id must be an existing UUID.
                          properties can be None if only updating/creating vectors.
            is_create: If True, performs a batch create. If False, performs a batch update.
        """
        ...

    def get_objects_by_ids(
        self, object_ids: List[str], properties: Optional[List[str]] = None
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Retrieves data objects (including vectors and properties) by their IDs.

        Args:
            object_ids: A list of object UUID strings.
            properties: A list of properties to return for each object (optional).

        Returns:
            A list of tuples: (object_id: str, vector: np.ndarray, properties: Dict).
            Returns only objects that were found.
        """
        ...


# --- Persistent/Long-Term Database Implementation (Weaviate - Full) ---
class WeaviateDatabase:
    """
    Full Weaviate implementation of the VectorDatabase protocol.
    Represents a persistent, potentially out-of-memory database.
    """

    def __init__(self, url: str, class_name: str, api_key: Optional[str] = None):
        self.url = url
        self.class_name = class_name
        self.api_key = api_key
        self._client: Optional[weaviate.Client] = None

    def connect(self):
        """Establishes connection to the Weaviate instance."""
        try:
            auth_config = (
                weaviate.auth.AuthApiKey(api_key=self.api_key) if self.api_key else None
            )
            self._client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config,
                # Add other configurations like additional_headers, startup_timeout if needed
            )
            self._client.check_schema()
            try:
                self._client.schema.get(self.class_name)
                logger.info(
                    f"Connected to Weaviate at {self.url} for class '{self.class_name}'"
                )
            except weaviate.exceptions.UnexpectedStatusCodeException as e:
                logger.warning(
                    f"Class '{self.class_name}' not found in Weaviate schema. Please create it. Error: {e}"
                )
                logger.info(
                    f"Connected to Weaviate at {self.url}, but class '{self.class_name}' not found."
                )

        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            self._client = None

    def disconnect(self):
        """Clears the Weaviate client instance."""
        if self._client:
            logger.info("WeaviateDatabase client instance cleared.")
            self._client = None

    def is_connected(self) -> bool:
        if self._client:
            try:
                self._client.is_live()
                return True
            except Exception:
                return False
        return False

    def batch_search_vectors(
        self,
        query_vectors: List[np.ndarray],
        k: int,
        properties: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, Optional[np.ndarray], Dict, float]]]:
        if not self.is_connected() or not query_vectors:
            if not self.is_connected():
                logger.warning("Weaviate client not connected for batch search.")
            return [[] for _ in query_vectors] if query_vectors else []

        logger.debug(
            f"Weaviate batch_search_vectors called for {len(query_vectors)} queries."
        )
        start_time = time.perf_counter()

        try:
            near_vectors = [{"vector": vec.tolist()} for vec in query_vectors]

            query_builder = (
                self._client.query.get(
                    class_name=self.class_name,
                    properties=properties if properties is not None else [],
                )
                .with_multi_near_vector(
                    near_vectors=near_vectors,
                    distance=None,  # Optional: specify distance if needed
                    certainty=None,  # Optional: specify certainty if needed
                )
                .with_limit(k)
                .with_additional(["id", "vector", "distance"])
            )

            results = query_builder.do()

            batch_retrieved_data: List[
                List[Tuple[str, Optional[np.ndarray], Dict, float]]
            ] = []

            if (
                "data" in results
                and "Get" in results["data"]
                and self.class_name in results["data"]["Get"]
            ):
                retrieved_items_per_query = results["data"]["Get"][self.class_name]

                if len(retrieved_items_per_query) != len(query_vectors):
                    logger.warning(
                        f"Weaviate returned {len(retrieved_items_per_query)} result sets, but {len(query_vectors)} queries were sent."
                    )

                for query_results in retrieved_items_per_query:
                    current_query_data = []
                    if query_results:
                        for item in query_results:
                            if (
                                "_additional" in item
                                and "id" in item["_additional"]
                                and "vector" in item["_additional"]
                                and "distance" in item["_additional"]
                            ):
                                obj_id = item["_additional"]["id"]
                                vec_list = item["_additional"]["vector"]
                                dist = item["_additional"]["distance"]
                                props = (
                                    {prop: item.get(prop) for prop in properties}
                                    if properties
                                    else {}
                                )
                                current_query_data.append(
                                    (obj_id, np.array(vec_list), props, dist)
                                )
                            else:
                                logger.warning(
                                    f"Retrieved item missing required fields: {item}"
                                )

                    batch_retrieved_data.append(current_query_data)

            while len(batch_retrieved_data) < len(query_vectors):
                batch_retrieved_data.append([])

            end_time = time.perf_counter()
            logger.debug(
                f"Weaviate batch search completed in {end_time - start_time:.4f}s."
            )
            return batch_retrieved_data

        except Exception as e:
            logger.error(f"Error during Weaviate batch vector search: {e}")
            end_time = time.perf_counter()
            logger.debug(
                f"Weaviate batch search failed after {end_time - start_time:.4f}s."
            )
            return [[] for _ in query_vectors]

    def write_vectors(
        self,
        data_objects: List[Tuple[str, np.ndarray, Optional[Dict]]],
        is_create: bool = False,
    ):
        if not self.is_connected() or not data_objects:
            if not self.is_connected():
                logger.warning(
                    f"Weaviate client not connected for batch {'create' if is_create else 'update'}."
                )
            return

        operation_type = "create" if is_create else "update"
        logger.info(
            f"Starting Weaviate batch {operation_type} for {len(data_objects)} objects in class '{self.class_name}'..."
        )
        start_time = time.perf_counter()

        try:
            with self._client.batch as batch:
                batch.configure(batch_size=WEAVIATE_BATCH_SIZE)
                for obj_id_or_base, vector_np, properties_dict in data_objects:
                    if is_create:
                        try:
                            uuid.UUID(obj_id_or_base)
                            obj_uuid = obj_id_or_base
                        except ValueError:
                            obj_uuid = (
                                generate_uuid5(obj_id_or_base)
                                if isinstance(obj_id_or_base, str)
                                else str(uuid.uuid4())
                            )

                        batch.add_data_object(
                            data_object=(
                                properties_dict if properties_dict is not None else {}
                            ),
                            class_name=self.class_name,
                            uuid=obj_uuid,
                            vector=vector_np.tolist(),
                        )
                    else:
                        if not isinstance(
                            obj_id_or_base, str
                        ) or not self._is_valid_uuid(obj_id_or_base):
                            logger.warning(
                                f"Invalid UUID for update: {obj_id_or_base}. Skipping."
                            )
                            continue

                        batch.update_object(
                            uuid=obj_id_or_base,
                            class_name=self.class_name,
                            data_object=(
                                properties_dict if properties_dict is not None else {}
                            ),
                            vector=vector_np.tolist(),
                        )
            end_time = time.perf_counter()
            logger.info(
                f"Weaviate batch {operation_type} completed in {end_time - start_time:.4f}s."
            )
        except Exception as e:
            logger.error(f"Error during Weaviate batch {operation_type}: {e}")
            end_time = time.perf_counter()
            logger.error(
                f"Weaviate batch {operation_type} failed after {end_time - start_time:.4f}s."
            )

    def get_objects_by_ids(
        self, object_ids: List[str], properties: Optional[List[str]] = None
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        if not self.is_connected() or not object_ids:
            if not self.is_connected():
                logger.warning(
                    "Weaviate client not connected for getting objects by IDs."
                )
            return []

        logger.debug(f"Weaviate get_objects_by_ids called for {len(object_ids)} IDs.")
        start_time = time.perf_counter()

        retrieved_objects: List[Tuple[str, np.ndarray, Dict]] = []
        # Weaviate supports batch gets by ID
        try:
            # Split IDs into batches for the Get API
            for i in range(0, len(object_ids), WEAVIATE_GET_BATCH_SIZE):
                batch_ids = object_ids[i : i + WEAVIATE_GET_BATCH_SIZE]
                where_filter = {
                    "operator": "ContainsAny",
                    "path": ["id"],
                    "valueText": batch_ids,
                }
                results = (
                    self._client.query.get(
                        class_name=self.class_name,
                        properties=properties if properties is not None else [],
                    )
                    .with_where(where_filter)
                    .with_additional(["id", "vector"])
                    .do()
                )

                if (
                    "data" in results
                    and "Get" in results["data"]
                    and self.class_name in results["data"]["Get"]
                ):
                    for item in results["data"]["Get"][self.class_name]:
                        if (
                            "_additional" in item
                            and "id" in item["_additional"]
                            and "vector" in item["_additional"]
                        ):
                            obj_id = item["_additional"]["id"]
                            vec_list = item["_additional"]["vector"]
                            props = (
                                {prop: item.get(prop) for prop in properties}
                                if properties
                                else {}
                            )
                            retrieved_objects.append(
                                (obj_id, np.array(vec_list), props)
                            )
                        else:
                            logger.warning(
                                f"Retrieved item missing required fields (id or vector): {item}"
                            )

        except Exception as e:
            logger.error(f"Error during Weaviate get_objects_by_ids: {e}")

        end_time = time.perf_counter()
        logger.debug(
            f"Weaviate get_objects_by_ids completed in {end_time - start_time:.4f}s. Found {len(retrieved_objects)} objects."
        )
        return retrieved_objects

    def _is_valid_uuid(self, uuid_string: str) -> bool:
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False


# --- In-Memory/Short-Term Database Implementation (FAISS) ---
class FaissDatabase:
    """FAISS implementation."""

    def __init__(
        self,
        vector_dim: int,
        index_type: str = "IndexFlatL2",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._index: Optional[faiss.Index] = None
        self._metadata: Dict[int, Tuple[str, Dict, np.ndarray]] = {}
        self._object_id_to_faiss_id: Dict[str, int] = {}
        self._connected = False
        self._lock = threading.Lock()

    def connect(self):
        with self._lock:
            try:
                if self.index_path and os.path.exists(self.index_path):
                    self._index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded FAISS index from {self.index_path}")
                else:
                    if self.index_type == "IndexFlatL2":
                        self._index = faiss.IndexFlatL2(self.vector_dim)
                    elif self.index_type == "IndexIVFFlat":
                        nlist = 100
                        quantizer = faiss.IndexFlatL2(self.vector_dim)
                        self._index = faiss.IndexIVFFlat(
                            quantizer, self.vector_dim, nlist, faiss.METRIC_L2
                        )
                        logger.info(
                            f"Created new FAISS IndexIVFFlat (nlist={nlist}). Requires training."
                        )
                    elif self.index_type == "IndexHNSWFlat":
                        M = 32
                        self._index = faiss.IndexHNSWFlat(
                            self.vector_dim, M, faiss.METRIC_L2
                        )
                        self._index.hnsw.efConstruction = 40
                        logger.info(
                            f"Created new FAISS IndexHNSWFlat (M={M}, efConstruction={self._index.hnsw.efConstruction})."
                        )
                    else:
                        raise ValueError(
                            f"Unsupported FAISS index type: {self.index_type}"
                        )
                    logger.info(f"Created new FAISS index of type {self.index_type}")

                if self.metadata_path and os.path.exists(self.metadata_path):
                    try:
                        with open(self.metadata_path, "rb") as f:
                            loaded_metadata = pickle.load(f)
                            if (
                                isinstance(loaded_metadata, tuple)
                                and len(loaded_metadata) >= 2
                            ):
                                self._metadata, self._object_id_to_faiss_id, *_ = (
                                    loaded_metadata
                                )
                                self._metadata = {
                                    faiss_id: (
                                        obj_id,
                                        props if isinstance(props, dict) else {},
                                        vec if isinstance(vec, np.ndarray) else None,
                                    )
                                    for faiss_id, (
                                        obj_id,
                                        props,
                                        *vec_extra,
                                    ) in self._metadata.items()
                                    for vec in (vec_extra if vec_extra else [None])
                                }
                            else:
                                raise ValueError("Unknown metadata format.")
                        logger.info(f"Loaded metadata from {self.metadata_path}")
                    except Exception as e:
                        logger.error(
                            f"Error loading metadata from {self.metadata_path}: {e}. Starting with empty metadata."
                        )
                        self._metadata = {}
                        self._object_id_to_faiss_id = {}
                else:
                    self._metadata = {}
                    self._object_id_to_faiss_id = {}
                    logger.info("Starting with empty metadata.")

                if self._index and self._index.ntotal != len(self._metadata):
                    logger.warning(
                        f"FAISS index size ({self._index.ntotal}) and metadata size ({len(self._metadata)}) mismatch."
                    )

                self._connected = True
                logger.info("FAISS database connected.")
            except Exception as e:
                logger.error(f"Error connecting to FAISS database: {e}")
                self._index = None
                self._metadata = {}
                self._object_id_to_faiss_id = {}
                self._connected = False

    def disconnect(self):
        with self._lock:
            if self._connected:
                try:
                    if self.index_path and self._index:
                        faiss.write_index(self._index, self.index_path)
                        logger.info(f"Saved FAISS index to {self.index_path}")

                    if self.metadata_path:
                        with open(self.metadata_path, "wb") as f:
                            pickle.dump(
                                (
                                    self._metadata,
                                    self._object_id_to_faiss_id,
                                    self._index.ntotal if self._index else 0,
                                ),
                                f,
                            )
                        logger.info(f"Saved metadata to {self.metadata_path}")

                except Exception as e:
                    logger.error(f"Error saving FAISS database: {e}")
                finally:
                    self._index = None
                    self._metadata = {}
                    self._object_id_to_faiss_id = {}
                    self._connected = False
                    logger.info("FAISS database disconnected.")

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected and self._index is not None

    def batch_search_vectors(
        self,
        query_vectors: List[np.ndarray],
        k: int,
        properties: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, Optional[np.ndarray], Dict, float]]]:
        if not self.is_connected() or not query_vectors:
            if not self.is_connected():
                logger.warning("FAISS database not connected for batch search.")
            return [[] for _ in query_vectors] if query_vectors else []

        try:
            query_vectors_np = np.vstack(query_vectors).astype("float32")
        except ValueError as e:
            logger.error(f"Error stacking query vectors: {e}")
            return [[] for _ in query_vectors]

        logger.debug(
            f"FAISS batch_search_vectors called for {len(query_vectors)} queries."
        )
        start_time = time.perf_counter()

        with self._lock:
            try:
                if self._index.is_trained is False and hasattr(self._index, "train"):
                    logger.warning(
                        "FAISS index requires training but is not trained. Skipping search."
                    )
                    return [[] for _ in query_vectors]

                distances, faiss_indices = self._index.search(query_vectors_np, k)
                batch_retrieved_data: List[
                    List[Tuple[str, Optional[np.ndarray], Dict, float]]
                ] = []

                for i in range(len(query_vectors)):
                    current_query_data = []
                    for j in range(k):
                        faiss_id = faiss_indices[i, j]
                        if (
                            faiss_id != -1
                            and faiss_id in self._metadata
                            and np.isfinite(distances[i, j])
                        ):
                            metadata_tuple = self._metadata.get(faiss_id)
                            if metadata_tuple:
                                obj_id, stored_props, stored_vec = metadata_tuple
                                item_props = (
                                    {
                                        prop: stored_props.get(prop)
                                        for prop in properties
                                    }
                                    if properties
                                    else stored_props
                                )

                                if item_props.get("_deleted", False):
                                    logger.debug(f"Skipping deleted item {obj_id}")
                                    continue

                                retrieved_vec_np = (
                                    stored_vec if stored_vec is not None else None
                                )
                                dist = float(distances[i, j])
                                current_query_data.append(
                                    (obj_id, retrieved_vec_np, item_props, dist)
                                )

                    batch_retrieved_data.append(current_query_data)

                end_time = time.perf_counter()
                logger.debug(
                    f"FAISS batch search completed in {end_time - start_time:.4f}s."
                )
                return batch_retrieved_data

            except Exception as e:
                logger.error(f"Error during FAISS batch vector search: {e}")
                end_time = time.perf_counter()
                logger.debug(
                    f"FAISS batch search failed after {end_time - start_time:.4f}s."
                )
                return [[] for _ in query_vectors]

    def write_vectors(
        self,
        data_objects: List[Tuple[str, np.ndarray, Optional[Dict]]],
        is_create: bool = False,
    ):
        if not self.is_connected() or not data_objects:
            if not self.is_connected():
                logger.warning(
                    f"FAISS database not connected for batch {'create' if is_create else 'update'}."
                )
            return

        operation_type = "create" if is_create else "update"
        logger.info(
            f"Starting FAISS batch {operation_type} for {len(data_objects)} objects..."
        )
        start_time = time.perf_counter()

        vectors_to_add = []
        metadata_to_add = []
        faiss_ids_to_remove = []

        with self._lock:
            try:
                if not is_create:
                    for obj_id, _, _ in data_objects:
                        if obj_id in self._object_id_to_faiss_id:
                            faiss_id_to_remove = self._object_id_to_faiss_id[obj_id]
                            faiss_ids_to_remove.append(faiss_id_to_remove)
                            del self._object_id_to_faiss_id[obj_id]
                        else:
                            logger.warning(
                                f"Object ID {obj_id} not found for update. Skipping removal."
                            )

                    if faiss_ids_to_remove:
                        if hasattr(self._index, "remove_ids"):
                            faiss_ids_np = np.array(faiss_ids_to_remove, dtype="int64")
                            self._index.remove_ids(faiss_ids_np)
                            logger.info(
                                f"Removed {len(faiss_ids_to_remove)} old vectors from FAISS index."
                            )
                            for faiss_id in faiss_ids_to_remove:
                                if faiss_id in self._metadata:
                                    del self._metadata[faiss_id]
                        else:
                            logger.warning(
                                "FAISS index type does not support remove_ids. Marking as deleted in metadata."
                            )
                            if self.index_type == "IndexFlatL2":
                                for faiss_id in faiss_ids_to_remove:
                                    if faiss_id in self._metadata:
                                        obj_id, props, vec = self._metadata[faiss_id]
                                        self._metadata[faiss_id] = (
                                            obj_id,
                                            {**props, "_deleted": True},
                                            vec,
                                        )

                for obj_id, vector_np, properties_dict in data_objects:
                    if vector_np.shape[0] != self.vector_dim:
                        logger.error(
                            f"Vector dimension mismatch for object {obj_id}. Expected {self.vector_dim}, got {vector_np.shape[0]}. Skipping."
                        )
                        continue

                    vectors_to_add.append(vector_np)
                    current_properties = (
                        properties_dict if properties_dict is not None else {}
                    )
                    metadata_to_add.append((obj_id, current_properties, vector_np))

                if vectors_to_add:
                    vectors_to_add_np = np.vstack(vectors_to_add).astype("float32")

                    if self._index.is_trained is False and hasattr(
                        self._index, "train"
                    ):
                        logger.info("Training FAISS index before adding vectors...")
                        if (
                            len(vectors_to_add_np) < FAISS_TRAINING_DATA_SIZE
                            and self.index_type == "IndexIVFFlat"
                        ):
                            logger.warning(
                                f"Using small batch ({len(vectors_to_add_np)}) for training. Recommended: {FAISS_TRAINING_DATA_SIZE}+."
                            )
                        try:
                            self._index.train(vectors_to_add_np)
                            logger.info("Index training complete.")
                        except Exception as train_e:
                            logger.error(
                                f"Error during FAISS index training: {train_e}. Cannot add vectors."
                            )
                            return

                    initial_size = self._index.ntotal
                    self._index.add(vectors_to_add_np)

                    for i, (obj_id, props, vec_np) in enumerate(metadata_to_add):
                        faiss_id = initial_size + i
                        self._metadata[faiss_id] = (obj_id, props, vec_np)
                        self._object_id_to_faiss_id[obj_id] = faiss_id

                    logger.info(f"Added {len(vectors_to_add)} vectors to FAISS index.")

                logger.info(f"FAISS batch {operation_type} completed.")
                end_time = time.perf_counter()
                logger.info(
                    f"FAISS batch {operation_type} completed in {end_time - start_time:.4f}s."
                )

            except Exception as e:
                logger.error(f"Error during FAISS batch {operation_type}: {e}")
                end_time = time.perf_counter()
                logger.error(
                    f"FAISS batch {operation_type} failed after {end_time - start_time:.4f}s."
                )

    def get_objects_by_ids(
        self, object_ids: List[str], properties: Optional[List[str]] = None
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        if not self.is_connected() or not object_ids:
            if not self.is_connected():
                logger.warning(
                    "FAISS database not connected for getting objects by IDs."
                )
            return []

        logger.debug(f"FAISS get_objects_by_ids called for {len(object_ids)} IDs.")
        start_time = time.perf_counter()

        retrieved_objects: List[Tuple[str, np.ndarray, Dict]] = []
        with self._lock:
            try:
                for obj_id in object_ids:
                    if obj_id in self._object_id_to_faiss_id:
                        faiss_id = self._object_id_to_faiss_id[obj_id]
                        if faiss_id in self._metadata:
                            metadata_tuple = self._metadata[faiss_id]
                            obj_id_meta, stored_props, stored_vec = metadata_tuple

                            # Check if marked as deleted
                            if stored_props.get("_deleted", False):
                                logger.debug(f"Skipping deleted item {obj_id}")
                                continue

                            # Ensure vector is available in metadata for retrieval
                            if stored_vec is not None:
                                item_props = (
                                    {
                                        prop: stored_props.get(prop)
                                        for prop in properties
                                    }
                                    if properties
                                    else stored_props
                                )
                                retrieved_objects.append(
                                    (obj_id_meta, stored_vec, item_props)
                                )
                            else:
                                logger.warning(
                                    f"Vector not found in metadata for object ID {obj_id}."
                                )
                        else:
                            logger.warning(
                                f"Metadata not found for FAISS ID corresponding to object ID {obj_id}."
                            )
                    else:
                        logger.warning(
                            f"Object ID {obj_id} not found in FAISS object ID mapping."
                        )

            except Exception as e:
                logger.error(f"Error during FAISS get_objects_by_ids: {e}")

        end_time = time.perf_counter()
        logger.debug(
            f"FAISS get_objects_by_ids completed in {end_time - start_time:.4f}s. Found {len(retrieved_objects)} objects."
        )
        return retrieved_objects


# --- RAG Engine using Generic Vector Databases ---
class RagEngine:
    """
    A RAG engine that uses separate short-term and long-term VectorDatabase backends.
    Processes batch query tensors, retrieves data, simulates model processing,
    calculates distance-based value, writes vectors, and supports data transfer.
    """

    def __init__(self, short_term_db: VectorDatabase, long_term_db: VectorDatabase):
        """
        Initializes the RagEngine with short-term and long-term database instances.

        Args:
            short_term_db: An instance of a VectorDatabase for short-term storage (e.g., FAISS).
            long_term_db: An instance of a VectorDatabase for long-term storage (e.g., Weaviate).
        """
        self.short_term_db = short_term_db
        self.long_term_db = long_term_db

        if not self.short_term_db.is_connected():
            logger.warning(
                "Short-term database not connected on RagEngine initialization. Call connect()."
            )
        if not self.long_term_db.is_connected():
            logger.warning(
                "Long-term database not connected on RagEngine initialization. Call connect()."
            )

    def connect_databases(self):
        """Connects to both short-term and long-term databases."""
        self.short_term_db.connect()
        self.long_term_db.connect()

    def disconnect_databases(self):
        """Disconnects from both short-term and long-term databases."""
        self.short_term_db.disconnect()
        self.long_term_db.disconnect()

    def process_queries(
        self,
        query_tensors: torch.Tensor,
        k: int,
        layer_name: str,
        std_dev: float,
        properties_to_retrieve: Optional[List[str]] = None,
        use_long_term_db: bool = False,
        update_retrieved_vectors: bool = True,
        create_new_vectors: bool = False,
    ):
        """
        Processes a batch of query tensors, retrieves similar vectors from the
        specified database, simulates model processing, calculates distance-based value,
        and potentially writes vectors.

        Args:
            query_tensors: A PyTorch tensor of shape (batch_size, vector_dimension).
                           Requires grad can be True if these are part of a computation graph.
            k: The number of similar items to retrieve for each query in the batch.
            layer_name: The name of the model layer that generated the query_tensors.
            std_dev: The standard deviation for calculating a value based on distance
                     (assuming query is mean=0 in a Gaussian distribution).
            properties_to_retrieve: List of properties to fetch for similar objects.
            use_long_term_db: If True, search in the long-term database; otherwise, use short-term.
            update_retrieved_vectors: If True, attempts to update the vectors of the
                                      retrieved objects based on model output.
            create_new_vectors: If True, attempts to create new objects with vectors
                                based on model output.
        """
        db_to_use = self.long_term_db if use_long_term_db else self.short_term_db
        db_name = "Long-term (Weaviate)" if use_long_term_db else "Short-term (FAISS)"

        if not db_to_use.is_connected():
            logger.error(f"Cannot process queries: {db_name} database not connected.")
            return

        if query_tensors.ndim != 2:
            logger.error(
                f"Expected query_tensors to have 2 dimensions (batch_size, vector_dim), but got {query_tensors.ndim}"
            )
            return

        batch_size, vector_dim = query_tensors.shape
        logger.info(
            f"Processing batch of {batch_size} query tensors from layer '{layer_name}' using {db_name} database."
        )

        # --- Step 1 & 2: Batch Search ---
        query_vectors_np = [vec.detach().numpy() for vec in query_tensors]
        logger.info(f"Searching {db_name} for top {k} similar items...")

        batch_retrieved_data = db_to_use.batch_search_vectors(
            query_vectors=query_vectors_np, k=k, properties=properties_to_retrieve
        )

        all_retrieved_data = [
            item for query_results in batch_retrieved_data for item in query_results
        ]

        retrieved_info = []
        for item in all_retrieved_data:
            obj_id, vec_np, props, dist = item
            if obj_id is not None:
                retrieved_info.append(
                    {
                        "object_id": obj_id,
                        "vector": vec_np,
                        "properties": props,
                        "distance": dist,
                    }
                )

        if not retrieved_info and not create_new_vectors:
            logger.info(
                f"No similar items retrieved from {db_name} and create_new_vectors is False. No tensors to process."
            )
            return

        logger.info(
            f"Retrieved data for {len(retrieved_info)} objects from {db_name} (including distances)."
        )

        # --- Calculate Value based on Distance (Gaussian PDF) ---
        calculated_values = {}
        if retrieved_info:
            logger.info(
                f"Calculating value for each retrieved item based on distance and std_dev ({std_dev:.4f})..."
            )
            if std_dev <= 0:
                logger.warning(
                    "Standard deviation must be positive to calculate Gaussian PDF. Skipping value calculation."
                )
            else:
                for item in retrieved_info:
                    dist = item["distance"]
                    try:
                        exponent = -((dist**2) / (2 * std_dev**2))
                        pdf_value = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(
                            exponent
                        )
                        calculated_values[item["object_id"]] = pdf_value
                        logger.debug(
                            f"  Object ID: {item['object_id']}, Distance: {dist:.4f}, Calculated Value (PDF): {pdf_value:.6f}"
                        )
                    except OverflowError:
                        logger.warning(
                            f"  Overflow calculating PDF for distance {dist:.4f}. Value set to 0."
                        )
                        calculated_values[item["object_id"]] = 0.0
                    except Exception as e:
                        logger.error(
                            f"  Error calculating PDF for distance {dist:.4f}: {e}. Value set to None."
                        )
                        calculated_values[item["object_id"]] = None

        else:
            logger.info("No retrieved items to calculate values for.")

        # --- Step 3: Prepare tensors for Model Processing ---
        tensors_to_process = None
        retrieved_vectors_for_update = [
            item["vector"] for item in retrieved_info if item["vector"] is not None
        ]

        if update_retrieved_vectors and retrieved_vectors_for_update:
            try:
                tensors_to_process = torch.tensor(
                    np.stack(retrieved_vectors_for_update),
                    dtype=torch.float32,
                    requires_grad=True,
                )
                logger.info(
                    f"Using retrieved vectors for model processing (update path). Shape: {tensors_to_process.shape}"
                )
            except Exception as e:
                logger.error(
                    f"Error converting retrieved vectors for update to tensors: {e}"
                )
                return
        elif create_new_vectors:
            tensors_to_process = query_tensors
            logger.info(
                "Using original query tensors for model processing to create new vectors..."
            )
        else:
            logger.info(
                "No tensors available for model processing based on selected options."
            )
            return

        logger.debug(f"Tensor requires gradient: {tensors_to_process.requires_grad}")

        # --- Step 4: Conceptual Model Integration and Backpropagation ---
        processed_tensors = None
        try:
            input_vector_dim = tensors_to_process.shape[1]
            dummy_layer = torch.nn.Linear(input_vector_dim, input_vector_dim)
            processed_tensors = dummy_layer(tensors_to_process)

            target_tensors = torch.randn_like(processed_tensors)
            loss_function = torch.nn.MSELoss()
            loss = loss_function(processed_tensors, target_tensors)

            logger.info(f"Simulated model processing and computed loss: {loss.item()}")
            loss.backward()
            logger.info("Performed backpropagation.")

            if tensors_to_process.grad is not None:
                logger.debug(
                    f"Gradients computed for input tensors to model. Shape: {tensors_to_process.grad.shape}"
                )

        except Exception as e:
            logger.error(
                f"Error during conceptual model processing or backpropagation: {e}"
            )
            processed_tensors = None

        # --- Step 5: Write Vectors Based on Model Output ---
        if processed_tensors is not None:
            logger.info(
                "Preparing to write vectors to the database based on processed tensors..."
            )

            output_vectors_np = processed_tensors.detach().numpy()
            write_objects_list = []

            if update_retrieved_vectors and retrieved_info:
                original_retrieved_object_ids_for_update = [
                    item["object_id"]
                    for item in retrieved_info
                    if item["vector"] is not None
                ]

                if len(output_vectors_np) == len(
                    original_retrieved_object_ids_for_update
                ):
                    logger.info(
                        f"Preparing to update {len(original_retrieved_object_ids_for_update)} retrieved objects in {db_name}..."
                    )
                    for i, obj_id in enumerate(
                        original_retrieved_object_ids_for_update
                    ):
                        write_objects_list.append((obj_id, output_vectors_np[i], None))
                    if write_objects_list:
                        db_to_use.write_vectors(write_objects_list, is_create=False)
                else:
                    logger.warning(
                        f"Mismatch between processed vectors and retrieved objects for update in {db_name}. Cannot update."
                    )

            elif create_new_vectors:
                if len(output_vectors_np) == batch_size:
                    logger.info(
                        f"Preparing to create {len(output_vectors_np)} new objects in {db_name}..."
                    )
                    for vector_np in output_vectors_np:
                        new_uuid = str(uuid.uuid4())
                        write_objects_list.append((new_uuid, vector_np, None))
                    if write_objects_list:
                        db_to_use.write_vectors(write_objects_list, is_create=True)
                else:
                    logger.warning(
                        f"Mismatch between original query tensors and processed vectors for creation in {db_name}. Cannot create."
                    )

            else:
                logger.info(
                    "Neither update_retrieved_vectors nor create_new_vectors is True. No vectors written."
                )

    def transfer_data(
        self,
        source_db: VectorDatabase,
        destination_db: VectorDatabase,
        object_ids: List[str],
        vector_transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        properties_to_retrieve: Optional[List[str]] = None,
    ):
        """
        Transfers data objects (vectors and properties) from a source database to a destination database.

        Args:
            source_db: The database to transfer data FROM.
            destination_db: The database to transfer data TO.
            object_ids: A list of object IDs to transfer.
            vector_transform_fn: An optional function to apply to vectors during transfer.
                                 It should accept a NumPy array of vectors (N, D) and return
                                 a NumPy array of transformed vectors (N, D').
                                 Assumes D' is compatible with the destination database.
            properties_to_retrieve: Optional list of properties to retrieve and transfer.
        """
        if not source_db.is_connected():
            logger.error("Source database not connected. Cannot transfer data.")
            return
        if not destination_db.is_connected():
            logger.error("Destination database not connected. Cannot transfer data.")
            return
        if not object_ids:
            logger.warning("No object IDs provided for transfer.")
            return

        logger.info(f"Starting data transfer for {len(object_ids)} objects.")

        # Step 1: Retrieve data from the source database
        logger.info(f"Retrieving {len(object_ids)} objects from source database...")
        retrieved_objects = source_db.get_objects_by_ids(
            object_ids, properties=properties_to_retrieve
        )

        if not retrieved_objects:
            logger.warning(
                "No objects found in the source database for the given IDs. Transfer complete (0 objects transferred)."
            )
            return

        logger.info(
            f"Successfully retrieved {len(retrieved_objects)} objects from source."
        )

        # Prepare data for writing
        data_to_write: List[Tuple[str, np.ndarray, Optional[Dict]]] = []
        vectors_to_transform = []
        original_ids_for_transform = (
            []
        )  # Keep track of original IDs corresponding to vectors

        for obj_id, vector_np, properties_dict in retrieved_objects:
            if vector_np is not None:  # Only transfer objects with vectors
                vectors_to_transform.append(vector_np)
                original_ids_for_transform.append(obj_id)
                # Store properties for writing (will be paired with transformed vectors later)
                data_to_write.append(
                    (obj_id, np.zeros(0), properties_dict)
                )  # Placeholder vector

        if not vectors_to_transform:
            logger.warning(
                "No objects with vectors found in the source database for the given IDs. Transfer complete (0 objects transferred)."
            )
            return

        # Step 2: Apply vector transformation if function is provided
        transformed_vectors_np = np.vstack(vectors_to_transform).astype(
            "float32"
        )  # Stack for batch transformation
        if vector_transform_fn:
            logger.info(
                f"Applying vector transformation to {len(vectors_to_transform)} vectors..."
            )
            try:
                # Ensure the transform function handles NumPy arrays and returns NumPy arrays
                transformed_vectors_np = vector_transform_fn(transformed_vectors_np)
                if not isinstance(transformed_vectors_np, np.ndarray):
                    raise TypeError(
                        "Vector transformation function must return a NumPy array."
                    )
                logger.info(
                    f"Transformation complete. Transformed vectors shape: {transformed_vectors_np.shape}"
                )
            except Exception as e:
                logger.error(
                    f"Error applying vector transformation: {e}. Skipping transformation."
                )
                # If transformation fails, proceed with original vectors or stop?
                # For robustness, let's stop the transfer if transformation fails.
                return

        # Update data_to_write with the transformed vectors
        if len(transformed_vectors_np) != len(data_to_write):
            logger.error(
                f"Mismatch between number of transformed vectors ({len(transformed_vectors_np)}) and objects to write ({len(data_to_write)}). Cannot complete transfer."
            )
            return

        # Reconstruct data_to_write with transformed vectors, maintaining original IDs and properties
        final_data_to_write: List[Tuple[str, np.ndarray, Optional[Dict]]] = []
        # Assuming the order of transformed_vectors_np matches the order of original_ids_for_transform
        for i, obj_id in enumerate(original_ids_for_transform):
            # Find the original properties for this object ID
            original_properties = next(
                (item[2] for item in retrieved_objects if item[0] == obj_id), None
            )
            final_data_to_write.append(
                (obj_id, transformed_vectors_np[i], original_properties)
            )

        # Step 3: Write data to the destination database
        logger.info(
            f"Writing {len(final_data_to_write)} objects to destination database..."
        )
        # When transferring, we are creating new objects in the destination database
        destination_db.write_vectors(final_data_to_write, is_create=True)

        logger.info("Data transfer process completed.")


# --- Example Usage ---
if __name__ == "__main__":
    example_k = 5
    properties_to_retrieve = ["text_content", "title"]
    example_layer_name = "encoder_output_layer"
    example_std_dev = 0.1

    # --- Instantiate Database Backends ---
    faiss_db = FaissDatabase(
        vector_dim=FAISS_VECTOR_DIM,
        index_type=FAISS_INDEX_TYPE,
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
    )

    weaviate_db = WeaviateDatabase(
        url=WEAVIATE_URL, class_name=WEAVIATE_CLASS_NAME, api_key=WEAVIATE_API_KEY
    )

    # --- Instantiate RagEngine with both databases ---
    rag_engine = RagEngine(short_term_db=faiss_db, long_term_db=weaviate_db)

    # Use a try...finally block to ensure disconnection
    try:
        rag_engine.connect_databases()

        # --- Example of using process_queries (e.g., with short-term FAISS) ---
        dummy_vector_dim = FAISS_VECTOR_DIM
        dummy_batch_size = 3
        example_query_tensors = torch.randn(
            dummy_batch_size, dummy_vector_dim, requires_grad=True
        )

        # Run a query process (e.g., create new data in FAISS)
        logger.info("\n--- Running Query Process (Creating data in FAISS) ---")
        if rag_engine.short_term_db.is_connected():
            rag_engine.process_queries(
                query_tensors=example_query_tensors,
                k=example_k,
                layer_name=example_layer_name,
                std_dev=example_std_dev,
                properties_to_retrieve=properties_to_retrieve,
                use_long_term_db=False,  # Use short-term FAISS
                update_retrieved_vectors=False,
                create_new_vectors=True,  # Create new objects in FAISS
            )
        else:
            logger.warning(
                "Short-term database not connected. Skipping query process example."
            )

        # --- Example of Data Transfer (FAISS to Weaviate) ---
        # Assuming the previous step created some objects in FAISS.
        # You would need to get the IDs of the objects you want to transfer.
        # For this example, let's assume some dummy IDs or retrieve recent ones from FAISS metadata
        # (Note: Retrieving recent IDs from FAISS metadata is not a standard feature;
        # you'd typically manage IDs externally or based on your data ingestion).
        # Let's simulate having a list of IDs from FAISS.

        # To get IDs from FAISS for transfer:
        # You would need to store the IDs generated during the create process.
        # For demonstration, let's add a simple way to get all IDs from FAISS metadata.
        def get_all_faiss_object_ids(faiss_db_instance: FaissDatabase) -> List[str]:
            if not faiss_db_instance.is_connected():
                return []
            with faiss_db_instance._lock:
                # Filter out deleted items if necessary
                return [
                    obj_id
                    for faiss_id, (
                        obj_id,
                        props,
                        vec,
                    ) in faiss_db_instance._metadata.items()
                    if not props.get("_deleted", False)
                ]

        # Get some IDs from FAISS after the create operation
        faiss_object_ids_to_transfer = get_all_faiss_object_ids(faiss_db)
        logger.info(
            f"\nFound {len(faiss_object_ids_to_transfer)} objects in FAISS for potential transfer."
        )

        if faiss_object_ids_to_transfer and rag_engine.long_term_db.is_connected():
            logger.info("\n--- Running Data Transfer (FAISS to Weaviate) ---")

            # Define an optional vector transformation function (e.g., a simple scaling)
            def scale_vector_transform(vectors_np: np.ndarray) -> np.ndarray:
                logger.info(
                    f"Applying scaling transformation to {vectors_np.shape[0]} vectors..."
                )
                return vectors_np * 0.5  # Example: scale vectors by 0.5

            rag_engine.transfer_data(
                source_db=rag_engine.short_term_db,  # Transfer FROM FAISS
                destination_db=rag_engine.long_term_db,  # Transfer TO Weaviate
                object_ids=faiss_object_ids_to_transfer,  # IDs to transfer
                vector_transform_fn=None,  # Set to scale_vector_transform to apply transformation
                properties_to_retrieve=properties_to_retrieve,  # Properties to transfer
            )
        elif not rag_engine.long_term_db.is_connected():
            logger.warning(
                "Long-term database not connected. Skipping data transfer example."
            )
        else:
            logger.info("No objects found in FAISS to transfer.")

        # --- Example of Data Transfer (Weaviate to FAISS) ---
        # This would require having data already in Weaviate.
        # Let's simulate having some dummy Weaviate IDs for transfer.
        # In a real scenario, you would retrieve these IDs from Weaviate.
        dummy_weaviate_ids_to_transfer = [
            str(uuid.uuid4()) for _ in range(2)
        ]  # Dummy Weaviate IDs

        if dummy_weaviate_ids_to_transfer and rag_engine.short_term_db.is_connected():
            logger.info("\n--- Running Data Transfer (Weaviate to FAISS) ---")

            # Define a different optional vector transformation function (e.g., adding noise)
            def add_noise_vector_transform(vectors_np: np.ndarray) -> np.ndarray:
                logger.info(
                    f"Applying noise transformation to {vectors_np.shape[0]} vectors..."
                )
                noise = (
                    np.random.randn(*vectors_np.shape).astype("float32") * 0.01
                )  # Small noise
                return vectors_np + noise

            # Note: For this example to work, the dummy_weaviate_ids_to_transfer
            # must actually exist in your Weaviate instance.
            rag_engine.transfer_data(
                source_db=rag_engine.long_term_db,  # Transfer FROM Weaviate
                destination_db=rag_engine.short_term_db,  # Transfer TO FAISS
                object_ids=dummy_weaviate_ids_to_transfer,  # IDs to transfer
                vector_transform_fn=add_noise_vector_transform,  # Apply noise transformation
                properties_to_retrieve=properties_to_retrieve,  # Properties to transfer
            )
        elif not rag_engine.short_term_db.is_connected():
            logger.warning(
                "Short-term database not connected. Skipping data transfer example."
            )
        else:
            logger.info("No dummy Weaviate IDs provided for transfer.")

    finally:
        # Ensure disconnect is called even if errors occur
        rag_engine.disconnect_databases()
