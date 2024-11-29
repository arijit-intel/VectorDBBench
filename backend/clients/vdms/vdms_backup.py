from __future__ import annotations

import base64
import logging
import uuid
from copy import deepcopy,copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sized,
    Tuple,
    Type,
    Union,
    get_args,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.vdms import VDMS, VDMS_Client

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import vdms


DISTANCE_METRICS = Literal[
    "L2",  # Euclidean Distance
    "IP",  # Inner Product
]
AVAILABLE_DISTANCE_METRICS: List[DISTANCE_METRICS] = list(get_args(DISTANCE_METRICS))
ENGINES = Literal[
    "TileDBDense",  # TileDB Dense
    "TileDBSparse",  # TileDB Sparse
    "FaissFlat",  # FAISS IndexFlat
    "FaissIVFFlat",  # FAISS IndexIVFFlat
    "Flinng",  # FLINNG
]
AVAILABLE_ENGINES: List[ENGINES] = list(get_args(ENGINES))
DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_INSERT_BATCH_SIZE = 512
# Number of Documents to return.
DEFAULT_K = 3
# Number of Documents to fetch to pass to knn when filters applied.
DEFAULT_FETCH_K = DEFAULT_K * 5
DEFAULT_PROPERTIES = ["_distance", "id", "content"]
INVALID_DOC_METADATA_KEYS = ["_distance", "content", "blob"]
INVALID_METADATA_VALUE = ["Missing property", None, {}]  # type: List


logger = logging.getLogger(__name__)



import logging 
from contextlib import contextmanager
from typing import Any
from ..api import VectorDB, DBCaseConfig

log = logging.getLogger(__name__)


def VDMS_Client(host: str = "localhost", port: int = 55555) -> vdms.vdms:
    """VDMS client for the VDMS server.

    Args:
        host: IP or hostname of VDMS server
        port: Port to connect to VDMS server
    """
    try:
        import vdms
    except ImportError:
        raise ImportError(
            "Could not import vdms python package. "
            "Please install it with `pip install vdms."
        )
    try:
      client = vdms.vdms()
      client.connect(host, port)
    except:
      return None
    return client


class VDMS_bench(VectorDB):
     

    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: DBCaseConfig,
            drop_old: bool = False,

            **kwargs
        ):

        self.client = VDMS_Client("localhost", 55555)
        assert self.client is not None
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = 'dim1536'
        '''
        self.embedding_dimension = 768

        # Check required parameters
        self._client = VDMS_Client("localhost", 55555)
        assert self._client is not None
        self.similarity_search_engine = "FaissFlat"
        self.distance_strategy = "L2"
        self.embedding = None
        self._check_required_inputs(self.collection_name)

        # Update other parameters
        self.override_relevance_score_fn = None
        
        print("Start of init vdms")
        # Initialize collection
        self._collection_name = self.__add_set(
            self.collection_name,
            engine=self.similarity_search_engine,
            metric=self.distance_strategy,
        )
        '''
        print("Inside __init__")
        '''
        self.collection =  VDMS(
                client=self.client,
                embedding=HuggingFaceEmbeddings(),
                collection_name=self.collection_name,
                distance_strategy="L2",
                engine="Flinng",#"TileDBDense",#"FaissFlat",
        )
        '''
        '''
        if self.collection.count(self.collection_name) > 0:
          if self.collection._VDMS__delete(self.collection_name):
            print("delete successfull...")
        
          self.collection =  VDMS(
                  client=self.client,
                  embedding=HuggingFaceEmbeddings(),
                  collection_name=self.collection_name,
                  distance_strategy="L2",
                  engine="Flinng",#"TileDBDense",#"FaissFlat",
          )
        '''
        #print("collection in __init__: ",self.collection._collection_name)
        
        #yield
    
    @contextmanager
    def init(self) -> None:
        
        self.collection_new =  VDMS(
                  client=self.client,
                  embedding=HuggingFaceEmbeddings(),
                  collection_name=self.collection_name,
                  distance_strategy="L2",
                  engine="TileDBDense"#"Flinng",#"TileDBDense",#"FaissFlat",
        )
        
        '''
        self._collection_name = self.__add_set(
            self.collection_name,
            engine=self.similarity_search_engine,
            metric=self.distance_strategy,
        )
        '''
        #self.collection_new=self.collection
        #print("End of vdms init")
        print("collection in init: ",self.collection_new._collection_name)
        yield
        #self.client = None
        #self.collection_new = None#self.collection_new
    
    
    
    def ready_to_search(self) -> bool:
        pass

    def ready_to_load(self) -> bool:
        pass

    def optimize(self) -> None:
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        """Insert embeddings into the database.

        Args:
            embeddings(list[list[float]]): list of embeddings
            metadata(list[int]): list of metadata
            kwargs: other arguments

        Returns:
            (int, Exception): number of embeddings inserted and exception if any
        """
        #batch_size = 512
        #inserted_ids = []
        #print("Inside insert embeddings...")
        
        ids=[str(i) for i in metadata]
        metadata = [{"id": int(i)} for i in metadata]
        #metadata_text = [{"id": str(i)} for i in metadata] 
        #print("before inserting embedddings...")
        '''
        for start_idx in range(0, len(metadata_text), batch_size):
            end_idx = min(start_idx + batch_size, len(metadata_text))

            batch_texts = metadata_text[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            if metadata:
                batch_metadata = metadata[start_idx:end_idx]
        '''
        #print("Callable functions: ",dir(self.collection))
        print("collection in insert: ",self.collection_new._collection_name)
        print("Embedding dimension: ",len(embeddings[0]))
        if len(embeddings) > 0:
            print("inserting embeddings...")
            inserted_ids = self.collection_new._VDMS__from(texts=ids,embeddings=embeddings, ids=ids, metadatas=metadata,batch_size=512)
            print("length of inserted_ids: ",len(inserted_ids))
            print("Sample inserted ids: ",inserted_ids[:5])
            #print("length of all query: ",len(all_query))
            #print("Sample inserted ids: ",all_query[:5])        
        #query_t = embeddings[0]
        #returned_docs = self.collection_new.similarity_search_by_vector(query_t, k=1, filter=None)
        #print("search result: ",returned_docs)
        #all_responses = self.collection_new.query_collection_embeddings(collection_name="langchain-demo",query_embeddings=[embeddings[0]], n_results=1)
        #print(all_responses)
        print("Total collection size: ",self.collection_new.count(self.collection_new._collection_name))
        return len(inserted_ids), None
    
    
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Search embeddings from the database.
        Args:
            embedding(list[float]): embedding to search
            k(int): number of results to return
            kwargs: other arguments

        Returns:
            Dict {ids: list[list[int]], 
                    embedding: list[list[float]] 
                    distance: list[list[float]]}
        """
        
        print("collection in search: ",self.collection_new._collection_name)
        print("Total collection size: ",self.collection_new.count(self.collection_new._collection_name))
        #query_t = "322406"
        #returned_docs = self.collection_new.similarity_search(query_t, k=1, filter=None)
        #print("search result: ",returned_docs)
        print("Embedding dimension: ",len(query))
        all_responses = self.collection_new.query_collection_embeddings(collection_name=self.collection_new._collection_name,query_embeddings=[query], n_results=k, filter=filters)
        #print("Sample Response: ",all_responses[0:5])
        #print("Response len: ",len(all_responses))
        '''
        return_list = []
        for item in all_responses:
          try:
            return_list.append(int(item[0][0]["FindDescriptor"]["entities"]["id"]))
          except:
            return_list.append(-1)
            continue
        return return_list    
        '''
        return [int(item[0][0]["FindDescriptor"]["entities"][return_k]["id"]) for item in all_responses for return_k in range(len(item[0][0]["FindDescriptor"]["entities"]))]