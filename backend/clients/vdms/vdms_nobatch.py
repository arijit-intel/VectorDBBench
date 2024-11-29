import time
import vdms
import logging
from contextlib import contextmanager
import numpy as np
import random
from ..api import VectorDB, DBCaseConfig
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
)

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
DEFAULT_INSERT_BATCH_SIZE = 32
# Number of Documents to return.
DEFAULT_K = 3
# Number of Documents to fetch to pass to knn when filters applied.
DEFAULT_FETCH_K = DEFAULT_K * 5
DEFAULT_PROPERTIES = ["_distance", "id", "content"]
INVALID_DOC_METADATA_KEYS = ["_distance", "content", "blob"]
INVALID_METADATA_VALUE = ["Missing property", None, {}]  # type: List

log = logging.getLogger(__name__)


class VDMS_bench(VectorDB):
    """VDMS client for VectorDB.
    To set up VDMS in docker, see https://github.com/IntelLabs/vdms/wiki/Docker
    or the instructions in tests/test_vdms.py
    """

    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: DBCaseConfig,
            drop_old: bool = False,
            **kwargs):

        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = self.db_config["db_label"]#'example2'

        client = vdms.vdms()
        client.connect(host=self.db_config["host"],
                       port=self.db_config["port"])
        assert client.connected
        # if drop_old:
        #     try:
        #         client.reset() # Reset the database
        #     except:
        #         drop_old = False
        #         log.info(f"VDMS client drop_old collection: {self.collection_name}")

    @contextmanager
    def init(self) -> None:
        """ create and destroy connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        #create connection
        self.client = vdms.vdms()
        self.client.connect(host=self.db_config["host"],
                            port=self.db_config["port"])

        self.collection_name = self.db_config["db_label"]#'example2'
        self.distance_strategy = self.db_config["distance_strategy"]#"L2"
        self.engine = self.db_config["engine"]#"FaissFlat"

        # self.similarity_search_engine =
        # self.collection = self.add_set(collection_name, engine=engine, metric=distance_strategy)
        yield
        self.client = None
        # self.collection = None

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
        batch_size: int=DEFAULT_INSERT_BATCH_SIZE,#batch_size
        #epoch_count: int=0,
        **kwargs: Any,) -> (int, Exception):
        """Insert embeddings into the database.

        Args:
            embeddings(list[list[float]]): list of embeddings
            metadata(list[int]): list of metadata
            kwargs: other arguments

        Returns:
            (int, Exception): number of embeddings inserted and exception if any
        """
        # ids=[str(i) for i in metadata]
        color_list = ['red', 'blue', 'green']
        #metadatas = [{"id": int(i),"color": random.choice(color_list)} for i in metadata]
        #metadatas = [{"id": int(i)} for i in metadata]
        metadatas = [{"id": int(i),"date": int(i)} for i in metadata]
        num_inserted = 0
        print("Insert batch size: ",batch_size)
        #print("Insert batch size: ",epoch_count)
        if len(embeddings) > 0:
            # Add Descriptor Set if not present
            #start_time = time.time()
            self.add_set(dim=len(embeddings[0]))
            #end_time = time.time()
            #print("time for set add: ",round(end_time-start_time,2))
            #num_inserted = self.batch_insertion_wo_metadata(batch_size, embeddings)
            num_inserted = self.batch_insertion(batch_size, metadatas, embeddings)
        return num_inserted, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,) -> dict:
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
        # if filters:
        #     # assumes benchmark test filters of format: {'metadata': '>=10000', 'id': 10000}
        #     id_value = filters.get("id")
        #     results = self.collection.query(query_embeddings=query, n_results=k,
        #                                         where={"id": {"$gt": id_value}})
        #     #return list of id's in results
        #     return [int(i) for i in results.get('ids')[0]]
        # results = self.collection.query(query_embeddings=query, n_results=k)
        # return [int(i) for i in results.get('ids')[0]]
        #constraint = None
        #if filters:
        #    print("Inside search filters...")
        #    constraint = {"id": [">", filters.get("id")]}
        #constraint = {"id": ["==", [354737,114802]],"color": ["==",["green","red"]]}
        start_timestamp = 6000#int(datetime(2024, 5, 1).timestamp())
        end_timestamp = 7000#int(datetime(2024, 9, 1).timestamp())
        constraint = {"date": [">=",start_timestamp,"<=",end_timestamp]}
        response, _ = self.get_descriptor_response(
                "FindDescriptor",
                self.collection_name,
                k_neighbors=k,
                constraints=constraint,
                results={"list": ["id"]},
                # normalize_distance=normalize_distance,
                query_embedding=query,
            )
        #print(response)
        #ids = [ent["id"] for ent in response[0]["FindDescriptor"]["entities"]]
        #print("Search id: ",ids)
        try:
            ids = [ent["id"] for ent in response[0]["FindDescriptor"]["entities"]]
        except:
            ids = [-1]
        #print("Search id: ",ids)
        return ids

    def add_set(
        self,
        # collection_name: str,
        # engine: ENGINES = "FaissFlat",
        # metric: DISTANCE_METRICS = "L2",
        dim
        ) -> str:
        query = add_descriptorset(
            "AddDescriptorSet",
            self.collection_name,
            dim,
            engine=self.engine,
            metric=self.distance_strategy,
        )

        response, _ = self.__run_vdms_query([query])

        if "FailedCommand" in response[0]:
            raise ValueError(f"Failed to add collection {self.collection_name}")

        return self.collection_name

    def get_descriptor_response(
        self,
        command_str: str,
        setname: str,
        k_neighbors: int = DEFAULT_K,
        fetch_k: int = None,
        constraints: Optional[dict] = None,
        results: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
        normalize_distance: bool = False,) -> Tuple[List[Dict[str, Any]], List]:
        if fetch_k is None:
            fetch_k = k_neighbors * 10

        all_blobs: List[Any] = []
        ##time it
        #start_blob = time.perf_counter()
        blob = embedding2bytes(query_embedding)
        #print("embedding2bytes time: ",time.perf_counter()-start_blob)
        if blob is not None:
            all_blobs.append(blob)

        #print("K: ",k_neighbors)
        k_neighbors = 100
        #print("K: ",k_neighbors)
        response, response_array, max_dist = self.get_k_candidates(
                setname, k_neighbors, results, all_blobs, normalize=normalize_distance, constraints=constraints
            )
        '''    
        if constraints is None:
            # K results returned
            #start_knn = time.perf_counter()
            response, response_array, max_dist = self.get_k_candidates(
                setname, k_neighbors, results, all_blobs, normalize=normalize_distance, constraints=constraints
            )
            #print("get_k_candidates time: ",time.perf_counter()-start_knn)
        else:
            if results is None:
                results = {"list": ["id"]}
            elif "list" not in results:
                results["list"] = ["id"]
            elif "id" not in results["list"]:
                results["list"].append("id")

            # (1) Find docs satisfy constraints
            query = add_descriptor(
                command_str,
                setname,
                constraints=constraints,
                results=results,
            )
            ##time it
            response, response_array = self.__run_vdms_query([query])
            ids_of_interest = [
                ent["id"] for ent in response[0][command_str]["entities"]
            ]

            # (2) Find top fetch_k results
            response, response_array, max_dist = self.get_k_candidates(
                setname, fetch_k, results, all_blobs, normalize=normalize_distance
            )

            # (3) Intersection of (1) & (2) using ids
            new_entities: List[Dict] = []
            for ent in response[0][command_str]["entities"]:
                if ent["id"] in ids_of_interest:
                    new_entities.append(ent)
                if len(new_entities) == k_neighbors:
                    break
            response[0][command_str]["entities"] = new_entities
            response[0][command_str]["returned"] = len(new_entities)
            #print("response: ",response)
            if len(new_entities) < k_neighbors:
                p_str = f"Returned items < k_neighbors ({k_neighbors}); Try increasing fetch_k to > {fetch_k}"
                print(p_str)  # noqa: T201

        '''
        if normalize_distance:
            max_dist = 1.0 if max_dist == 0 else max_dist
            for ent_idx, ent in enumerate(response[0][command_str]["entities"]):
                ent["_distance"] = ent["_distance"] / max_dist
                response[0][command_str]["entities"][ent_idx]["_distance"] = ent[
                    "_distance"
                ]

        return response, None

    def get_k_candidates(
        self,
        setname: str,
        fetch_k: Optional[int],
        results: Optional[Dict[str, Any]] = None,
        all_blobs: Optional[List] = None,
        normalize: Optional[bool] = False,
        constraints: Optional[dict] = None,
    ) -> Tuple[List[Dict[str, Any]], List, float]:
        max_dist = 1
        command_str = "FindDescriptor"
        query = add_descriptor(
            command_str,
            setname,
            k_neighbors=fetch_k,
            results=results,
            constraints=constraints
        )
        #print("Size of k in knn: ",fetch_k)
        response, response_array = self.__run_vdms_query([query], all_blobs)

        if normalize:
            max_dist = response[0][command_str]["entities"][-1]["_distance"]

        return response, response_array, max_dist

    def count(self, collection_name: str) -> int:
        all_queries: List[Any] = []
        all_blobs: List[Any] = []

        results = {"count": "", "list": ["id"]}  # collection_properties}
        query = add_descriptor(
            "FindDescriptor",
            collection_name,
            label=None,
            ref=None,
            props=None,
            link=None,
            k_neighbors=None,
            constraints=None,
            results=results,
        )

        all_queries.append(query)

        response, response_array = self.__run_vdms_query(all_queries, all_blobs)
        return response[0]["FindDescriptor"]["returned"]

    # def __get_add_query(
    #     self,
    #     collection_name: str,
    #     metadata: Optional[Any] = None,
    #     embedding: Union[List[float], None] = None,
    #     document: Optional[Any] = None,
    #     id: Optional[str] = None,) -> Tuple[Dict[str, Dict[str, Any]], Union[bytes, None]]:
    #     if id is None:
    #         props: Dict[str, Any] = {}
    #     else:
    #         props = {"id": id}
    #         id_exists, query = _check_descriptor_exists_by_id(
    #             self._client, collection_name, id
    #         )
    #         if id_exists:
    #             skipped_value = {
    #                 prop_key: prop_val[-1]
    #                 for prop_key, prop_val in query["FindDescriptor"][
    #                     "constraints"
    #                 ].items()
    #             }
    #             pstr = f"[!] Embedding with id ({id}) exists in DB;"
    #             pstr += "Therefore, skipped and not inserted"
    #             print(pstr)  # noqa: T201
    #             print(f"\tSkipped values are: {skipped_value}")  # noqa: T201
    #             return query, None

    #     if metadata:
    #         props.update(metadata)
    #     if document:
    #         props["content"] = document

    #     for k in props.keys():
    #         if k not in self.collection_properties:
    #             self.collection_properties.append(k)

    #     query = add_descriptor(
    #         "AddDescriptor",
    #         collection_name,
    #         label=None,
    #         ref=None,
    #         props=props,
    #         link=None,
    #         k_neighbors=None,
    #         constraints=None,
    #         results=None,
    #     )

    #     blob = embedding2bytes(embedding)

    #     return (
    #         query,
    #         blob,
    #     )

    def __run_vdms_query(
        self,
        all_queries: List[Dict] = None,
        all_blobs: Optional[List] = None,
        print_last_response: Optional[bool] = False,) -> Tuple[Any, Any]:

        if all_queries is None:
            all_queries = []
        if all_blobs is None:
            all_blobs = []
        #start_time = time.time()
        response, response_array = self.client.query(all_queries, all_blobs)
        #end_time = time.time()
        #print("Time for 32 batch insertaion: ",round(end_time-start_time,2))
        
        #_ = _check_valid_response(all_queries, response)
        #if print_last_response:
        #    self.client.print_last_response()
        return response, response_array

    def batch_insertion(self, batch_size, metadatas, embeddings):
        inserted_ids = 0
        for start_idx in range(0, len(embeddings), batch_size):
            end_idx = min(start_idx + batch_size, len(embeddings))

            batch_embedding_vectors = embeddings[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]

            all_queries, all_blobs = [], []
            for metadata, embedding in zip(batch_metadatas, batch_embedding_vectors, strict=True):
                query = add_descriptor(
                    "AddDescriptor",
                    self.collection_name,
                    label=None,
                    ref=None,
                    props=metadata,
                    link=None,
                    k_neighbors=None,
                    constraints=None,
                    results=None,
                )
                blob = embedding2bytes(embedding)
                all_queries.append(query)
                all_blobs.append(blob)
            response, response_array = self.__run_vdms_query(all_queries, all_blobs)

            if "FailedCommand" not in response[0]:
                inserted_ids += len(response)
        return inserted_ids


    def batch_insertion_w_batch(self, batch_size, metadatas, embeddings):
        inserted_ids = 0
        total_emd = len(embeddings)
        for start_idx in range(0, total_emd, batch_size):
            end_idx = min(start_idx + batch_size, total_emd)

            batch_embedding_vectors = np.array(embeddings[start_idx:end_idx],dtype="float32").flatten()
            batch_metadatas = metadatas[start_idx:end_idx]
            
            properties_list = []
            rem_count = total_emd - start_idx
            for x in range(min(batch_size,rem_count)):
                props = {"id":batch_metadatas[x]["id"]}#{"id": (total_emd*epoch_count)+x+start_idx}
                properties_list.append(props)

            all_queries, all_blobs = [], []
            #for metadata, embedding in zip(batch_metadatas, batch_embedding_vectors, strict=True):
            query = add_descriptor(
                    "AddDescriptor",
                    self.collection_name,
                    label=None,
                    ref=None,
                    #props=batch_metadatas,
                    batch_props=properties_list,
                    link=None,
                    k_neighbors=None,
                    constraints=None,
                    results=None,
            )    
            blob = embedding2bytes(batch_embedding_vectors)
            all_queries.append(query)
            all_blobs.append(blob)
            #print("I'm here before __run_vdms_query")
            response, response_array = self.__run_vdms_query(all_queries, all_blobs)
            #print("I'm here after __run_vdms_query")
            #print(response)
            
            if "FailedCommand" not in response[0]:
                inserted_ids += len(response)
        return inserted_ids  
    
    def batch_insertion_wo_metadata(self, batch_size, embeddings):
        inserted_ids,total_time,total_count = 0,0,0
        for start_idx in range(0, len(embeddings), batch_size):
            end_idx = min(start_idx + batch_size, len(embeddings))

            batch_embedding_vectors = embeddings[start_idx:end_idx]
            #batch_metadatas = metadatas[start_idx:end_idx]

            all_queries, all_blobs = [], []
            for embedding in batch_embedding_vectors:
                query = add_descriptor(
                    "AddDescriptor",
                    self.collection_name,
                    label=None,
                    ref=None,
                    props=None,
                    link=None,
                    k_neighbors=None,
                    constraints=None,
                    results=None,
                )
                blob = embedding2bytes(embedding)
                all_queries.append(query)
                all_blobs.append(blob)
            start_time = time.time()
            #response, response_array = self.__run_vdms_query(all_queries, all_blobs)
            response, response_array = self.client.query(all_queries, all_blobs)
            total_time += time.time() - start_time
            total_count += 1
            if "FailedCommand" not in response[0]:
                inserted_ids += len(response)
        print("Total insert time: ",round(total_time,2))
        print("Total Epoch: ",total_count)
        return inserted_ids
def add_descriptorset(
    command_str: str,
    name: str,
    num_dims: Optional[int] = None,
    engine: Optional[str] = None,
    metric: Optional[str] = None,
    ref: Optional[int] = None,
    props: Optional[Dict] = None,
    link: Optional[Dict] = None,
    storeIndex: bool = False,
    constraints: Optional[Dict] = None,
    results: Optional[Dict] = None,) -> Dict[str, Any]:
    if command_str == "AddDescriptorSet" and all(
        var is not None for var in [name, num_dims]
    ):
        entity: Dict[str, Any] = {
            "name": name,
            "dimensions": num_dims,
        }

        if engine is not None:
            entity["engine"] = engine

        if metric is not None:
            entity["metric"] = metric

        if ref is not None:
            entity["_ref"] = ref

        if props not in [None, {}]:
            entity["properties"] = props

        if link is not None:
            entity["link"] = link

    elif command_str == "FindDescriptorSet":
        entity = {"set": name}

        if storeIndex:
            entity["storeIndex"] = storeIndex

        if constraints not in [None, {}]:
            entity["constraints"] = constraints

        if results is not None:
            entity["results"] = results

    else:
        raise ValueError(f"Unknown command: {command_str}")

    query = {command_str: entity}
    return query

def add_descriptor(
    command_str: str,
    setname: str,
    label: Optional[str] = None,
    ref: Optional[int] = None,
    props: Optional[dict] = None,
    link: Optional[dict] = None,
    k_neighbors: Optional[int] = None,
    constraints: Optional[dict] = None,
    results: Optional[dict] = None,) -> Dict[str, Dict[str, Any]]:
    entity: Dict[str, Any] = {"set": setname}

    if "Add" in command_str and label:
        entity["label"] = label

    if ref is not None:
        entity["_ref"] = ref

    if props not in INVALID_METADATA_VALUE:
        entity["properties"] = props
        
    if "Add" in command_str and link is not None:
        entity["link"] = link

    if "Find" in command_str and k_neighbors is not None:
        entity["k_neighbors"] = int(k_neighbors)

    if "Find" in command_str and constraints not in INVALID_METADATA_VALUE:
        entity["constraints"] = constraints

    if "Find" in command_str and results not in INVALID_METADATA_VALUE:
        entity["results"] = results

    query = {command_str: entity}
    return query

def _get_cmds_from_query(all_queries: list) -> List[str]:
    return list(set([k for q in all_queries for k in q.keys()]))

def _check_valid_response(all_queries: List[dict], response: Any) -> bool:
    cmd_list = _get_cmds_from_query(all_queries)
    valid_res = isinstance(response, list) and any(
        cmd in response[0]
        and "returned" in response[0][cmd]
        and response[0][cmd]["returned"] > 0
        for cmd in cmd_list
    )
    return valid_res

def embedding2bytes(embedding: Union[List[float], None]) -> Union[bytes, None]:
    """Convert embedding to bytes."""

    blob = None
    if embedding is not None:
        emb = np.array(embedding, dtype="float32")
        blob = emb.tobytes()
    return blob

