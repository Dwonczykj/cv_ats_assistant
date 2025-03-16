from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
from typing import Literal
import requests
from PIL import Image
import io
from pydantic import SecretStr
from langgraph.graph.state import CompiledStateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json
from typing import Any, Literal, overload
from langgraph.types import Checkpointer
from cv_ats_assistant.models import GraphState
from cv_ats_assistant.logging_config import logging
from PIL import Image
import io
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.sqlite import SqliteSaver

# /Users/joey/Library/CloudStorage/GoogleDrive-joey@joeysjars.co.uk/My Drive/google_colab_codebases/langchain-academy/module-1/deployment.ipynb


@overload
async def get_last_graph_state(memory: SqliteSaver, graph: CompiledStateGraph, thread: RunnableConfig) -> tuple[CheckpointTuple | None, StateSnapshot | None]:
    ...


@overload
async def get_last_graph_state(memory: AsyncSqliteSaver, graph: CompiledStateGraph, thread: RunnableConfig) -> tuple[CheckpointTuple | None, StateSnapshot | None]:
    ...


async def get_last_graph_state(memory: Checkpointer, graph: CompiledStateGraph, thread: RunnableConfig) -> tuple[CheckpointTuple | None, StateSnapshot | None]:
    all_states = [s async for s in graph.aget_state_history(config=thread)]
    if isinstance(memory, SqliteSaver):
        all_checkpoints = [s for s in memory.list(thread)]
        to_fork = all_checkpoints[-1]
        return to_fork, all_states[-1]
    elif isinstance(memory, AsyncSqliteSaver):
        all_checkpoints = [s async for s in memory.alist(thread)]
        to_fork = all_checkpoints[-1]
        return to_fork, all_states[-1]
    return None, None


async def fork_graph(memory: Checkpointer, graph: CompiledStateGraph, prev_graph_config: RunnableConfig, updated_state: GraphState | dict[str, Any]):
    fork_config = graph.update_state(
        config=prev_graph_config,
        values=updated_state,
    )
    return fork_config


async def get_state_history(checkpointer: AsyncSqliteSaver | SqliteSaver, graph: CompiledStateGraph, graph_config: RunnableConfig):
    state_history = [s async for s in graph.aget_state_history(config=graph_config)]
    checkpoint_history = [s async for s in checkpointer.alist(config=graph_config)]
    return state_history, checkpoint_history


async def get_state_history_checkpoint_with_id(checkpointer: AsyncSqliteSaver | SqliteSaver, graph: CompiledStateGraph, graph_config: RunnableConfig, checkpoint_id: str):
    if not checkpoint_id:
        return None, None
    state_history = [s async for s in graph.aget_state_history(config=graph_config) if s.config.get("configurable", {}).get("checkpoint_id", "") == checkpoint_id]
    checkpoint_history = [s async for s in checkpointer.alist(config=graph_config) if s.checkpoint["id"] == checkpoint_id]
    return (state_history[0] if state_history else None), (checkpoint_history[0] if checkpoint_history else None)


class GetCVVectorstoreManager:
    """
    Manager for the CV vector store that takes care of loading CV from uri, creating embeddings and caching them alongside metadata.
    """

    def get_vectorstore(self):
        self._cv_embedding = FAISS.load_local(
            self.vectorstore_path,
            OpenAIEmbeddings(api_key=SecretStr(
                os.getenv("OPENAI_API_KEY", ""))),
            allow_dangerous_deserialization=True
        )
        return self._cv_embedding

    def __init__(self, vectorstore_path: str, vectorstore_metadata_path: str, cv_path: str):
        self._vectorstore_path = vectorstore_path
        self._vectorstore_metadata_path = vectorstore_metadata_path
        self._cv_path = cv_path
        self._cv_embedding: FAISS | None = None

    @property
    def vectorstore_path(self):
        return self._vectorstore_path

    @property
    def vectorstore_metadata_path(self):
        return self._vectorstore_metadata_path

    @property
    def cv_path(self):
        return self._cv_path

    @property
    def cv_vectorstore(self):
        if not self._cv_embedding:
            self._cv_embedding = self.get_vectorstore()
        return self._cv_embedding


def get_cv_vectorstore(cv_uri: str) -> GetCVVectorstoreManager:
    """Create embeddings for the CV document"""
    if not cv_uri:
        raise ValueError("CV path is required")

    # Get the last modified date of the CV
    if cv_uri.startswith("http"):
        response = requests.get(cv_uri)
        with open("./cv_ats_assistant/cv_temp.pdf", "wb") as f:
            f.write(response.content)
        cv_path = Path("./cv_ats_assistant/cv_temp.pdf")
    else:
        cv_path = Path(cv_uri)
    if not cv_path.exists():
        raise ValueError(f"CV file not found at {cv_path}")

    last_modified = cv_path.stat().st_mtime
    last_modified_date = datetime.fromtimestamp(last_modified)

    # Create cache directory if it doesn't exist
    cache_dir = Path("./cv_ats_assistant/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache file paths
    cv_hash = hashlib.md5(cv_path.read_bytes()).hexdigest()
    cache_file = cache_dir / f"cv_embedding_{cv_hash}.faiss"
    metadata_file = cache_dir / f"cv_embedding_{cv_hash}.json"

    # Check if cached version exists and is up to date
    if cache_file.exists() and metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            cached_modified = datetime.fromisoformat(
                metadata['last_modified'])

            if cached_modified >= last_modified_date:
                # Load cached embeddings
                embeddings = OpenAIEmbeddings(
                    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")))
                vectorstore = FAISS.load_local(
                    str(cache_file),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logging.info(
                    f"Loaded cached CV embeddings from {cache_file}")
                return GetCVVectorstoreManager(
                    vectorstore_path=str(cache_file),
                    vectorstore_metadata_path=str(metadata_file),
                    cv_path=str(cv_path)
                )
        except Exception as e:
            logging.warning(f"Error loading cached embeddings: {e}")

    # If no valid cache exists, create new embeddings
    loader = PyPDFLoader(str(cv_path))
    pages = loader.load()

    embeddings = OpenAIEmbeddings(
        api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")))
    vectorstore = FAISS.from_documents(pages, embeddings)

    # Save embeddings and metadata
    vectorstore.save_local(str(cache_file))
    with open(metadata_file, 'w') as f:
        json.dump({
            'last_modified': last_modified_date.isoformat(),
            'cv_path': str(cv_path),
            'cv_hash': cv_hash
        }, f)

    logging.info(f"Created and cached new CV embeddings to {cache_file}")
    return GetCVVectorstoreManager(
        vectorstore_path=str(cache_file),
        vectorstore_metadata_path=str(metadata_file),
        cv_path=str(cv_path)
    )


def is_colab() -> Literal['google_colab', 'mac', 'mac_ipynb', 'linux', 'linux_ipynb', 'windows', 'windows_ipynb', 'java', 'java_ipynb']:
    try:
        from IPython import get_ipython  # type: ignore
        is_notebook = True
        if is_notebook and 'google.colab' in str(get_ipython()):
            return 'google_colab'
    except ImportError:
        is_notebook = False
    logging.debug(f"Running on '{platform.system()}' system with processor: '{
                  platform.processor()}'")
    if 'Darwin' in platform.system():
        return 'mac' + ('_ipynb' if is_notebook else '')
    elif 'Windows' in platform.system():
        return 'windows' + ('_ipynb' if is_notebook else '')
    elif 'Java' in platform.system():
        return 'java' + ('_ipynb' if is_notebook else '')
    else:
        logging.warning(f"Running on {platform.system()} -> Returning 'linux'")
        return 'linux' + ('_ipynb' if is_notebook else '')


def require_mac():
    if is_colab() != 'mac':
        raise Exception(
            "Unfortunately LangGraph Studio is currently not supported on Google Colab or requires a Mac")


def draw_graph(graph: CompiledStateGraph):
    image_data = graph.get_graph(xray=1).draw_mermaid_png()
    image = Image.open(io.BytesIO(image_data))
    image.save("./cv_curator_graph.png")
