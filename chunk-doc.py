import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
import os
import ssl
import logging

from langchain_text_splitters import TokenTextSplitter, TextSplitter, RecursiveCharacterTextSplitter, HTMLSectionSplitter, HTMLHeaderTextSplitter
from langchain_community.document_loaders import BSHTMLLoader, CSVLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models import BaseLanguageModel
from openai import BadRequestError
from pydantic import SecretStr
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_STUDIO_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_GEMINI_STUDIO_API_KEY is not set")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('openai')
logger.setLevel(logging.DEBUG)

# Configure SSL context


def configure_ssl():
    """Configure SSL context for requests"""
    try:
        # Create SSL context with system certificates
        ssl_context = ssl.create_default_context()
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True

        # Set certificate paths
        ssl_context.load_verify_locations(
            cafile="/etc/ssl/cert.pem",  # MacOS system certificates
            capath=None,
            cadata=None
        )

        return ssl_context
    except Exception as e:
        logger.error(f"Failed to configure SSL context: {e}")
        raise


# Configure SSL at module level
ssl_context = configure_ssl()


@dataclass
class ChunkResult:
    """Result of chunking a document"""
    chunks: List[str]
    messages: List[HumanMessage]


class TokenSplittingParams(TypedDict):
    """Parameters for token splitting"""
    embedding_model: str
    use_tokens: bool


def get_document_loader(file_path: Union[str, Path]) -> BaseLoader:
    """Get the appropriate document loader based on file extension or doc_type"""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.as_uri().startswith("http"):
        return WebBaseLoader(str(file_path))

    file_ext = file_path.suffix.lower()
    if file_ext.startswith("."):
        file_ext = file_ext[1:]

    if file_ext:
        if file_ext.lower() == "pdf":
            return PyPDFLoader(str(file_path))
        elif file_ext.lower() in ["html", "htm"]:
            return BSHTMLLoader(str(file_path))
        elif file_ext.lower() == "csv":
            return CSVLoader(str(file_path))
        elif file_ext.lower() == "md":
            return TextLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file extension: \"{file_ext}\"")
    else:
        raise ValueError(f"Unsupported file path: \"{file_path}\"")


def chunk_document(
    doc_path: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    user_query: Optional[str] = None,
    by: Literal["tagname", "markdown"] = "markdown",
    token_splitting_params: Optional[TokenSplittingParams] = None,
    max_chunks: Optional[int] = None
) -> ChunkResult:
    """
    Load and chunk a document, optionally creating messages for each chunk if user_query is provided.

    Args:
        doc_path: Path to the document or URL
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        user_query: Optional query to create messages for each chunk
        by: Chunking method - either by HTML tags or markdown
        token_splitting_params: Optional parameters for token-based splitting
        max_chunks: Optional maximum number of chunks to return

    Returns:
        ChunkResult containing the chunks and optional messages
    """
    # Get appropriate loader
    loader = get_document_loader(doc_path)

    # Load document
    docs = loader.load()

    if token_splitting_params and token_splitting_params["use_tokens"]:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                token_splitting_params["embedding_model"]),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("thenlper/gte-small"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    # Split into chunks
    chunks = []
    for doc in docs:
        doc_chunks = text_splitter.split_text(doc.page_content)
        chunks.extend(doc_chunks)

    # Limit number of chunks if specified
    if max_chunks and len(chunks) > max_chunks:
        logger.warning(f"Truncating {len(chunks)} chunks to {
                       max_chunks} chunks")
        chunks = chunks[:max_chunks]

    # Create messages if query provided
    messages: List[HumanMessage] = []
    if user_query:
        n = len(chunks)
        messages = [
            HumanMessage(content=f"{user_query}\nHere the following content is the document chunked with a chunk overlap of {
                         chunk_overlap} characters and contains {n} chunks, please do not answer the query above until you have read the following {n} chunks:\n\n"),
            *[HumanMessage(content=f"Document chunk {i+1}: {chunk}")
              for i, chunk in enumerate(chunks)]
        ]

    return ChunkResult(chunks=chunks, messages=messages)


async def pass_chunks_to_agent(
        messages: List[HumanMessage],
        context_strategy: Literal["RAG_VectorStore", "Summarize", "Query_Each_Chunk"],
        agent: Optional[BaseChatModel] = None,
        ) -> str:
    """Pass the chunks to the agent and return the response

    Args:
        messages: List[HumanMessage]: List of messages to process
        context_strategy: Literal["RAG_VectorStore", "Summarize", "Query_Each_Chunk"]
        agent: Optional[BaseChatModel]: Optional language model to use, defaults to ChatOpenAI

    Returns:
        str: The agent's response

    Raises:
        ValueError: If the combined chunks exceed the model's context length
        Exception: If there is an error connecting to the API or processing the chunks
    """
    if not agent:
        agent = ChatOpenAI(
            model="gpt-4",
            temperature=0
        )

    # Format chunks into a single message

    result = await agent.ainvoke([messages[0]])

    

    # Check token count before making the API call
    try:
        # num_tokens = agent.get_num_tokens_from_messages(messages)
        # context_length = getattr(agent, "max_tokens", None)

        if isinstance(agent, ChatOpenAI):
            # Default context lengths for OpenAI models
            model_context_lengths = {
                "gpt-4": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo-preview": 128000,
                "gpt-3.5-turbo": 16385,
                "gpt-3.5-turbo-16k": 16385,
                "gpt-4o": ...,
                "gpt-4o-mini": ...,
                "gpt-o1": ...,
            }
            model_name = agent.model_name
            context_length = model_context_lengths.get(model_name, 8192)
        elif isinstance(agent, ChatGoogleGenerativeAI):
            model_context_lengths = {
                "gemini-1.5-flash-latest": ...,
            }
            model_name = agent.model
            context_length = model_context_lengths.get(model_name, 8192)
        elif isinstance(agent, ChatAnthropic):
            model_context_lengths = {
                "claude-3-5-sonnet-20240620": ...,
            }
            model_name = agent.model
            context_length = model_context_lengths.get(model_name, 8192)
        elif isinstance(agent, ChatHuggingFace):
            model_context_lengths = {
                "gpt2": ...,
            }
            model_name = agent.model_id or "gpt2"
            context_length = model_context_lengths.get(model_name, 8192)

        if context_length and num_tokens > context_length:
            raise ValueError(
                f"Input length ({
                    num_tokens} tokens) exceeds model's maximum context length "
                f"({context_length} tokens). Consider reducing chunk size or using fewer chunks."
            )

        logger.debug(f"Token count for request: {num_tokens}")
    except NotImplementedError:
        # If get_num_tokens is not implemented, log a warning and proceed
        logger.warning(
            "Unable to check token count - model doesn't implement get_num_tokens")

    # Get response with retries
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1}: Making API request...")
            result = agent.invoke(messages)
            if isinstance(result.content, str):
                return result.content
            raise ValueError("Unexpected response format from agent")
        except BadRequestError as e:
            logger.error(f"BadRequestError: {str(e)}")
            raise  # Re-raise BadRequestError since retrying won't help with context length issues
        except Exception as e:
            last_error = e
            logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                continue

    error_msg = f"Error processing chunks with agent after {
        max_retries} attempts: {str(last_error)}"
    logger.error(error_msg)
    raise last_error if last_error else RuntimeError(error_msg)


if __name__ == "__main__":
    # Example usage
    import sys

    csv_path = "/Users/joey/Library/Mobile Documents/com~apple~CloudDocs/SleepCycle/sleepdata2025-01-25.csv"

    if len(sys.argv) == 2:
        doc_path = sys.argv[1]
        query = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        doc_path = csv_path
        query = """Please perform exploratory data analysis on this csv to do some k-means clustering, perhaps creating a Neo-4j db and do some community analysis with Differing algorithms and maybe apply a random forest for the predictive stuff, i.e. given all of these conditions, what can I expect over the next few days, can I predict any of the other labels based on what labels occur in the same neighborhoods.
        Can you write the python code for a recommendation system on how to predict trends ie predict good sleep quality based on labels of previous nights and location etc, time of year etc and how to predict the likelihood of each label occuring based on the timeseries of all information over the last window of n days where n days is chosen to be optimal"""
        # print("Usage: python chunk-doc.py <document_path> [query]")
        # sys.exit(1)

    try:
        result = chunk_document(doc_path, user_query=query)
        print(f"Successfully chunked document into {
              len(result.chunks)} chunks")
        if result.messages:
            print(f"Created {len(result.messages)} messages")
        else:
            print("No messages created")
            sys.exit(1)

        result = asyncio.run(pass_chunks_to_agent(
            result.messages,
            context_strategy="Query_Each_Chunk",
            # agent=ChatGoogleGenerativeAI(
            #     model="gemini-1.5-flash-latest",
            #     temperature=0,
            #     api_key=SecretStr(GOOGLE_API_KEY)),
            agent=ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                # api_key=SecretStr(GOOGLE_API_KEY)
            ),
        ))
    except Exception as e:
        print(f"Error processing document: {e}")
        sys.exit(1)
