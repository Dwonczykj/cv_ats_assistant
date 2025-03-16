import getpass
import asyncio
import hashlib

import requests
import cv_ats_assistant.helpers as helpers
from cv_ats_assistant.type_to_typeddict import create_typed_dict_from_class
from cv_ats_assistant.logging_config import logging, setup_logging, configure_logging
import os
import pickle
import re
import time
from pprint import pprint
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Generic, List, Literal, Optional, Protocol, Sequence, Tuple, Type, TypeVar, TypedDict, Union, cast
import warnings
from PIL import Image
import io
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import StructuredTool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableBinding, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph_sdk.schema import Thread
from langgraph.types import StreamMode, StateSnapshot
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from markdownify import markdownify as md
from PyPDF2 import PdfReader, PdfWriter
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr
from promptwatch import register_prompt_template
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from urllib.parse import quote, urlparse
from webdriver_manager.chrome import ChromeDriverManager
import json

from cv_ats_assistant.models import (AuthAction, AuthActionActionEnum,
                                     AuthenticationGraphInputState, AuthenticationGraphOutputState, AuthenticationGraphOverallState,
                                     CVScores, GraphInputState, GraphOutputState, GraphState, GraphStateDict,
                                     JobGraphInputState, JobGraphOutputState, JobGraphOverallState, JobInfo, JobPost,
                                     LinkedInCredentials, WebContentActioniserInputState, WebContentActioniserOverallState,
                                     WebDocument, WebDriverProtocol)
import cv_ats_assistant.helpers as helpers

load_dotenv(dotenv_path="./env")

configure_logging(log_level=logging.DEBUG)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_STUDIO_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_GEMINI_STUDIO_API_KEY is not set in the environment variables")

if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

# if not os.getenv("GOOGLE_API_KEY"):
#     raise ValueError("GOOGLE_API_KEY is not set in the environment variables")


def _set_env(var: str):
    if not os.environ.get(var):
        if helpers.is_colab().endswith('_ipynb'):
            os.environ[var] = getpass.getpass(f"['{var}']: ")
        else:
            os.environ[var] = input(f"['{var}']: ")


_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"


CHUNK_SIZE = 8000
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CHALLENGE_PROBABILITY_THRESHOLD = 0.2
USE_CACHED_JOBPOST_WEBCONTENT = False
if USE_CACHED_JOBPOST_WEBCONTENT:
    file_name = "www.linkedin.com_20250112_122624_99d8d40f_page_source.html"
    cached_jobpost_webcontent_exists = (os.path.exists("./cv_ats_assistant/page_source/")
                                        and os.path.isdir("./cv_ats_assistant/page_source/")
                                        and os.path.isfile(f"./cv_ats_assistant/page_source/{file_name}"))
    if not cached_jobpost_webcontent_exists:
        raise ValueError(
            f"Cached jobpost webcontent does not exist: {file_name}")


# def dummy_init_faiss_store():
#     from transformers.agents import Tool
#     from langchain_community.vectorstores.utils import DistanceStrategy
#     from langchain_community.embeddings import HuggingFaceEmbeddings
#     from langchain_community.vectorstores import FAISS
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     from langchain.docstore.document import Document
#     from transformers import AutoTokenizer
#     from tqdm import tqdm
#     import datasets

#     knowledge_base = datasets.load_dataset(
#         "m-ric/huggingface_doc", split="train")
#     # Now we prepare the knowledge base by processing the dataset and storing it into a vector database to be used by the retriever.
#     #
#     # We use [LangChain](https://python.langchain.com/) for its excellent vector database utilities.
#     # For the embedding model, we use [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) since it performed well in our `RAG_evaluation` cookbook.

#     source_docs = [
#         Document(page_content=doc["text"], metadata={
#             "source": doc["source"].split("/")[1]})
#         for doc in knowledge_base
#     ]

#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         AutoTokenizer.from_pretrained("thenlper/gte-small"),
#         chunk_size=200,
#         chunk_overlap=20,
#         add_start_index=True,
#         strip_whitespace=True,
#         separators=["\n\n", "\n", ".", " ", ""],
#     )

#     # Split docs and keep only unique ones
#     print("Splitting documents...")
#     docs_processed = []
#     unique_texts = {}
#     for doc in tqdm(source_docs):
#         new_docs = text_splitter.split_documents([doc])
#         for new_doc in new_docs:
#             if new_doc.page_content not in unique_texts:
#                 unique_texts[new_doc.page_content] = True
#                 docs_processed.append(new_doc)

#     print(
#         "Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)"
#     )
#     embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
#     vectordb = FAISS.from_documents(
#         documents=docs_processed,
#         embedding=embedding_model,
#         distance_strategy=DistanceStrategy.COSINE,
#     )
#     return vectordb


def get_linkedin_credentials() -> LinkedInCredentials:
    """Get LinkedIn credentials from environment variables."""
    username = os.getenv('LINKEDIN_USERNAME')
    password = os.getenv('LINKEDIN_PASSWORD')

    if not username or not password:
        raise ValueError(
            "LinkedIn credentials not found in environment variables. "
            "Please set LINKEDIN_USERNAME and LINKEDIN_PASSWORD.")

    return LinkedInCredentials(username=username, password=password)


class CookieManager:
    """Manages browser cookies for different domains."""

    def __init__(self, cookie_dir: str = ".cookies"):
        self.cookie_dir = Path(cookie_dir)
        self.cookie_dir.mkdir(exist_ok=True)

    def _get_cookie_file(self, domain: str) -> Path:
        """Get the path to the cookie file for a domain."""
        return self.cookie_dir / f"{domain}.cookies"

    def save_cookies(self, domain: str, cookies: List[Dict[str, Any]], expiry_days: int = 7) -> None:
        """Save cookies for a domain with expiration date."""
        cookie_data = {
            'cookies': cookies,
            'expires': datetime.now() + timedelta(days=expiry_days)
        }
        with open(self._get_cookie_file(domain), 'wb') as f:
            pickle.dump(cookie_data, f)
        logging.info(f"Saved cookies for domain: {domain}")

    def load_cookies(self, domain: str) -> Optional[List[Dict[str, Any]]]:
        """Load cookies for a domain if they exist and haven't expired."""
        cookie_file = self._get_cookie_file(domain)
        if not cookie_file.exists():
            return None

        try:
            with open(cookie_file, 'rb') as f:
                cookie_data = pickle.load(f)

            if datetime.now() > cookie_data['expires']:
                logging.info(f"Cookies expired for domain: {domain}")
                cookie_file.unlink()  # Delete expired cookies
                return None

            logging.info(f"Loaded valid cookies for domain: {domain}")
            return cookie_data['cookies']
        except Exception as e:
            logging.error(f"Error loading cookies for {domain}: {str(e)}")
            return None

    def delete_cookies(self, domain: str) -> None:
        """Delete cookies for a domain."""
        cookie_file = self._get_cookie_file(domain)
        if cookie_file.exists():
            cookie_file.unlink()
            logging.info(f"Deleted cookies for domain: {domain}")


class WebScraperOutput(BaseModel):
    """Output of the web scraper"""
    current_page_source: str = Field(...,
                                     description="The current page source")
    current_url: str = Field(..., description="The current URL")
    current_title: str = Field(..., description="The current title")
    error: Optional[str] = Field(
        default="", description="The error message if any")


class WebScraperManager:
    """Manages a persistent WebDriver instance and provides a StructuredTool interface."""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            logging.debug("Initializing WebScraperManager")
            self._driver: Optional[WebDriverProtocol] = None
            self._wait: Optional[WebDriverWait] = None
            self._initialized = True

    def create_tool(self) -> StructuredTool:
        """Create a StructuredTool instance with the scraper function. See https://python.langchain.com/docs/how_to/custom_tools/#subclass-basetool"""
        # TODO: I need to set the USER_AGENT TO Identify My device to the scraper tool so that linkedin doesnt think Im a new unknown device.
        return StructuredTool.from_function(
            func=self.scrape_with_auth,
            name="web_scraper",
            description="Scrapes web content with optional authentication",
            handle_tool_error=True,
            # response_format=WebScraperOutput
            # return_direct=
        )

    @property
    def driver(self) -> WebDriverProtocol:
        """Get the current driver instance."""
        if not self._driver:
            self.initialize_driver()
        return cast(WebDriverProtocol, self._driver)

    @property
    def wait(self) -> WebDriverWait:
        """Get the current wait instance."""
        if not self._wait:
            self.initialize_driver()
        return cast(WebDriverWait, self._wait)

    def initialize_driver(self):
        """Initialize or reset the Chrome driver."""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(
            '--disable-blink-features=AutomationControlled')
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36')

        # TODO: How do I set the user's device to identify as a Mac?
        options.add_argument('--Sec-Ch-Ua-Platform="MacOS"')
        options.add_argument('--Sec-Ch-Ua-Mobile=?0')
        # options.add_argument('--=')

        service = Service(ChromeDriverManager().install())
        try:
            self._driver = webdriver.Chrome(service=service, options=options)
            self._wait = WebDriverWait(self._driver, 10)
        except Exception as e:
            logging.error(f"Error initializing driver: {str(e)}")
            logging.info("If driver continues to fail, consider running 'chromedriver --port=9515' or '   rm -rf /Users/joey/.wdm/drivers/chromedriver/mac64/131.0.6778.264/' but for the current driver path.")
            raise

    def scrape_with_auth(self, url: str, credentials: Optional[Dict] = None) -> WebScraperOutput:
        """Scrape content with authentication support."""
        logging.debug("Scraping with authentication support")
        if not self._driver:
            self.initialize_driver()
        logging.debug(f"Driver initialized: {self._driver}")
        domain = urlparse(url).netloc
        cookie_manager = CookieManager()

        try:
            logging.debug("Loading existing cookies")
            # Try loading existing cookies
            cookies = cookie_manager.load_cookies(domain)  # LAZY
            if cookies:
                logging.debug(f"Cookies loaded: {cookies}")
                logging.debug(
                    f"{type(self.driver).__name__} making GET request to 'https://{domain}'")
                self.driver.get("https://" + domain)
                for cookie in cookies:
                    try:
                        self.driver.add_cookie(cookie)
                        logging.debug(f"Cookie for domain '{
                                      domain}' added: {cookie}")
                    except Exception as e:
                        logging.warning(f"Failed to add cookie for domain '{
                                        domain}': {str(e)}")
                        raise Exception(f"Failed to add cookie for domain '{
                                        domain}': {str(e)}")
                logging.info(f"Loaded existing cookies for {domain}")

            self.driver.get(url)
            logging.info(f"Accessing URL: {url}")

            # Handle login if needed
            if "/login" in self.driver.current_url or "login" in self.driver.title.lower():
                if not credentials:
                    raise ValueError("Credentials required but not provided")

                logging.info("Login required. Attempting authentication...")
                try:
                    username_field = self.wait.until(
                        EC.presence_of_element_located((By.ID, "username")))
                    password_field = self.wait.until(
                        EC.presence_of_element_located((By.ID, "password")))

                    username_field.send_keys(credentials['username'])
                    password_field.send_keys(credentials['password'])

                    submit_button = self.wait.until(EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "button[type='submit']")))
                    submit_button.click()

                    time.sleep(3)
                    cookie_manager.save_cookies(
                        domain, self.driver.get_cookies())
                    logging.info(f"Saved new cookies for {domain}")
                    self.driver.get(url)
                except TimeoutException:
                    logging.error("Timeout during login process")
                    raise

            # Wait for content and verify it's loaded
            self.wait.until(EC.presence_of_element_located(
                (By.TAG_NAME, "body")))
            if "login" in self.driver.title.lower() or "login" in self.driver.current_url:
                logging.error(f"Login failed for {domain}")
                cookie_manager.delete_cookies(domain)
                return WebScraperOutput(
                    current_page_source=self.driver.page_source,
                    current_url=self.driver.current_url,
                    current_title=self.driver.title,
                    error="Login failed and might require human confirmation",
                )
                # raise WebDriverException("Login failed")

            content = self.driver.page_source

            # Basic content validation
            if len(content) < 100 or "Page Not Found" in content:
                raise WebDriverException("Invalid or empty content received")

            # Save the page source
            saved_path = save_page_source(
                url, self.driver.current_url, content)
            if saved_path:
                logging.info(f"Page source saved to: {saved_path}")

            logging.info("Content successfully scraped")
            return WebScraperOutput(
                current_page_source=content,
                current_url=self.driver.current_url,
                current_title=self.driver.title,
            )

        except Exception as e:
            logging.error(f"Error during web scraping: {str(e)}")
            self.initialize_driver()  # Reset driver on error
            raise

    def cleanup(self):
        """Clean up resources."""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
            self._driver = None
            self._wait = None


def create_web_scraper() -> tuple[StructuredTool, WebScraperManager]:
    """Create a web scraper tool instance."""
    manager = WebScraperManager()
    return manager.create_tool(), manager


T = TypeVar("T", bound=BaseModel)


def create_dummy_graph[T](output_state_type: Type[T], input: T) -> StateGraph:
    """Create a dummy graph for testing"""
    class DummyInputState(BaseModel):
        """Dummy input state"""
        initial_dummy_value: str = Field(...,
                                         description="The initial dummy value")

    class DummyOverallState(DummyInputState, output_state_type):
        """Dummy overall state"""
        internal_value: str = Field(..., description="The internal value")

    workflow = StateGraph(DummyOverallState,
                          input=DummyInputState,
                          output=output_state_type)
    assert isinstance(input, BaseModel)

    async def dummy_node1(state: DummyOverallState, config: RunnableConfig):
        """Dummy node 1"""
        return {
            **input.model_dump(),
            "current_url": "https://www.google.com",
            "internal_value": f"{state.initial_dummy_value};node1",
        }

    async def dummy_node2(state: DummyOverallState, config: RunnableConfig):
        """Dummy node 2"""
        return {
            "current_url": "https://www.bing.com",
            "internal_value": f"{state.initial_dummy_value};node2",
        }

    async def dummy_node3(state: DummyOverallState, config: RunnableConfig):
        """Dummy node 3"""
        return {
            "current_title": "Dummy node 3 Bing Title",
            "internal_value": f"{state.initial_dummy_value};node3",
        }

    workflow.add_node("dummy_node1", dummy_node1)
    workflow.add_node("dummy_node2", dummy_node2)
    workflow.add_node("dummy_node3", dummy_node3)

    workflow.add_edge(START, "dummy_node1")
    workflow.add_edge("dummy_node1", "dummy_node2")
    workflow.add_edge("dummy_node2", "dummy_node3")
    workflow.add_edge("dummy_node3", END)

    workflow.set_entry_point("dummy_node1")

    return workflow


def create_auth_graph(checkpointer: Checkpointer) -> StateGraph:
    """Create a LangGraph for handling authentication"""
    workflow = StateGraph(AuthenticationGraphOverallState,
                          input=AuthenticationGraphInputState,
                          output=AuthenticationGraphOutputState)

    # Create LLM with tools
    llm = ChatOpenAI(temperature=0, api_key=SecretStr(OPENAI_API_KEY))
    # Initialize the Gemini model
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash-latest", api_key=SecretStr(GOOGLE_API_KEY))
    # BUG IS THIS BEING CALLED TWICE SOMEHOW OR IS THE SCRAPER TOOL CALLING THE DRIVER TOO?
    scraper_tool, manager = create_web_scraper()
    # llm_with_tools = llm.bind_tools([scraper_tool])

    def is_test_mode(state: AuthenticationGraphOverallState) -> Literal['run_test_mode', 'attempt_auth']:
        """Determine if the test mode is enabled"""
        logging.debug(f"running is_test_mode conditional_edge")
        return 'run_test_mode' if state.test_mode else 'attempt_auth'

    def filter_tests(state: AuthenticationGraphInputState, config: RunnableConfig):
        """Run test mode"""
        logging.debug(f"running filter_tests node")
        if state.test_mode:
            content = load_page_source(state.job_post_url)
            return {
                "current_page_source": content,
                "current_url": state.job_post_url,
                "url_history": [state.job_post_url],
                "driver": manager.driver,
                "challenge_probability": 0.0,
                "credentials": None,
                "error": None,
            }
        else:
            return {}

    def initialize_auth(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Initialize authentication process"""
        logging.debug(f"running initialize_auth node")
        credentials = get_linkedin_credentials()
        state.credentials = credentials
        return {
            "current_url": state.job_post_url,
            "url_history": [state.job_post_url],
            "credentials": credentials,
        }

    def request_url_initial(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Attempt to authenticate using the web scraper"""
        logging.debug(f"running attempt_auth node")
        try:
            content: WebScraperOutput = scraper_tool.invoke({
                "url": state.job_post_url,
                "credentials": state.credentials
            })
            if content.error:
                return {
                    "current_page_source": content.current_page_source,
                    "current_url": content.current_url,
                    "current_title": content.current_title,
                    "authenticated": False,
                    "error": f"WebScraperError: {content.error}",
                }
        except Exception as e:
            logging.error(f"Error during web scraping: {str(e)}")
            raise
        saved_path = save_page_source(
            state.job_post_url, content.current_url, content.current_page_source)
        logging.info(f"Page source saved to: {saved_path}")
        return {
            "current_page_source": content.current_page_source,
            "current_url": content.current_url,
            "current_title": content.current_title,
            "authenticated": True,
        }

    async def request_url_again(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Attempt to authenticate using the web scraper"""
        logging.debug(f"running attempt_auth node")

        try:
            llm = ChatOpenAI(temperature=0, api_key=SecretStr(OPENAI_API_KEY))
            page_source = state.current_page_source
            chunks = await chunk_html_content_by_tagname(page_source, by="markdown")

            messages = [
                SystemMessage(
                    content="You are an assistant that can help bypass challenge pages on web pages."),
                HumanMessage(content=f"Current URL: {state.current_url}\nPage title: {
                             state.current_title}\nContent chunks:"),
                *[HumanMessage(content=c) for c in chunks],
                HumanMessage(
                    content=f"# First three chunks should be enough\n\nPlease tell me what the page is asking the user to do?")
            ]
            if not os.path.exists("./auth_challenge_pages"):
                os.makedirs("./auth_challenge_pages")
            with open(f"auth_challenge_pages/{state.current_url}.md", "w") as f:
                f.write("\n".join([c for c in chunks]))

            register_prompt_template(
                "case_a.create_auth_graph.request_url_again.AuthChallengePageIsAbout",
                ChatPromptTemplate.from_messages(messages),
                version="0.0.1+alpha")

            class AuthChallengePageIsAbout(BaseModel):
                """The probability that this is a login/sign in/challenge/2FA page/captcha page"""
                page_action_type: str = Field(
                    ...,
                    description="The type of action the user is being asked to do. This should be one of the following: login, sign in, challenge, 2FA, captcha, confirm_using_mobile_device or other")
                reasoning: str = Field(...,
                                       description="Explanation of the page_action_type")
                reason_source: str = Field(...,
                                           description="The section of the page that the reasoning is based on including the markdown the reasoning was taken from")
                list_summary: str = Field(...,
                                          description="An ordered list of points summarizing each part of the page")
            llm.max_tokens = 1000
            chain: RunnableBinding = llm.with_structured_output(
                AuthChallengePageIsAbout)  # type: ignore
            result: AuthChallengePageIsAbout = chain.invoke(messages)
            logging.info(f"Page action type: {
                         result.page_action_type} with reasoning: {result.reasoning} based on section: {result.reason_source}")
            return {
                "require_human_auth": True,
                "auth_challenge_page_action_type": result.page_action_type,
                "auth_challenge_page_action_reasoning": result.reasoning,
                "auth_challenge_page_action_reason_source": result.reason_source,
                "auth_challenge_page_action_list_summary": result.list_summary,
            }
        except Exception as e:
            return {
                "require_human_auth": True,
                "auth_challenge_page_action_type": "other",
                "auth_challenge_page_action_reasoning": f"Error during web scraping: {str(e)}",
            }

    async def human_auth(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Human authentication"""
        logging.debug(f"running human_auth node")
        return {
            "authenticated": True,
        }

    async def get_prob_page_is_auth_challenge(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Analyze current page to determine if it's a challenge/2FA page"""
        logging.debug(f"running analyze_page node")
        content = state.current_page_source
        chunks = await chunk_html_content_by_tagname(content)

        class AuthChallengeProbability(BaseModel):
            """The probability that this is a login/sign in/challenge/2FA page/captcha page"""
            probability_page_is_challenge: float = Field(...,
                                                         description="The probability is the probability that the page is a challenge page, security verification, or 2FA page. If the page is likely a challenge page, security verification, or 2FA page, then the probability should be close to 1. If the page is not likely a challenge page, security verification, or 2FA page, then the probability should be close to 0.")
            reasoning: str = Field(...,
                                   description="Explanation of the probability")
        chain: RunnableBinding = llm.with_structured_output(
            AuthChallengeProbability)  # type: ignore

        pt_contents = f"""Analyze this page content and determine the probability (0-1) that it is a challenge page,
            security verification, or 2FA page or session redirect page to force a re-authentication or check that a user is a real user and not a bot.
            Consider URL, title, and content.
            Current URL: {state.current_url}
            Page title: {state.current_title}
            Content chunks: {chunks[:2]}  # First two chunks should be enough
            Make sure that the Probability is the probability that the page is a challenge page, security verification, or 2FA page. If the page is likely a challenge page, security verification, or 2FA page, then the probability should be close to 1. If the page is not likely a challenge page, security verification, or 2FA page, then the probability should be close to 0.
            """
        register_prompt_template(
            "case_a.create_auth_graph.analyze_page.AuthChallengeProbability",
            ChatPromptTemplate.from_template(pt_contents),
            version="0.0.1+alpha")
        result: AuthChallengeProbability = await chain.ainvoke(
            pt_contents
        )
        logging.info(f"Challenge probability: {
            state.challenge_probability} for {state.current_url}\nwith reasoning: {result.reasoning}")
        return {
            "challenge_probability": result.probability_page_is_challenge,
        }

    def set_actioniser_actions_and_query(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Set the actions and query for the actioniser"""
        logging.debug(f"running set_actioniser_actions_and_query node")
        predefined_actions: list[AuthActionActionEnum] = [
            "wait",
            "reload",
            "execute_js",
            "click_button",
            "sign_in_with_credentials",
            "solve_captcha",
            "dont_know"
        ]
        query = (
            "You are an assistant that analyzes an authentication redirect/challenge/login/captcha web page and classifies the webpage with the best action to bypass the page and pass all authentication hurdles." +
            f"The only actions that you can recommend are {', '.join(predefined_actions[:-1])} or {predefined_actions[-1]}. " +
            "If you don't know the answer, the best action is dont_know. "
            # "Use the following pieces of retrieved context from the authentication web page to suggest the best action to bypass the web page's authentication. " +
            # "\nInstruc: {url} \nContext: {context} \nAnswer:"
        )
        if state.test_mode:
            return {
                "predefined_actions": predefined_actions,
                "query": query,
                "uri": "./page_source/www.linkedin.com_20250113_140134_07c34036_page_source.html",
                "uri_type": "file_path",
            }
        else:
            return {
                "predefined_actions": predefined_actions,
                "query": query,
                "uri": state.current_url,
                "uri_type": "url",
            }

    async def convert_webcontent_state_back_to_auth_state(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Convert the webcontent state back to the auth state"""
        logging.debug(
            f"running convert_webcontent_state_back_to_auth_state node")
        return {
            "current_page_source": "\n\n".join([doc.page_content for doc in state.documents]),
            "current_url": state.uri,
            "current_title": state.documents[0].title,
            "url_history": [state.uri],
        }

    async def handle_challenge(state: AuthenticationGraphOverallState, config: RunnableConfig):
        """Handle challenge pages by attempting various bypass strategies"""
        logging.debug(f"running handle_challenge node with action: {
            state.action}")
        if state.action and state.action.action not in state.predefined_actions:
            raise ValueError(f"Invalid action for challenge page: {
                             state.action.action}")

            chain: RunnableBinding = llm.with_structured_output(
                AuthAction)  # type: ignore

            messages = [
                SystemMessage(content="""You are a helpful assistant that can help bypass challenge pages on LinkedIn.
                    When suggesting an action, always include appropriate details:
                    - For 'wait': include 'duration' in seconds
                    - For 'reload': empty details are fine
                    - For 'execute_js': include 'script' with the JavaScript code
                    - For 'click_button': include 'selector' with the CSS selector
                    - For 'sign_in_with_credentials': include 'username' and 'password'
                    - For 'solve_captcha': include 'captcha_type' and 'captcha_id'
                    - For 'dont_know': empty details are fine"""),
                HumanMessage(content=f"""Analyze this challenge page and suggest the best action to bypass it.
                    Current URL: {state.current_url}
                    Page Title: {manager.driver.title}

                    Choose the most appropriate action from: {', '.join(predefined_actions[:-1])} or {predefined_actions[-1]}.
                    Always include relevant details for the chosen action.

                    Example response formats:
                    - For wait: {{"action": "wait", "details": {{"duration": 5}}}}
                    - For reload: {{"action": "reload", "details": {{}}}}
                    - For execute_js: {{"action": "execute_js", "details": {{"script": "window.scrollTo(0, document.body.scrollHeight)"}}}}
                    - For sign_in_with_credentials: {{"action": "sign_in_with_credentials", "details": {{"username": "your_username", "password": "your_password"}}}}
                    - For click_button: {{"action": "click_button", "details": {{"selector": "button.challenge-button"}}}}
                    - For dont_know: {{"action": "dont_know", "details": {{}}}}
                    - For solve_captcha: {{"action": "solve_captcha", "details": {{"captcha_type": "recaptcha", "captcha_id": "1234567890"}}}}
                    """),
                HumanMessage(content=f"Page source: {
                    state.current_page_source[:5000]}")  # Limit page source size
            ]

            result: AuthAction = chain.invoke(messages)

        # Execute suggested action with proper error handling
        result = state.action or AuthAction(action="dont_know", details={
            "reasoning": "No action returned from actioniser llm graph"
        })
        try:
            if result.action == "wait":
                duration = result.details.get("duration", 5)
                logging.info(f"Waiting for {duration} seconds")
                time.sleep(duration)
                return {
                    "current_url": manager.driver.current_url,
                    "current_page_source": manager.driver.page_source,
                }

            elif result.action == "reload":
                logging.info("Reloading page")
                manager.driver.refresh()
                manager.wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "body")))
                return {
                    "current_url": manager.driver.current_url,
                    "current_page_source": manager.driver.page_source,
                }

            elif result.action == "execute_js":
                script = result.details.get("script")
                if script:
                    logging.info(f"Executing JavaScript: {script[:100]}...")
                    manager.driver.execute_script(script)
                    manager.wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "body")))
                    return {
                        "current_url": manager.driver.current_url,
                        "current_page_source": manager.driver.page_source,
                    }
                else:
                    logging.warning("No script provided for execute_js action")
                    return {
                        "error": "No script provided for execute_js action"
                    }

            elif result.action == "sign_in_with_credentials":
                username = result.details.get("username")
                password = result.details.get("password")
                if username and password:
                    logging.info(f"Signing in with credentials: {
                        username[:10]}...")
                    manager.driver.find_element(
                        By.ID, "username").send_keys(username)
                    manager.driver.find_element(
                        By.ID, "password").send_keys(password)
                    manager.driver.find_element(
                        By.CSS_SELECTOR, "button[type='submit']").click()
                    time.sleep(1)
                    return {
                        "current_url": manager.driver.current_url,
                        "current_page_source": manager.driver.page_source,
                    }
                else:
                    return {
                        "error": "No credentials provided for sign_in_with_credentials action"
                    }
                    logging.warning(
                        "No credentials provided for sign_in_with_credentials action")
                    raise ValueError(
                        "No credentials provided for sign_in_with_credentials action")

            elif result.action == "solve_captcha":
                captcha_type = result.details.get("captcha_type")
                captcha_id = result.details.get("captcha_id")
                if captcha_type and captcha_id:
                    logging.info(
                        f"Solving {captcha_type} captcha with ID: {captcha_id}")
                    # Implement captcha solving logic here by requesting the captcha image from the page and solving it using an LLM vision model
                    # TODO: Implement captcha solving logic here
                    return {
                        "error": f"Captcha solving logic not implemented for {captcha_type} captcha with ID: {captcha_id}"
                    }
                    raise Exception("Captcha solving logic not implemented")
                else:
                    return {
                        "error": "No captcha details provided for solve_captcha action"
                    }
                    logging.warning(
                        "No captcha details provided for solve_captcha action")
                    raise ValueError(
                        "No captcha details provided for solve_captcha action")

            elif result.action == "click_button":
                selector = result.details.get("selector")
                if selector:
                    logging.info(
                        f"Attempting to click button with selector: {selector}")
                    button = manager.driver.find_element(
                        By.CSS_SELECTOR, selector)
                    button.click()
                    time.sleep(1)
                    return {
                        "current_url": manager.driver.current_url,
                        "current_page_source": manager.driver.page_source,
                    }
                else:
                    logging.warning(
                        "No selector provided for click_button action")
            elif result.action == "dont_know":
                logging.info(f"No action found for the challenge page with reasoning: {
                    result.details.get('reasoning', "")}")
            else:
                raise ValueError(
                    f"Invalid action for challenge page: {result.action}")

            # Update state after action
            time.sleep(1)  # Brief pause to let any changes take effect
            return {
                "current_url": manager.driver.current_url,
                "current_page_source": manager.driver.page_source,
            }

        except Exception as e:
            logging.error(f"Error executing action {result.action}: {str(e)}")
            state.error = f"Failed to execute {result.action}: {str(e)}"

        return {}

    def is_challenge_page(state: AuthenticationGraphOverallState) -> Union[Literal["set_actioniser_actions_and_query"], Literal["cleanup"]]:
        """Determine if current page is a challenge page and map to node based on this condition"""
        logging.debug(f"running is_challenge_page conditional_edge")
        return "set_actioniser_actions_and_query" if state.challenge_probability > CHALLENGE_PROBABILITY_THRESHOLD else "cleanup"

    def requires_human_auth(state: AuthenticationGraphOverallState):
        """Determine if current page requires human authentication and map to node based on this condition"""
        logging.debug(f"running requires_human_auth conditional_edge")
        return "analyze_page" if state.authenticated else "request_url_again"

    def cleanup(state: AuthenticationGraphOverallState):
        """Clean up resources when graph completes."""
        logging.debug(f"running cleanup node")
        manager.cleanup()
        return {}

    # subgraphs
    webcontent_actioniser_graph = create_webcontent_actioniser_langgraph(
        checkpointer=checkpointer)

    # Add nodes
    workflow.add_node("initialize_auth", initialize_auth)
    # workflow.add_node("run_test_mode", filter_tests)
    workflow.add_node("request_url_initial", request_url_initial)
    workflow.add_node("request_url_again", request_url_again)
    workflow.add_node("human_auth", human_auth)
    workflow.add_node("analyze_page", get_prob_page_is_auth_challenge)
    workflow.add_node("set_actioniser_actions_and_query",
                      set_actioniser_actions_and_query)
    workflow.add_node("webcontent_actioniser_graph",
                      webcontent_actioniser_graph.compile(checkpointer=checkpointer))
    workflow.add_node("convert_webcontent_state_back_to_auth_state",
                      convert_webcontent_state_back_to_auth_state)
    workflow.add_node("handle_challenge", handle_challenge)
    workflow.add_node("cleanup", cleanup)

    workflow.add_edge(START, "initialize_auth")
    workflow.add_edge("initialize_auth", "request_url_initial")
    workflow.add_edge("request_url_initial", "analyze_page")
    workflow.add_conditional_edges(
        "request_url_initial", requires_human_auth)
    workflow.add_edge("request_url_again", "human_auth")
    workflow.add_edge("human_auth", "request_url_initial")
    workflow.add_conditional_edges("analyze_page", is_challenge_page, {
        "set_actioniser_actions_and_query": "set_actioniser_actions_and_query",
        "cleanup": "cleanup"
    })
    workflow.add_edge("set_actioniser_actions_and_query",
                      "webcontent_actioniser_graph")
    workflow.add_edge("webcontent_actioniser_graph",
                      "convert_webcontent_state_back_to_auth_state")
    workflow.add_edge(
        "convert_webcontent_state_back_to_auth_state", "handle_challenge")
    workflow.add_edge("handle_challenge", "analyze_page")
    workflow.add_edge("cleanup", END)

    return workflow

    # Add edges
    # workflow.set_entry_point("initialize_auth")
    workflow.add_edge("run_test_mode", "analyze_page")

    return workflow


def create_webcontent_actioniser_langgraph(checkpointer: Checkpointer) -> StateGraph:
    workflow = StateGraph(WebContentActioniserOverallState,
                          input=WebContentActioniserInputState)
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    # Step 1: Load and Extract Webpage Content

    async def load_webpage_content(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Retrieve the content of the webpage using WebBaseLoader.
        """
        logging.debug(
            f"Entering load_webpage_content node of create_webcontent_actioniser_langgraph subgraph")
        url = state.uri if state.uri_type == "url" else f"{state.uri}"
        if state.uri_type == "file_path":
            loader = BSHTMLLoader(file_path=url)
            documents = await loader.aload()
        else:
            loader = WebBaseLoader(url)
            documents = loader.load()
        # Extract the main content (usually the `page_content` field).
        return {
            "documents": [
                WebDocument(
                    url=url,
                    page_content=document.page_content,
                    # markdownified_page_source=md(document.page_content),
                    markdownified_page_source="",
                    chunks=[],
                    title=document.metadata.get("title", ""),
                    webpage_category="",
                    preprocessed_document_str=""
                )
                for document in documents
            ]
        }

    def load_html_content_from_path(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Retrieve the content of the webpage using WebBaseLoader.
        """
        logging.debug(
            f"Entering load_html_content_from_path node of create_webcontent_actioniser_langgraph subgraph")
        return load_webpage_content(state, config)

    # Step 2: Pre-process Content - Extract Relevant Sections

    # TODO: this method is completely useless in current implementation as the needing to arbitrarily define the tags to extract is impractical and impossible between different web pages
    def preprocess_content(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Filter and clean the content from the raw HTML.
        Retain key tags like <h1>, <h2>, <p>, etc., and remove noise.
        """
        logging.debug(
            f"Entering preprocess_content node of create_webcontent_actioniser_langgraph subgraph")
        for i, doc in enumerate(state.documents):
            relevant_text = []
            # Example: Extract just <h1>, <h2>, and <p> content
            html = "".join(line.strip()
                           for line in doc.page_content.split("\n"))
            for line in html.splitlines():

                if line.strip().startswith(("<h1>", "<h2>", "<p>", "<title>")):
                    relevant_text.append(line.strip())
            doc.preprocessed_document_str = "\n".join(
                relevant_text)
        return {
            "documents": state.documents
        }

    def markdownify_content(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Markdownify the content of the web page
        """
        logging.debug(
            f"Entering markdownify_content node of create_webcontent_actioniser_langgraph subgraph")
        for i, doc in enumerate(state.documents):
            markdown_content = md(doc.page_content).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            doc.markdownified_page_source = markdown_content
            if not doc.preprocessed_document_str:
                doc.preprocessed_document_str = markdown_content

        return {
            "documents": state.documents
        }

    # Step 3: Summarize or Chunk the Content

    def chunk_content(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Split content into smaller chunks suitable for an LLM.
        """
        logging.debug(
            f"Entering chunk_content node of create_webcontent_actioniser_langgraph subgraph")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=state.chunk_size, chunk_overlap=state.chunk_overlap
        )
        for i, doc in enumerate(state.documents):
            logging.debug(f"Chunking document {i} of {len(state.documents)}")
            doc.chunks = text_splitter.split_text(
                doc.preprocessed_document_str or doc.markdownified_page_source)
        return {
            "documents": state.documents
        }

    # Step 4: Generate Embeddings for Larger Contexts
    def generate_embeddings(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Create embeddings for each chunk using OpenAI embeddings.
        """
        logging.debug(
            f"Entering generate_embeddings node of create_webcontent_actioniser_langgraph subgraph")
        # embeddings = OpenAIEmbeddings()
        # vectordb = FAISS.from_texts(chunks, embeddings)
        all_chunks = [chunk for doc in state.documents for chunk in doc.chunks]
        vectordb = FAISS.from_texts(
            texts=all_chunks,
            embedding=embedding_model,
            distance_strategy=DistanceStrategy.COSINE,)

        # vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        return {
            "vectorstore": vectordb
        }

    # Step 5: Retrieve Relevant Chunks (Optional for Large Contexts)

    def retrieve_relevant_chunks(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Retrieve the top-k most relevant chunks for a given query.
        """
        logging.debug(
            f"Entering retrieve_relevant_chunks node of create_webcontent_actioniser_langgraph subgraph")
        query = state.query
        top_k = state.top_k
        if state.vectorstore_path:
            results = FAISS.load_local(
                state.vectorstore_path, embeddings=embedding_model).similarity_search(query, k=top_k)
            state.k_similar_chunks = results
            return {
                "k_similar_chunks": results
            }
        else:
            return {
                "error": "No vectorstore found"
            }

    def summarise_page(state: WebContentActioniserOverallState, config: RunnableConfig):
        """
        Use an LLM to summarise all the chunks
        """
        logging.debug(
            f"Entering summarise_page node of create_webcontent_actioniser_langgraph subgraph")
        # Define the prompt template
        prompt_template = PromptTemplate(
            input_variables=["content", "summary"],
            template=(
                "You are an expert content analyst. Below is a chunk of text from a webpage:\n\n"
                "{content}\n\n"
                "Below is the summary of the previous chunks of this web page:"
                "{summary}\n\n"
                "Please summarize both the existing summary and the new content and return the summary."
            ),
        )

        # Initialize the LLM (OpenAI GPT model)
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        llm.max_tokens = 300
        # chain = llm.with_structured_output(AuthAction)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Classify each chunk
        summary = ""
        for i, doc in enumerate(state.documents):
            for chunk in doc.chunks:
                summary = llm_chain.run(content=chunk, summary=summary)

        return {
            "summary": summary
        }

    def actionise_summary(state: WebContentActioniserOverallState, config: RunnableConfig):
        """Convert a summary into an action with proper validation."""
        logging.debug(
            f"Entering actionise_summary node of create_webcontent_actioniser_langgraph subgraph")

        prompt_template = PromptTemplate(
            input_variables=["summary", "actions"],
            template=(
                "You are tasked with bypassing an authentication challenge. Below is a summary of a webpage of the redirected authentication web page challenge:\n\n"
                "{summary}\n\n"
                "Decide which of the following actions to take to bypass the authentication challenge:\n{actions}\n\n"
                "Please provide:\n"
                "1. The most appropriate action from the list to bypass the authentication challenge\n"
                "2. Any necessary details for executing the action\n"
                "3. A brief explanation of why you chose this action\n\n"
                "Format your response to match this structure exactly."
            ),
        )

        llm = ChatOpenAI(model="gpt-4", temperature=0,
                         api_key=SecretStr(OPENAI_API_KEY))
        llm.max_tokens = 300
        chain = llm.with_structured_output(AuthAction)

        result = chain.invoke(prompt_template.format(
            summary=state.summary,
            actions="\n".join(state.predefined_actions)
        ))
        return {
            "action": AuthAction.model_validate(result)
        }

    # Step 6: Define LLM Chain to Classify Content

    # Step 8: Feedback Loop for Iterative Improvement

    # def feedback_loop(state: WebContentState, config: RunnableConfig):
    #     """
    #     Refine the pipeline by re-querying the vectorstore for better context.
    #     """
    #     improved_results = []
    #     for result in state.invalid_results:
    #         chunk = result["chunk"]
    #         print(f"Revisiting invalid classification for chunk: {
    #               chunk[:100]}...")

    #         # Retrieve additional relevant chunks to add context
    #         additional_context = retrieve_relevant_chunks(
    #             state, config)
    #         additional_text = "\n".join(
    #             [chunk.page_content for chunk in additional_context])

    #         # Re-classify with added context
    #         new_classifications = classify_content(
    #             [chunk + "\n" + additional_text],
    #             state.predefined_actions
    #         )
    #         improved_results.extend(new_classifications)

    #     return improved_results

    workflow.add_node("load_webpage_content",
                      load_webpage_content)
    workflow.add_node("preprocess_content", preprocess_content)
    workflow.add_node("markdownify_content", markdownify_content)
    workflow.add_node("chunk_content", chunk_content)
    workflow.add_node("generate_embeddings", generate_embeddings)
    workflow.add_node("retrieve_relevant_chunks", retrieve_relevant_chunks)
    workflow.add_node("summarise_page", summarise_page)
    workflow.add_node("actionise_summary", actionise_summary)

    workflow.add_edge(START, "load_webpage_content")
    workflow.add_edge("load_webpage_content", "markdownify_content")
    workflow.add_edge("markdownify_content", "chunk_content")
    # workflow.add_edge("chunk_content", "generate_embeddings")
    # workflow.add_edge("generate_embeddings", "retrieve_relevant_chunks")
    # workflow.add_edge("retrieve_relevant_chunks", "summarise_page")
    workflow.add_edge("chunk_content", "summarise_page")
    workflow.add_edge("summarise_page", "actionise_summary")
    workflow.add_edge("actionise_summary", END)
    return workflow


async def chunk_html_content_by_tagname(html_content: str, chunk_size: int = CHUNK_SIZE, by: Literal["tagname", "markdown"] = "tagname") -> List[str]:
    """Split HTML content into meaningful chunks with better text processing."""
    if by == "tagname":
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unnecessary elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Find all relevant elements first
        potential_job_content = soup.find_all(
            ['title', 'div', 'section', 'article'])

        # Filter elements based on class names
        job_content = [
            element for element in potential_job_content
            if any(term in (element.get('class') or []) for term in ['job', 'position', 'description', 'requirements'])
        ]

        if not job_content:
            # Fallback to all content if no job-specific elements found
            job_content = soup.find_all(['div', 'section', 'article'])

        chunks = []
        current_chunk = []
        current_size = 0

        for element in job_content:
            # Clean and normalize text
            text = ' '.join(element.get_text(
                separator=' ', strip=True).split())
            if not text or len(text) < 50:  # Skip very short sections
                continue

            if current_size + len(text) > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [text]
                current_size = len(text)
            else:
                current_chunk.append(text)
                current_size += len(text)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Ensure we have meaningful chunks
        if not chunks:
            logging.warning("No meaningful content chunks found")
            # Create a single chunk from all text as fallback
            all_text = soup.get_text(separator=' ', strip=True)
            return [all_text] if all_text else []

        return chunks
    else:
        markdown = md(html_content)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("thenlper/gte-small"),
            chunk_size=200,
            chunk_overlap=20,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        return text_splitter.split_text(markdown)


def merge_job_info(existing: Optional[JobPost], new_info: JobPost) -> JobPost:
    """Merge new job information with existing information."""
    if not existing:
        return new_info

    merged_dict = existing.model_dump()
    new_dict = new_info.model_dump()

    # Merge logic for each field
    for field, value in new_dict.items():
        if value is not None and (field not in merged_dict or merged_dict[field] is None):
            merged_dict[field] = value
        elif field == 'description' and value:
            # Concatenate descriptions
            existing_desc = merged_dict.get('description', '')
            if existing_desc and value not in existing_desc:
                merged_dict['description'] = f"{existing_desc}\n{value}"

    return JobPost.model_validate(merged_dict)


def save_page_source(url: str, current_url: str, content: str) -> str:
    """
    Save page source to a file with a sanitized name pattern.
    Returns the path to the saved file.
    """
    # Create page_source directory if it doesn't exist
    page_source_dir = Path("./cv_ats_assistant/page_source")
    page_source_dir.mkdir(parents=True, exist_ok=True)

    # Parse URL components
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # Create a shortened hash of the path to avoid filename length issues
    path_hash = hashlib.md5(parsed_url.path.encode()).hexdigest()[:8]

    # Generate datetime string
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"{domain}_{datetime_str}_{path_hash}_page_source.html"

    # Ensure filename is valid
    filename = quote(filename, safe='_-.')

    content += f"\n\n<!-- Driver Current URL: {current_url} -->"

    # Save the content
    file_path = page_source_dir / filename
    try:
        file_path.write_text(content, encoding='utf-8')
        logging.info(f"Saved page source to: {file_path}")
        return str(file_path)
    except Exception as e:
        logging.error(f"Failed to save page source: {str(e)}")
        return ""


def load_page_source(url: str, datetime_str: Optional[str] = None) -> Optional[str]:
    """
    Load a saved page source for testing purposes.
    If datetime_str is provided, loads that specific version,
    otherwise loads the most recent version.
    """
    page_source_dir = Path("./cv_ats_assistant/page_source")
    if not page_source_dir.exists():
        logging.error("Page source directory does not exist")
        return None

    domain = urlparse(url).netloc
    path_hash = hashlib.md5(urlparse(url).path.encode()).hexdigest()[:8]

    if datetime_str:
        # Look for specific version
        pattern = f"{domain}_{datetime_str}_{path_hash}_page_source.html"
    else:
        # Look for any version
        pattern = f"{domain}_*_{path_hash}_page_source.html"

    matching_files = sorted(page_source_dir.glob(pattern))
    if not matching_files:
        logging.error(f"No saved page source found for URL: {url}")
        return None

    # Get most recent file if no specific datetime provided
    file_path = matching_files[-1]
    try:
        content = file_path.read_text(encoding='utf-8')
        logging.info(f"Loaded page source from: {file_path}")
        return content
    except Exception as e:
        logging.error(f"Failed to load page source: {str(e)}")
        return None


def create_job_extraction_graph(checkpointer: Checkpointer) -> StateGraph:
    """Create a LangGraph for job extraction"""
    workflow = StateGraph(JobGraphOverallState,
                          input=JobGraphInputState, output=JobGraphOutputState)
    llm = ChatOpenAI(temperature=0, api_key=SecretStr(OPENAI_API_KEY))
    chain: RunnableBinding = llm.with_structured_output(
        JobInfo)  # type: ignore

    def check_chunks_visited(state: JobGraphOverallState, config: RunnableConfig) -> Literal["extract_job_info_from_chunks"] | str:
        """Check if the chunks have been visited"""
        if not state.chunks_to_visit:
            if state.incorrect_job_info_extraction:
                return 'improve_job_info_extraction_prompt'
            return END
        return "extract_job_info_from_chunks"

    def improve_job_info_extraction_prompt(state: JobGraphOverallState, config: RunnableConfig):
        """Improve the job information extraction prompt"""
        with open("./cv_ats_assistant/corrected_job_info.txt", "r") as f:
            corrected_job_info = f.read()
        prompt_template = PromptTemplate(
            input_variables=["corrected_job_info"],
            template="""
            You are an expert AI Prompt Engineer.
            You are currently given a prompt for a job information extraction task but this prompt failed to extract the job information correctly.
            You are given a corrected job information that you need to use to update the prompt to extract the job information correctly from the page source.
            Here is the corrected job information:
            {corrected_job_info}
            You will be given the page source in chunks below:
            """
        )
        prompt = prompt_template.format(
            corrected_job_info=corrected_job_info, page_source=state.chunks_to_visit)
        messages = [
            SystemMessage(content=prompt),
            *[HumanMessage(content=chunk) for chunk in state.chunks_to_visit],
            HumanMessage(
                content="Please only return the updated prompt and nothing else.")
        ]
        response = llm.invoke(messages)
        with open("./cv_ats_assistant/corrected_job_info_prompt.txt", "w") as f:
            f.write(str(response.content))
        return {
            "prompt_job_info_extraction_template": response.content,
            "chunks_to_visit": [corrected_job_info]
        }

    def extract_job_info_from_corrected_job_info(state: JobGraphOverallState, config: RunnableConfig):
        """Extract job information from the corrected job information"""
        prompt_t = f"""
            You are a job description extractor. You are given a text file that contains a job post within it. A job post contains a job name, a job description, a company name, a company domain url, a company own job posting url, a hiring manager name, a hiring manager url, a company about content, a company values, a key skills section and a keywords section.
            You need to extract the job information from the this context.
            The job post text file is:
            """
        prompt = f"""{prompt_t}{" ".join(state.chunks_to_visit[0])}"""
        result: JobInfo = chain.invoke(prompt)
        return {
            "job_info": result,
            "chunks_to_visit": []
        }

    def extract_job_info_from_chunks(state: JobGraphInputState, config: RunnableConfig):
        """Extract job information from the chunks"""
        next_prompt = ""
        summarise_current_jobinfo = state.job_info_summary
        if not state.chunks_to_visit:
            return {}
        i = 1
        next_chunk = state.chunks_to_visit[0:i]
        prompt_t = f"""
            You are a job description extractor. You are given a web page that contains a job post within it. A job post contains a job name, a job description, a company name, a company domain url, a company own job posting url, a hiring manager name, a hiring manager url, a company about content, a company values, a key skills section and a keywords section.
            You are given a chunk of this webpage's source and a summary of the job information you have already extracted in previous steps.
            You need to extract the job information from the this context. You should only modify the existing extracted job information if the new information is more accurate or complete.
            The current job information is: {summarise_current_jobinfo}
            The next chunk of job information is:
            """
        prompt = f"""{prompt_t}{" ".join(next_chunk)}"""
        while i < len(state.chunks_to_visit) and len(prompt) < (10000 - CHUNK_SIZE):
            i += 1
            next_chunk = state.chunks_to_visit[:i]
            prompt = f"""{prompt_t}{" ".join(next_chunk)}"""

        # TODO: add another node to ask human to check job info is correct, if not, set incorrect job info extraction on graph state and then nav to node that asks the user to input the correct job info in a corrected_job_info.txt file which is then read in, llm is asked to adust this prompt so that it would be able to read this job information in from this page_source whilst keeping the existing functionality.
        result: JobInfo = chain.invoke(prompt)
        return {
            "job_info": result,
            "chunks_to_visit": state.chunks_to_visit[i:],
            "prompt_job_info_extraction_template": prompt_t
        }

    def summarise_job_info(state: JobGraphOverallState, config: RunnableConfig):
        """Summarise the job information"""
        jbi = state.job_info
        summarised_job_info = f"""
        ----START----
        Job Name: {jbi.job_name}
        Job URL: {jbi.job_url}
        Company Domain URL: {jbi.company_domain_url}
        Company Own Job Posting URL: {jbi.company_own_job_posting_url}
        Company Name: {jbi.company_name}
        Hiring Manager Name: {jbi.hiring_manager_name}
        Hiring Manager URL: {jbi.hiring_manager_url}
        Company About Content: {jbi.company_about_content}
        Company Values: {jbi.company_values}
        Key Skills: {jbi.key_skills}
        Keywords: {jbi.keywords}
        Job Description: {jbi.full_job_post_on_source}
        ----END----
        """
        return {
            "job_info_summary": summarised_job_info
        }

    workflow.add_node("extract_job_info_from_chunks",
                      extract_job_info_from_chunks)
    workflow.add_node("summarise_job_info", summarise_job_info)
    workflow.add_node("improve_job_info_extraction_prompt",
                      improve_job_info_extraction_prompt)
    workflow.add_node("extract_job_info_from_corrected_job_info",
                      extract_job_info_from_corrected_job_info)

    workflow.add_edge(START, "extract_job_info_from_chunks")
    workflow.add_conditional_edges("summarise_job_info", check_chunks_visited)
    workflow.add_edge("improve_job_info_extraction_prompt",
                      "extract_job_info_from_corrected_job_info")
    workflow.add_edge(
        "extract_job_info_from_corrected_job_info", "summarise_job_info")
    return workflow


def create_CV_curation_graph(checkpointer: Checkpointer):
    """Create the LangGraph to curate a CV based on a job post for Case A"""

    # Subgraphs
    # authenticate_url node sub graph
    authenticate_url_sub_graph = create_auth_graph(checkpointer)
    # extract_job_info node sub graph
    extract_job_info_sub_graph = create_job_extraction_graph(checkpointer)
    # dummy sub graph node
    # dummy_sub_graph = create_dummy_graph(checkpointer=checkpointer,
    #     AuthenticationGraphOutputState,
    #     AuthenticationGraphOutputState(
    #         # job_post_url="https://www.linkedin.com/jobs/collections/similar-jobs/?currentJobId=4116841841&originToLandingJobPostings=4116841841",
    #         current_url="https://www.linkedin.com/jobs/collections/similar-jobs/?currentJobId=4116841841&originToLandingJobPostings=4116841841",
    #         current_title="LinkedIn Job Postings",
    #         current_page_source="",
    #         domain="www.linkedin.com"
    #     )
    # )

    async def chunk_url(state: GraphState, config: RunnableConfig):
        """Chunk the URL"""
        if not state.current_page_source:
            return {
                "error": "No page source found in 'chunk_url' node"
            }
        chunks = await chunk_html_content_by_tagname(state.current_page_source, by="markdown")
        unique_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk not in unique_chunks:
                unique_chunks.append(chunk)
        url_path_ = urlparse(state.current_url).path.split("/")[-1]
        with open(f"./cv_ats_assistant/job_posts/markdown/{url_path_}.md", "w") as f:
            f.write("\n".join(unique_chunks))
        return {
            "chunks_to_visit": unique_chunks
        }

    def embed_cv(state: GraphState, config: RunnableConfig) -> GraphStateDict:
        """Create embeddings for the CV document"""
        if not state.cv_path:
            raise ValueError("CV path is required")
        cv_vectorstore_manager = helpers.get_cv_vectorstore(state.cv_path)

        logging.info(f"Created and cached new CV embeddings to {
                     cv_vectorstore_manager.vectorstore_path}")

        return {
            "cv_embedding": cv_vectorstore_manager.vectorstore_path,
            "cv_path": str(cv_vectorstore_manager.cv_path)
        }

    def suggest_cv_changes(state: GraphState, config: RunnableConfig):
        """Suggest changes to the CV based on job requirements"""
        llm = ChatOpenAI(temperature=0, api_key=SecretStr(OPENAI_API_KEY))
        try:
            assert state.current_url, "Current URL is required to suggest changes to the CV"
            assert state.current_page_source, "Current page source is required to suggest changes to the CV"
            assert state.cv_embedding, "CV embedding is required to suggest changes to the CV"
            assert state.job_info, "Job info is required to suggest changes to the CV"
            assert state.job_info_summary, "Job info summary is required to suggest changes to the CV"
        except AssertionError as e:
            return {
                "error": str(e)
            }

        class CVAlteration(BaseModel):
            alteration: str
            location: str
            importance: int

        # Retrieve CV content from the embedding
        if not state.cv_embedding:
            return {
                "error": "CV embedding path is required to suggest changes to the CV"
            }
        if not state.cv_path:
            return {
                "error": "CV path is required to suggest changes to the CV"
            }

        cv_vectorstore_manager = helpers.get_cv_vectorstore(state.cv_path)
        cv_content = cv_vectorstore_manager.cv_vectorstore.similarity_search("", k=1)[
            0].page_content

        messages = [
            SystemMessage(content="You are a hiring manager for a company that is hiring for a job and you are tasked with suggesting changes to a candidate's CV based on the job requirements and the job information you have extracted to ensure that the candidate is the perfect fit for the job."),
            HumanMessage(content=f"The current job information is: {
                         state.job_info_summary}"),
            HumanMessage(content="""
1. Extract all key words, from this job description into an ordered list, most important first dictated by semantic similarity to the job name and number of occurences in the job description and whether the key word appears in the required skills section of the job description.
2. List all the key words in the list above in order importance that also appear (or have a semantically very similar token) in my CV which is attached to the project space. If you cannot see my CV, stop at this point and ask me to provide it first.
3. List the 5 most important key words that I need to include from point 1 in my CV that I have not included.
4. Give my CV a score out of K where K:= is defined as the number of required skills appearing in the job description (which does not include optional or nice-to-have skills) in the Job Description. Give my CV 1 point from K possible points for each skill that I also mention in my CV from the required skills only. Do not hallucinate. Sum the points and return a response as "{i}/K".format(my_skills_required).
5. Please give my CV a completely separate score to all other scoring above, this score is a tuple[int,int] where each int \\in [0,100] inclusive and is a measure of:
    a) how well my CV semantically aligns to the job name as score out of 100
    b) the semantic similarity score out of 100 between my Title name on my CV at the top of the summary section at the beginning of the document and the job name
"""),
            HumanMessage(
                content=f"Here is the content of my CV:\n\n<Document>{cv_content}</Document>"),
        ]
        register_prompt_template(
            "case_a.suggest_cv_changes",
            ChatPromptTemplate.from_messages(messages),
            version="0.0.1+alpha")
        result: CVAlteration = llm.with_structured_output(
            CVAlteration).invoke(messages)  # type: ignore
        return {
            "suggested_changes": [result.model_dump(mode="python")]
        }

    def check_linkedin_consistent_with_cv(state: GraphState, config: RunnableConfig):
        """Check if the LinkedIn Profile is consistent with the CV"""
        llm = ChatOpenAI(temperature=0, api_key=SecretStr(OPENAI_API_KEY))
        if not state.cv_embedding:
            return {
                "error": "CV embedding path is required to suggest changes to the CV"
            }
        if not state.cv_path:
            return {
                "error": "CV path is required to suggest changes to the CV"
            }

        cv_vectorstore_manager = helpers.get_cv_vectorstore(state.cv_path)
        cv_content = cv_vectorstore_manager.cv_vectorstore.similarity_search("", k=1)[
            0].page_content
        linkedin_profile_path = Path(
            "/Users/joey/Downloads/LinkedInProfile-2025-01-14-18-03.pdf")
        loader = PyPDFLoader(str(linkedin_profile_path))
        pages = loader.load()
        linkedin_profile_content = pages[0].page_content

        class LinkedInProfileAlteration(BaseModel):
            alteration: str
            location: str
            importance: int
        messages = [
            SystemMessage(content="You are a job description extractor. You are given a job description and a summary of the job information you have already extracted. You need to extract the job information from the job description. You should only modify the existing extracted job information if the new information is more accurate or complete."),
            HumanMessage(content="""
First:
Please compare my linkedin profile (attached as LinkedInProfile-YYYY-MM-DD-HH-MM) to my CV show all the points that are inconsistent between the 2 and the product manager roles issue with founder etc. 
Use my CV as the more up to date document and make recommendations for my LinkedIn Profile to make it consistent with my CV.

Secondly:
Is the content shared on my LinkedIn profile both descriptive and relevant. Please give the profile a dictionary of scores str to int out of 100 where 0 is bad and 100 is perfect and 50 is average and score my linked in profile on how descriptive the content is as relates to the job title and job description, how relevant the content is as relates to the job title and job description, how relevant. Please then also return a list of suggestions where each suggestion contains a suggested alteration to make to my LinkedIn profile along with the precise location on my LinkedIn profile to make the alteration and the importance of making this alteration using 1, 2 or 3 exclamation marks to indicate increasing levels of important for appealing to hiring managers that are hiring for this job name and description. 

Here is the Job Name and Description
<div>
    Job Name: {job_name}
    Company Name: {company_name}
    Location: {location}

    Job Description:
    {job_description}
</div>
Don't do anything until I have attached my Linkedin profile as a PDF and my CV as a PDF in the next message
"""),
            # TODO: Add the LinkedIn Profile and CV as attachments to the messages
            HumanMessage(content=f"Here is the content of my CV:\n\n<Document>{
                         cv_content}</Document>"),
            HumanMessage(content=f"Here is my LinkedIn Profile:\n\n<Document>{
                         linkedin_profile_content}</Document>"),
        ]
        register_prompt_template(
            "case_a.check_linkedin_consistent_with_cv",
            ChatPromptTemplate.from_messages(messages),
            version="0.0.1+alpha")
        result: LinkedInProfileAlteration = llm.with_structured_output(
            LinkedInProfileAlteration).invoke(messages)  # type: ignore
        return {
            "linkedin_profile_alteration": result.model_dump_json()
        }

    def human_interrupt(state: GraphState, config: RunnableConfig):
        """Handle user interaction for CV changes"""
        print("Suggested changes to your CV:")
        for i, change in enumerate(state.suggested_changes or []):
            print(f"{i+1}. {change['description']}")

        approved = input(
            "Do you approve these changes? (y/n): ").lower() == 'y'

        if approved:
            return {
                "user_approved_changes": state.suggested_changes,
                "user_approved_changes_str": json.dumps(state.user_approved_changes)
            }
        return {}

    def alter_cv(state: GraphState, config: RunnableConfig):
        """Alter the CV based on the suggested changes"""
        if not state.user_approved_changes or not state.user_approved_changes_str or not state.cv_path:
            return {
                "error": "No approved changes or CV path not provided"
            }

        try:
            # Parse the JSON string of approved changes
            changes = json.loads(state.user_approved_changes_str)

            # Create output path with timestamp
            cv_path = Path(state.cv_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = cv_path.parent / \
                f"{cv_path.stem}_modified_{timestamp}{cv_path.suffix}"

            # Create PDF reader and writer objects
            reader = PdfReader(state.cv_path)
            writer = PdfWriter()

            # Track if any changes were made
            changes_made = False

            # Copy all pages from the original PDF
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                modified_page = False

                # Apply each approved change to the text
                for change in changes:
                    location = change.get('location', '')
                    alteration = change.get('alteration', '')
                    importance = change.get('importance', 1)

                    if location and location in text:
                        logging.info(f"Found text to modify on page {
                                     page_num + 1}: {location[:50]}...")

                        # Create a new page with the modified content
                        from reportlab.pdfgen import canvas
                        from reportlab.lib.pagesizes import letter
                        from io import BytesIO

                        # Create a temporary buffer for the new page
                        packet = BytesIO()
                        can = canvas.Canvas(packet, pagesize=letter)

                        # Get the original page dimensions
                        original_width = float(page.mediabox.width)
                        original_height = float(page.mediabox.height)

                        # Scale canvas to match original page size
                        can.setPageSize((original_width, original_height))

                        # Write the modified text
                        # Note: This is a simplified approach - you might need more sophisticated
                        # text positioning and formatting
                        modified_text = text.replace(location, alteration)
                        can.drawString(50, original_height - 50, modified_text)
                        can.save()

                        # Move to the beginning of the buffer
                        packet.seek(0)

                        # Create a new PDF with the modified content
                        new_pdf = PdfReader(packet)
                        modified_page = new_pdf.pages[0]

                        # Merge the modified content with the original page
                        page.merge_page(modified_page)
                        changes_made = True
                        modified_page = True

                        logging.info(
                            f"Applied change with importance {importance}")

                # Add the page (modified or original) to the writer
                writer.add_page(page)

            if not changes_made:
                logging.warning(
                    "No changes were applied to the CV - could not find specified locations")
                return {
                    "error": "No changes were applied - could not find specified locations in CV"
                }

            # Save the modified PDF
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)

            logging.info(f"Modified CV saved to: {output_path}")
            return {
                "altered_cv_path": str(output_path),
                "changes_applied": changes
            }

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing approved changes JSON: {e}")
            return {
                "error": f"Invalid JSON format in approved changes: {e}"
            }
        except Exception as e:
            logging.error(f"Error modifying CV: {e}")
            return {
                "error": f"Failed to modify CV: {e}"
            }

    def score_cv(state: GraphState, config: RunnableConfig):
        """Score the CV against job requirements"""
        llm = ChatOpenAI(temperature=0, api_key=SecretStr(OPENAI_API_KEY))
        # Retrieve CV content from the embedding
        if not state.cv_embedding:
            return {
                "error": "CV embedding path is required to suggest changes to the CV"
            }
        if not state.cv_path:
            return {
                "error": "CV path is required to suggest changes to the CV"
            }

        cv_vectorstore_manager = helpers.get_cv_vectorstore(state.cv_path)
        cv_content = cv_vectorstore_manager.cv_vectorstore.similarity_search("", k=1)[
            0].page_content
        messages = [
            SystemMessage(content="You are a hiring manager for a company that is hiring for a job and you are tasked with scoring a candidate's CV based on the job requirements and the job information you have extracted to ensure that the candidate is the perfect fit for the job."),
            HumanMessage(content=f"The current job information is: {
                         state.job_info_summary}"),
            HumanMessage(content=f"Here is the content of my CV:\n\n<Document>{
                         cv_content}</Document>"),
        ]
        register_prompt_template(
            "case_a.score_cv",
            ChatPromptTemplate.from_messages(messages),
            version="0.0.1+alpha")
        result: CVScores = llm.with_structured_output(
            CVScores).invoke(messages)  # type: ignore
        # scores = CVScores(
        #     experience_score=85,  # Implement actual scoring logic
        #     grammar_score=90,
        #     skills_score=85,
        #     ats_score=88,
        #     clarity_score=92,
        #     role_fit_score=87,
        #     overall=88
        # )
        return {
            "cv_scores": result.model_dump_json()
        }

    def save_cv(state: GraphState, config: RunnableConfig):
        """Save the modified CV"""
        if state.altered_cv_path:
            print(f"Updated CV saved to: {state.altered_cv_path}")
        return {"job_info": state.job_info}

    workflow = StateGraph(
        GraphState, input=GraphInputState, output=GraphOutputState)

    # workflow.add_node("dummy_graph", dummy_sub_graph.compile(checkpointer=checkpointer))
    workflow.add_node("authenticate_url", authenticate_url_sub_graph.compile(
        checkpointer=checkpointer, interrupt_after=["request_url_again"]))
    workflow.add_node("chunk_url", chunk_url)
    workflow.add_node("extract_job_info", extract_job_info_sub_graph.compile(
        checkpointer=checkpointer, interrupt_after=["summarise_job_info"]))
    workflow.add_node("embed_cv", embed_cv)
    workflow.add_node("suggest_cv_changes", suggest_cv_changes)
    workflow.add_node("human_interrupt", human_interrupt)
    workflow.add_node("alter_cv", alter_cv)
    workflow.add_node("score_cv", score_cv)
    workflow.add_node("save_cv", save_cv)
    # TODO: Add nodes to score my linkedin profile and nodes to confirm that the LinkedIn Profile is consistent with my CV.

    workflow.add_edge(START, "authenticate_url")
    # workflow.set_entry_point("authenticate_url")
    workflow.add_edge("authenticate_url", "chunk_url")
    workflow.add_edge("chunk_url", "extract_job_info")
    workflow.add_edge(START, "embed_cv")
    # Use a list to make the graph wait for both start key nodes to be reached before continuing to end_key.
    workflow.add_edge(
        start_key=["embed_cv", "extract_job_info"],
        end_key="suggest_cv_changes")
    workflow.add_edge("suggest_cv_changes", "human_interrupt")
    workflow.add_edge("human_interrupt", "score_cv")

    workflow.add_conditional_edges(
        "score_cv",
        lambda x: "alter_cv" if x.cv_scores.overall >= 90 else "suggest_cv_changes",
        {
            "alter_cv": "alter_cv",
            "suggest_cv_changes": "suggest_cv_changes"
        }
    )

    # Modify the edges - avoid creating a cycle
    # Go directly to save after alterations
    workflow.add_edge("alter_cv", "save_cv")
    workflow.add_edge("save_cv", END)

    async def set_graph_config(
            graph: CompiledStateGraph,
            graph_args: dict[str, Any] | None,
            checkpointer: AsyncSqliteSaver | SqliteSaver,
            # sqlite_db_path: str,
            checkpoint_id: str,
            thread_id: str,
            use_last_thread: bool = False):

        # async with AsyncSqliteSaver.from_conn_string(sqlite_db_path) as checkpointer:

        graph_config: RunnableConfig = {"configurable": {}}
        thread: Thread | dict[str, Any] = {
            "thread_id": thread_id}
        graph_config = {"configurable": thread}
        if checkpoint_id:
            graph_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint_id}}
            previous_state, checkpoint = await helpers.get_state_history_checkpoint_with_id(checkpointer, graph, graph_config, checkpoint_id)
            if checkpoint:
                # graph_config = {
                #     'configurable': checkpoint.config['configurable']}
                graph_config = checkpoint.config
                print(f"Using checkpoint with timestamp {
                    checkpoint.checkpoint['ts']} and graph_config: {graph_config}")
            else:
                raise ValueError(
                    f"Checkpoint with id {checkpoint_id} not found")
        else:
            existing_state = await checkpointer.aget(graph_config)
            if existing_state:
                new_graph_config: RunnableConfig = {
                    "configurable": {
                        "thread_id": thread['thread_id'],
                        "checkpoint_ns": "",
                        "checkpoint_id": existing_state.get('id', ""),

                    }
                }
            else:
                new_graph_config = graph_config
            if use_last_thread:
                graph_config = new_graph_config or graph_config
                logging.info(f"Using memory checkpoint with thread_id: {
                    thread['thread_id']} and graph_state: {existing_state}")
            else:
                logging.info(f"Using new graph state with graph_config: {
                    graph_config}")

        graph.debug = True

        async def _stream_graph(runnable: Callable[[], AsyncIterator[dict[str, Any] | Any]] | None = None):
            stream_mode: Literal["values", "updates", "debug", "messages",
                                 "custom"] | StreamMode | list[StreamMode] | None = "values"
            # BUG: Exception has occurred: TypeError
            # Object of type FAISS is not serializable
            async for event in (runnable() if runnable else graph.astream(
                input=None if graph_config.get('configurable', {}).get(
                    'checkpoint_id') else graph_args,
                config=graph_config,
                stream_mode=stream_mode,
            )):
                yield event
                # BUG: the issue is that our sub graphs dont have checkpoints and we therefore have to start any subgraph that we were half way through again as the subgraph will not have been saved.
                # if stream_mode == "values":
                #     pprint(event)
                # elif stream_mode == 'messages':
                #     event['messages'][-1].pretty_print()
                # elif stream_mode == 'updates':
                #     pprint(event)
                # elif stream_mode == 'debug':
                #     pprint(event)
                # elif stream_mode == 'custom':
                #     pprint(event)

        return graph_config, _stream_graph

    async def interruption_handler(
            graph: CompiledStateGraph,
            thread: RunnableConfig,
            checkpointer: AsyncSqliteSaver | SqliteSaver,
    ):
        """Handle interruptions"""
        all_states = [s async for s in graph.aget_state_history(thread)]
        next_node = next(iter(all_states[-1].next), "")
        # if len(next_node) == 1 and next_node[0] == END:
        #     return
        # NOTE: hack to get the previous node using time travel and next node
        to_replay = all_states[-2]
        prev_node = next(iter(to_replay.next), "")
        if prev_node.endswith("summarise_job_info"):
            incorrect_job_info_extraction = input(
                "Please check the job info and if necessary update the path at cv_ats_assistant/corrected_job_info.txt and respond y if you have done so else enter") == "y"
            graph.update_state(
                thread,
                {"incorrect_job_info_extraction": incorrect_job_info_extraction, }
            )
        elif prev_node.endswith("suggest_cv_changes"):
            user_approved_changes = input(
                "Please check the suggested changes and if necessary update the path at cv_ats_assistant/user_approved_changes.txt and respond y if you have done so else enter") == "y"
            graph.update_state(
                thread,
                {"user_approved_changes": user_approved_changes, }
            )

        _unused_graph_config, stream_graph_fn = await set_graph_config(
            graph=graph,
            graph_args=None,
            checkpointer=checkpointer,
            checkpoint_id=thread.get(
                'configurable', {}).get('checkpoint_id', ""),
            thread_id=thread.get('configurable', {}).get('thread_id', ""),
            use_last_thread=False
        )
        return stream_graph_fn

    def compiler(graph: StateGraph, checkpointer: AsyncSqliteSaver | SqliteSaver):
        nodes_names = [node for node in graph.nodes.keys()]
        interrupt_before = [
            "human_interrupt",
        ]
        interrupt_after = [
            "suggest_cv_changes",
            "summarise_job_info",
            "request_url_again",
        ]
        for n in nodes_names:
            if isinstance(_subgraph := graph.nodes[n].runnable, CompiledStateGraph):
                logging.debug(
                    f"Adding interrupt_before|after nodes to Subgraph:'{n}'")
                subgraph: CompiledStateGraph = _subgraph
                sns = [sn for sn in interrupt_before if sn in subgraph.nodes.keys(
                ) and sn not in subgraph.interrupt_before_nodes]
                subgraph.interrupt_before_nodes = list(
                    subgraph.interrupt_before_nodes) + sns if subgraph.interrupt_before_nodes != "*" else "*"
                sns = [sn for sn in interrupt_after if sn in subgraph.nodes.keys(
                ) and sn not in subgraph.interrupt_after_nodes]
                subgraph.interrupt_after_nodes = list(
                    subgraph.interrupt_after_nodes) + sns if subgraph.interrupt_after_nodes != "*" else "*"

        return graph.compile(
            checkpointer=checkpointer,
            interrupt_before=[n for n in interrupt_before if n in nodes_names],
            interrupt_after=[n for n in interrupt_after if n in nodes_names])

    return workflow, set_graph_config, interruption_handler, compiler
