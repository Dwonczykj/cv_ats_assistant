import base64
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypedDict

from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from cv_ats_assistant.browser.llm_deepseek import DeepSeekR1ChatOpenAI
from cv_ats_assistant.case_a import CHUNK_SIZE
from cv_ats_assistant.models import LinkedInCredentials
from markdownify import markdownify as md
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr
from transformers import AutoTokenizer
# import gradio as gr

LLM_TYPE = ChatAnthropic | ChatGoogleGenerativeAI | ChatOllama | AzureChatOpenAI | ChatOpenAI | DeepSeekR1ChatOpenAI


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


def get_llm_model(provider: str, **kwargs):
    """
    获取LLM 模型
    :param provider: 模型类型
    :param kwargs:
    :return:
    """
    if provider == "anthropic":
        if not kwargs.get("base_url", ""):
            base_url = "https://api.anthropic.com"
        else:
            base_url = kwargs.get("base_url")

        if not kwargs.get("api_key", ""):
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-3-5-sonnet-20240620"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv(
                "OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            base_url = kwargs.get("base_url")

        if not kwargs.get("api_key", ""):
            api_key = os.getenv("OPENAI_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "deepseek":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("DEEPSEEK_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")

        if not kwargs.get("api_key", ""):
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
    elif provider == "gemini":
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("GOOGLE_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            google_api_key=api_key,
        )
    elif provider == "ollama":
        return ChatOllama(
            model=kwargs.get("model_name", "qwen2.5:7b"),
            temperature=kwargs.get("temperature", 0.0),
            num_ctx=kwargs.get("num_ctx", 32000),
            base_url=kwargs.get("base_url", "http://localhost:11434"),
        )
    elif provider == "azure_openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version="2024-05-01-preview",
            azure_endpoint=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Predefined model names for common providers
MODEL_PROVIDER_NAMES_TYPE = Literal[
    "anthropic",
    "openai",
    "deepseek",
    "gemini",
    "ollama",
    "azure_openai"]


class ModelNameLookup(TypedDict):
    anthropic: Literal["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]
    openai: Literal["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    deepseek: Literal["deepseek-chat", "deepseek-reasoner"]
    gemini: Literal["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp",
                    "gemini-1.5-flash-latest", "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-1219"]
    ollama: Literal["qwen2.5:7b", "llama2:7b", "llama3.1:8b", "llama3.1:13b"]
    azure_openai: Literal["gpt-4o", "gpt-4", "gpt-3.5-turbo"]


# model_names: dict[MODEL_PROVIDER_NAMES_TYPE, list[str]] = {
model_link_list = {
    "gemini": "https://ai.google.dev/gemini-api/docs/models/gemini"
}

model_names: ModelNameLookup = {
    "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-o1", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-v3",],
    "kluster_ai": [
        "deepseek-ai/DeepSeek-R1",
        "klusterai/Meta-Llama-3.1-405B-Instruct-Turbo",
        "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
    ],
    "openrouter": ['mistralai/mistral-large-2411',
                   'meta-llama/llama-3.2-90b-vision-instruct:free',
                   'anthropic/claude-3.5-haiku-20241022:beta',
                   'anthropic/claude-2.1',
                   'anthropic/claude-2.0:beta',
                   'perplexity/llama-3-sonar-large-32k-chat',
                   'sao10k/l3.1-euryale-70b',
                   'meta-llama/llama-3-8b',
                   'microsoft/phi-3-medium-4k-instruct',
                   'deepseek/deepseek-r1-distill-qwen-14b',
                   'jondurbin/bagel-34b',
                   'mistralai/mistral-tiny',
                   'perplexity/llama-3.1-sonar-small-128k-online',
                   'openai/gpt-4-vision-preview',
                   'openai/gpt-3.5-turbo',
                   'meta-llama/llama-3.1-405b',
                   'allenai/olmo-7b-instruct',
                   'meta-llama/llama-3.2-11b-vision-instruct:free',
                   'anthropic/claude-3.5-haiku-20241022',
                   'perplexity/sonar',
                   'openai/gpt-4-0314',
                   'mistralai/mistral-large',
                   'cohere/command-r-plus-08-2024',
                   'meta-llama/llama-3.1-405b-instruct:nitro',
                   'eva-unit-01/eva-qwen-2.5-72b',
                   'qwen/qwen-72b-chat',
                   'sophosympatheia/rogue-rose-103b-v0.2:free',
                   'openai/gpt-4o-2024-05-13',
                   'google/palm-2-chat-bison-32k',
                   'anthropic/claude-1',
                   'openai/gpt-4o-mini-2024-07-18',
                   'open-orca/mistral-7b-openorca',
                   '01-ai/yi-large-fc',
                   'meta-llama/codellama-34b-instruct',
                   'meta-llama/llama-3.1-8b-instruct',
                   'qwen/qwen-2.5-72b-instruct',
                   'openai/gpt-4',
                   'mistralai/mistral-small',
                   'nousresearch/nous-hermes-llama2-70b',
                   'google/gemini-pro-vision',
                   'mistralai/codestral-2501',
                   '01-ai/yi-6b',
                   'perplexity/llama-3.1-sonar-huge-128k-online',
                   'openai/gpt-4o-mini',
                   'microsoft/wizardlm-2-8x22b',
                   'mistralai/mixtral-8x22b',
                   'qwen/qwen-2-vl-72b-instruct',
                   'meta-llama/llama-3.2-3b-instruct:free',
                   'x-ai/grok-beta',
                   'nousresearch/hermes-3-llama-3.1-405b',
                   'microsoft/phi-3-mini-128k-instruct',
                   'meta-llama/llama-3.1-405b-instruct',
                   'openai/shap-e',
                   'cohere/command-r',
                   'mistralai/mistral-7b-instruct-v0.1',
                   'qwen/qwen-2-vl-7b-instruct',
                   'nvidia/llama-3.1-nemotron-70b-instruct',
                   'google/palm-2-codechat-bison-32k',
                   'anthropic/claude-instant-1',
                   'sao10k/fimbulvetr-11b-v2',
                   'sao10k/l3.3-euryale-70b',
                   'gryphe/mythomist-7b',
                   'openai/gpt-3.5-turbo-0125',
                   'google/gemini-pro-1.5',
                   'anthropic/claude-3.5-haiku',
                   'microsoft/phi-4',
                   'meta-llama/llama-3-8b-instruct:extended',
                   'qwen/qvq-72b-preview',
                   'anthropic/claude-3-sonnet:beta',
                   'anthropic/claude-3.5-haiku:beta',
                   'mistralai/mixtral-8x22b-instruct',
                   'cognitivecomputations/dolphin-mixtral-8x22b',
                   'openai/gpt-4o-2024-08-06',
                   'thedrummer/rocinante-12b',
                   'cohere/command-r-08-2024',
                   'x-ai/grok-2',
                   'openai/o1-preview-2024-09-12',
                   'gryphe/mythomax-l2-13b:free',
                   'mistralai/mistral-7b-instruct',
                   'eva-unit-01/eva-llama-3.33-70b',
                   'anthropic/claude-1.2',
                   'meta-llama/llama-3-8b-instruct:nitro',
                   'anthropic/claude-2:beta',
                   'austism/chronos-hermes-13b',
                   'amazon/nova-lite-v1',
                   'undi95/toppy-m-7b',
                   'x-ai/grok-vision-beta',
                   'neversleep/llama-3.1-lumimaid-70b',
                   'anthropic/claude-3-opus:beta',
                   'nousresearch/nous-hermes-llama2-13b',
                   'openai/gpt-3.5-turbo-instruct',
                   'teknium/openhermes-2-mistral-7b',
                   'undi95/remm-slerp-l2-13b',
                   'liquid/lfm-7b',
                   'thedrummer/unslopnemo-12b',
                   'meta-llama/llama-3.2-90b-vision-instruct',
                   'nousresearch/nous-hermes-2-vision-7b',
                   'nousresearch/nous-capybara-7b',
                   'cognitivecomputations/dolphin-mixtral-8x7b',
                   'meta-llama/llama-3-70b-instruct:nitro',
                   'snowflake/snowflake-arctic-instruct',
                   '01-ai/yi-large-turbo',
                   'anthropic/claude-instant-1.0',
                   'nothingiisreal/mn-celeste-12b',
                   'google/gemini-flash-1.5-exp',
                   '01-ai/yi-34b',
                   'qwen/qwen-14b-chat',
                   'google/gemini-2.0-flash-thinking-exp-1219:free',
                   'nousresearch/nous-hermes-yi-34b',
                   'migtissera/synthia-70b',
                   'openai/gpt-4o',
                   'anthropic/claude-3-sonnet',
                   'mistralai/pixtral-12b',
                   'deepseek/deepseek-chat-v2.5',
                   'mistralai/mistral-nemo',
                   'openai/gpt-4-32k',
                   'qwen/qwen-4b-chat',
                   'meta-llama/llama-3-70b',
                   'anthropic/claude-2.1:beta',
                   'liquid/lfm-40b',
                   'meta-llama/llama-3.3-70b-instruct',
                   'anthropic/claude-2.0',
                   'qwen/qwq-32b-preview',
                   'google/gemini-exp-1206:free',
                   'mattshumer/reflection-70b',
                   'sao10k/l3-stheno-8b',
                   'google/gemini-flash-1.5-8b',
                   'lizpreciatior/lzlv-70b-fp16-hf',
                   'neversleep/noromaid-mixtral-8x7b-instruct',
                   'mistralai/mixtral-8x7b-instruct:nitro',
                   'neversleep/llama-3-lumimaid-70b',
                   'togethercomputer/stripedhyena-hessian-7b',
                   'anthropic/claude-3.5-sonnet-20240620',
                   'nousresearch/nous-hermes-2-mistral-7b-dpo',
                   'nousresearch/hermes-2-theta-llama-3-8b',
                   'alpindale/magnum-72b',
                   'xwin-lm/xwin-lm-70b',
                   'mistralai/mistral-7b-instruct:nitro',
                   'anthropic/claude-3-opus',
                   'mistralai/ministral-3b',
                   '01-ai/yi-34b-chat',
                   'neversleep/llama-3-lumimaid-8b:extended',
                   'databricks/dbrx-instruct',
                   '01-ai/yi-1.5-34b-chat',
                   'teknium/openhermes-2.5-mistral-7b',
                   'liuhaotian/llava-yi-34b',
                   'meta-llama/llama-3-8b-instruct:free',
                   'mistralai/ministral-8b',
                   'openchat/openchat-7b',
                   'raifle/sorcererlm-8x22b',
                   'qwen/qwen-2-7b-instruct',
                   'jebcarter/psyfighter-13b',
                   'neversleep/llama-3-lumimaid-8b',
                   'openrouter/auto',
                   'google/gemini-flash-1.5-8b-exp',
                   'perplexity/llama-3.1-sonar-large-128k-online',
                   'openai/gpt-4-32k-0314',
                   'perplexity/llama-3-sonar-small-32k-online',
                   'ai21/jamba-1-5-mini',
                   'lynn/soliloquy-v3',
                   'nvidia/nemotron-4-340b-instruct',
                   'deepseek/deepseek-coder',
                   'sao10k/l3-lunaris-8b',
                   'meta-llama/llama-2-13b-chat',
                   'meta-llama/codellama-70b-instruct',
                   '01-ai/yi-34b-200k',
                   'nousresearch/nous-hermes-2-mixtral-8x7b-dpo',
                   'sao10k/l3-euryale-70b',
                   'meta-llama/llama-3.1-70b-instruct:free',
                   'deepseek/deepseek-r1:nitro',
                   'perplexity/llama-3.1-sonar-large-128k-chat',
                   'undi95/remm-slerp-l2-13b:extended',
                   'alpindale/goliath-120b',
                   'meta-llama/llama-3.1-70b-instruct',
                   'qwen/qwen-2.5-coder-32b-instruct',
                   'liquid/lfm-3b',
                   'deepseek/deepseek-r1:free',
                   'anthropic/claude-3.5-sonnet:beta',
                   'mistralai/pixtral-large-2411',
                   'mistralai/mixtral-8x7b',
                   'google/gemini-pro-1.5-exp',
                   'koboldai/psyfighter-13b-2',
                   'openai/gpt-4o-2024-11-20',
                   'inflatebot/mn-mag-mell-r1',
                   'microsoft/phi-3-medium-128k-instruct:free',
                   'nousresearch/hermes-3-llama-3.1-70b',
                   'huggingfaceh4/zephyr-7b-beta:free',
                   'mistralai/mistral-7b-instruct-v0.3',
                   'google/gemma-2-9b-it',
                   'perplexity/llama-3-sonar-small-32k-chat',
                   'microsoft/phi-3.5-mini-128k-instruct',
                   'meta-llama/llama-3.2-1b-instruct',
                   'mistralai/mistral-7b-instruct-v0.2',
                   'nousresearch/hermes-2-pro-llama-3-8b',
                   'openai/gpt-3.5-turbo-16k',
                   'perplexity/llama-3.1-sonar-small-128k-chat',
                   'microsoft/phi-3-medium-128k-instruct',
                   'amazon/nova-pro-v1',
                   'google/gemma-2-9b-it:free',
                   'pygmalionai/mythalion-13b',
                   'gryphe/mythomax-l2-13b:extended',
                   'gryphe/mythomax-l2-13b:nitro',
                   'neversleep/noromaid-20b',
                   'meta-llama/llama-3.2-3b-instruct',
                   'huggingfaceh4/zephyr-orpo-141b-a35b',
                   'deepseek/deepseek-r1-distill-llama-70b',
                   'inflection/inflection-3-pi',
                   'minimax/minimax-01',
                   'meta-llama/llama-guard-2-8b',
                   'meta-llama/llama-3.1-8b-instruct:free',
                   'anthropic/claude-3.5-sonnet',
                   'microsoft/wizardlm-2-7b',
                   'google/gemini-pro',
                   'deepseek/deepseek-r1',
                   'anthropic/claude-3.5-sonnet-20240620:beta',
                   'meta-llama/llama-3-8b-instruct',
                   'openai/gpt-4-1106-preview',
                   'fireworks/firellava-13b',
                   'sao10k/l3.1-70b-hanami-x1',
                   'google/palm-2-chat-bison',
                   'mancer/weaver',
                   'neversleep/llama-3.1-lumimaid-8b',
                   'cognitivecomputations/dolphin-llama-3-70b',
                   'google/gemini-2.0-flash-thinking-exp:free',
                   '01-ai/yi-large',
                   'cohere/command',
                   'google/gemini-flash-1.5',
                   'amazon/nova-micro-v1',
                   'x-ai/grok-2-1212',
                   'qwen/qwen-110b-chat',
                   'meta-llama/llama-3-70b-instruct',
                   'openai/gpt-3.5-turbo-0301',
                   'mistralai/mistral-medium',
                   'meta-llama/llama-3.2-11b-vision-instruct',
                   'google/gemini-2.0-flash-exp:free',
                   'ai21/jamba-1-5-large',
                   'google/gemini-exp-1114:free',
                   'deepseek/deepseek-r1-distill-qwen-32b',
                   'cohere/command-r-plus-04-2024',
                   'infermatic/mn-inferor-12b',
                   'openai/o1-preview',
                   'openai/gpt-4-turbo-preview',
                   'x-ai/grok-2-vision-1212',
                   'anthracite-org/magnum-v4-72b',
                   'openai/gpt-4o:extended',
                   'x-ai/grok-2-mini',
                   'undi95/toppy-m-7b:nitro',
                   'anthracite-org/magnum-v2-72b',
                   'meta-llama/llama-3.2-1b-instruct:free',
                   'openai/gpt-3.5-turbo-1106',
                   'aetherwiing/mn-starcannon-12b',
                   'openai/gpt-4-turbo',
                   'google/learnlm-1.5-pro-experimental:free',
                   'inflection/inflection-3-productivity',
                   'cohere/command-r-plus',
                   'meta-llama/llama-3.1-405b-instruct:free',
                   'openchat/openchat-8b',
                   'nousresearch/nous-hermes-2-mixtral-8x7b-sft',
                   'openai/gpt-3.5-turbo-0613',
                   'nousresearch/nous-capybara-34b',
                   'qwen/qwen-2-72b-instruct',
                   'eva-unit-01/eva-qwen-2.5-32b',
                   'undi95/toppy-m-7b:free',
                   'google/gemini-exp-1121:free',
                   'cohere/command-r7b-12-2024',
                   '01-ai/yi-vision',
                   'meta-llama/llama-3.1-70b-instruct:nitro',
                   'liuhaotian/llava-13b',
                   'openrouter/cinematika-7b',
                   'openai/o1',
                   'rwkv/rwkv-5-world-3b',
                   'perplexity/sonar-reasoning',
                   'deepseek/deepseek-chat',
                   'mistralai/mistral-7b-instruct:free',
                   'anthropic/claude-instant-1.1',
                   'qwen/qwen-32b-chat',
                   'lynn/soliloquy-l3',
                   'mistralai/mixtral-8x7b-instruct',
                   'anthropic/claude-3-haiku:beta',
                   'mistralai/mistral-large-2407',
                   'intel/neural-chat-7b',
                   'anthropic/claude-3-haiku',
                   'microsoft/phi-3-mini-128k-instruct:free',
                   'google/gemma-2-27b-it',
                   'jondurbin/airoboros-l2-70b',
                   'meta-llama/llama-2-70b-chat',
                   'gryphe/mythomax-l2-13b',
                   'google/palm-2-codechat-bison',
                   'qwen/qwen-7b-chat',
                   'recursal/rwkv-5-3b-ai-town',
                   'google/gemma-7b-it',
                   'cohere/command-r-03-2024',
                   'openai/o1-mini-2024-09-12',
                   'anthropic/claude-2',
                   'ai21/jamba-instruct',
                   'bigcode/starcoder2-15b-instruct',
                   'perplexity/llama-3-sonar-large-32k-online',
                   'togethercomputer/stripedhyena-nous-7b',
                   'sophosympatheia/midnight-rose-70b',
                   'qwen/qwen-2.5-7b-instruct',
                   'openchat/openchat-7b:free',
                   'phind/phind-codellama-34b',
                   'mistralai/codestral-mamba',
                   'openai/chatgpt-4o-latest',
                   'openai/o1-mini',
                   'eva-unit-01/eva-qwen-2.5-14b',
                   'recursal/eagle-7b',
                   'qwen/qwen-2-7b-instruct:free'],
    "gemini": ["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-latest", "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-1219"],
    "ollama": ["qwen2.5:7b", "llama2:7b", "llama3.1:8b", "llama3.1:13b", "mistral:7b"],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
}


class ModelProviderNamesEnum(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    KLUSTER_AI = "kluster_ai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


class AnthropicModelNamesEnum(Enum):
    CLAUDE_3_5_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"


class OpenaiModelNamesEnum(Enum):
    GPT_4O = "gpt-4o"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class DeepseekModelNamesEnum(Enum):
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"


class OpenRouterModelNamesEnum(Enum):
    MISTRAL_LARGE_2411 = 'mistralai/mistral-large-2411'
    META_LLAMA_3_2_90B_VISION_INSTRUCT = 'meta-llama/llama-3.2-90b-vision-instruct:free'
    CLAUDE_3_5_HAIKU_20241022_BETA = 'anthropic/claude-3.5-haiku-20241022:beta'
    CLAUDE_2_1 = 'anthropic/claude-2.1'
    CLAUDE_2_0_BETA = 'anthropic/claude-2.0:beta'
    PERPLEXITY_LLAMA_3_SONAR_LARGE_32K_CHAT = 'perplexity/llama-3-sonar-large-32k-chat'
    SAO10K_L3_1_EURYALE_70B = 'sao10k/l3.1-euryale-70b'
    META_LLAMA_3_8B = 'meta-llama/llama-3-8b'
    MICROSOFT_PHI_3_MEDIUM_4K_INSTRUCT = 'microsoft/phi-3-medium-4k-instruct'
    DEEPSEEK_R1_DISTILL_QWEN_14B = 'deepseek/deepseek-r1-distill-qwen-14b'
    JONDURBIN_BAGEL_34B = 'jondurbin/bagel-34b'
    MISTRAL_TINY = 'mistralai/mistral-tiny'
    PERPLEXITY_LLAMA_3_1_SONAR_SMALL_128K_ONLINE = 'perplexity/llama-3.1-sonar-small-128k-online'
    OPENAI_GPT_4_VISION_PREVIEW = 'openai/gpt-4-vision-preview'
    OPENAI_GPT_3_5_TURBO = 'openai/gpt-3.5-turbo'
    META_LLAMA_3_1_405B = 'meta-llama/llama-3.1-405b'
    ALLENAI_OLMO_7B_INSTRUCT = 'allenai/olmo-7b-instruct'
    META_LLAMA_3_2_11B_VISION_INSTRUCT = 'meta-llama/llama-3.2-11b-vision-instruct:free'
    CLAUDE_3_5_HAIKU_20241022 = 'anthropic/claude-3.5-haiku-20241022'
    PERPLEXITY_SONAR = 'perplexity/sonar'
    OPENAI_GPT_4_0314 = 'openai/gpt-4-0314'
    MISTRAL_LARGE = 'mistralai/mistral-large'
    COHERE_COMMAND_R_PLUS_08_2024 = 'cohere/command-r-plus-08-2024'
    META_LLAMA_3_1_405B_INSTRUCT_NITRO = 'meta-llama/llama-3.1-405b-instruct:nitro'
    EVA_UNIT_01_EVA_QWEN_2_5_72B = 'eva-unit-01/eva-qwen-2.5-72b'
    QWEN_72B_CHAT = 'qwen/qwen-72b-chat'
    SOPHOSYMPATHAEIA_ROGUE_ROSE_103B_V0_2 = 'sophosympatheia/rogue-rose-103b-v0.2:free'
    OPENAI_GPT_4O_2024_05_13 = 'openai/gpt-4o-2024-05-13'
    GOOGLE_PALM_2_CHAT_BISON_32K = 'google/palm-2-chat-bison-32k'
    ANTHROPIC_CLAUDE_1 = 'anthropic/claude-1'
    OPENAI_GPT_4O_MINI_2024_07_18 = 'openai/gpt-4o-mini-2024-07-18'
    OPEN_ORCA_MISTRAL_7B_OPENORCA = 'open-orca/mistral-7b-openorca'
    _01_AI_YI_LARGE_FC = '01-ai/yi-large-fc'
    META_LLAMA_CODELLAMA_34B_INSTRUCT = 'meta-llama/codellama-34b-instruct'
    META_LLAMA_LLAMA_3_1_8B_INSTRUCT = 'meta-llama/llama-3.1-8b-instruct'
    QWEN_2_5_72B_INSTRUCT = 'qwen/qwen-2.5-72b-instruct'
    OPENAI_GPT_4 = 'openai/gpt-4'
    MISTRAL_SMALL = 'mistralai/mistral-small'
    NOUSRESEARCH_NOUS_HERMES_LLAMA2_70B = 'nousresearch/nous-hermes-llama2-70b'
    GOOGLE_GEMINI_PRO_VISION = 'google/gemini-pro-vision'
    MISTRAL_CODERESTRAL_2501 = 'mistralai/codestral-2501'
    _01_AI_YI_6B = '01-ai/yi-6b'
    PERPLEXITY_LLAMA_3_1_SONAR_HUGE_128K_ONLINE = 'perplexity/llama-3.1-sonar-huge-128k-online'
    OPENAI_GPT_4O_MINI = 'openai/gpt-4o-mini'
    MICROSOFT_WIZARDLM_2_8X22B = 'microsoft/wizardlm-2-8x22b'
    MISTRAL_MIXTRAL_8X22B = 'mistralai/mixtral-8x22b'
    QWEN_2_VL_72B_INSTRUCT = 'qwen/qwen-2-vl-72b-instruct'
    META_LLAMA_LLAMA_3_2_3B_INSTRUCT_FREE = 'meta-llama/llama-3.2-3b-instruct:free'
    X_AI_GROK_BETA = 'x-ai/grok-beta'
    NOUSRESEARCH_HERMES_3_LLAMA_3_1_405B = 'nousresearch/hermes-3-llama-3.1-405b'
    MICROSOFT_PHI_3_MINI_128K_INSTRUCT = 'microsoft/phi-3-mini-128k-instruct'
    META_LLAMA_LLAMA_3_1_405B_INSTRUCT = 'meta-llama/llama-3.1-405b-instruct'
    OPENAI_SHAPE = 'openai/shap-e'
    COHERE_COMMAND_R = 'cohere/command-r'
    MISTRAL_7B_INSTRUCT_V0_1 = 'mistralai/mistral-7b-instruct-v0.1'
    QWEN_2_VL_7B_INSTRUCT = 'qwen/qwen-2-vl-7b-instruct'
    NVIDIA_LLAMA_3_1_NEMOTRON_70B_INSTRUCT = 'nvidia/llama-3.1-nemotron-70b-instruct'
    GOOGLE_PALM_2_CODECHAT_BISON_32K = 'google/palm-2-codechat-bison-32k'
    ANTHROPIC_CLAUDE_INSTANT_1 = 'anthropic/claude-instant-1'
    SAO10K_FIMBULVETR_11B_V2 = 'sao10k/fimbulvetr-11b-v2'
    SAO10K_L3_3_EURYALE_70B = 'sao10k/l3.3-euryale-70b'
    GRYPHE_MYTHOMIST_7B = 'gryphe/mythomist-7b'
    OPENAI_GPT_3_5_TURBO_0125 = 'openai/gpt-3.5-turbo-0125'
    GOOGLE_GEMINI_PRO_1_5 = 'google/gemini-pro-1.5'
    ANTHROPIC_CLAUDE_3_5_HAIKU = 'anthropic/claude-3.5-haiku'
    MICROSOFT_PHI_4 = 'microsoft/phi-4'
    META_LLAMA_LLAMA_3_8B_INSTRUCT_EXTENDED = 'meta-llama/llama-3-8b-instruct:extended'
    QWEN_QVQ_72B_PREVIEW = 'qwen/qvq-72b-preview'
    ANTHROPIC_CLAUDE_3_SONNET_BETA = 'anthropic/claude-3-sonnet:beta'
    ANTHROPIC_CLAUDE_3_5_HAIKU_BETA = 'anthropic/claude-3.5-haiku:beta'
    MISTRAL_MIXTRAL_8X22B_INSTRUCT = 'mistralai/mixtral-8x22b-instruct'
    COGNITIVECOMPUTATIONS_DOLPHIN_MIXTRAL_8X22B = 'cognitivecomputations/dolphin-mixtral-8x22b'
    OPENAI_GPT_4O_2024_08_06 = 'openai/gpt-4o-2024-08-06'
    THEDRUMMER_ROCINANTE_12B = 'thedrummer/rocinante-12b'
    COHERE_COMMAND_R_08_2024 = 'cohere/command-r-08-2024'
    X_AI_GROK_2 = 'x-ai/grok-2'
    OPENAI_O1_PREVIEW_2024_09_12 = 'openai/o1-preview-2024-09-12'
    GRYPHE_MYTHOMAX_L2_13B_FREE = 'gryphe/mythomax-l2-13b:free'
    MISTRAL_7B_INSTRUCT = 'mistralai/mistral-7b-instruct'
    EVA_UNIT_01_EVA_LLAMA_3_33_70B = 'eva-unit-01/eva-llama-3.33-70b'
    ANTHROPIC_CLAUDE_1_2 = 'anthropic/claude-1.2'
    META_LLAMA_LLAMA_3_8B_INSTRUCT_NITRO = 'meta-llama/llama-3-8b-instruct:nitro'
    ANTHROPIC_CLAUDE_2_BETA = 'anthropic/claude-2:beta'
    AUSTISM_CHRONOS_HERMES_13B = 'austism/chronos-hermes-13b'
    AMAZON_NOVA_LITE_V1 = 'amazon/nova-lite-v1'
    UNDI95_TOPPY_M_7B = 'undi95/toppy-m-7b'
    X_AI_GROK_VISION_BETA = 'x-ai/grok-vision-beta'
    NEVERSLEEP_LLAMA_3_1_LUMIMAID_70B = 'neversleep/llama-3.1-lumimaid-70b'
    ANTHROPIC_CLAUDE_3_OPUS_BETA = 'anthropic/claude-3-opus:beta'
    NOUSRESEARCH_NOUS_HERMES_LLAMA2_13B = 'nousresearch/nous-hermes-llama2-13b'
    OPENAI_GPT_3_5_TURBO_INSTRUCT = 'openai/gpt-3.5-turbo-instruct'
    TEKNIUM_OPENHERMES_2_MISTRAL_7B = 'teknium/openhermes-2-mistral-7b'
    UNDI95_REMM_SLERP_L2_13B = 'undi95/remm-slerp-l2-13b'
    LIQUID_LFM_7B = 'liquid/lfm-7b'
    THEDRUMMER_UNSLOPNEMO_12B = 'thedrummer/unslopnemo-12b'
    META_LLAMA_LLAMA_3_2_90B_VISION_INSTRUCT = 'meta-llama/llama-3.2-90b-vision-instruct'
    NOUSRESEARCH_NOUS_HERMES_2_VISION_7B = 'nousresearch/nous-hermes-2-vision-7b'
    NOUSRESEARCH_NOUS_CAPYBARA_7B = 'nousresearch/nous-capybara-7b'
    COGNITIVECOMPUTATIONS_DOLPHIN_MIXTAL_8X7B = 'cognitivecomputations/dolphin-mixtral-8x7b'
    META_LLAMA_LLAMA_3_70B_INSTRUCT_NITRO = 'meta-llama/llama-3-70b-instruct:nitro'
    SNOWFLAKE_SNOWFLAKE_ARCTIC_INSTRUCT = 'snowflake/snowflake-arctic-instruct'
    _01_AI_YI_LARGE_TURBO = '01-ai/yi-large-turbo'
    ANTHROPIC_CLAUDE_INSTANT_1_0 = 'anthropic/claude-instant-1.0'
    NOTHINGIISREAL_MN_CELESTE_12B = 'nothingiisreal/mn-celeste-12b'
    GOOGLE_GEMINI_FLASH_1_5_EXP = 'google/gemini-flash-1.5-exp'
    _01_AI_YI_34B = '01-ai/yi-34b'
    QWEN_QWEN_14B_CHAT = 'qwen/qwen-14b-chat'
    GOOGLE_GEMINI_2_0_FLASH_THINKING_EXP_1219_FREE = 'google/gemini-2.0-flash-thinking-exp-1219:free'
    NOUSRESEARCH_NOUS_HERMES_YI_34B = 'nousresearch/nous-hermes-yi-34b'
    MIGTISSERA_SYNTHIA_70B = 'migtissera/synthia-70b'
    OPENAI_GPT_4O = 'openai/gpt-4o'
    ANTHROPIC_CLAUDE_3_SONNET = 'anthropic/claude-3-sonnet'
    MISTRALAI_PIXTAL_12B = 'mistralai/pixtral-12b'
    DEEPSEEK_DEEPSEEK_CHAT_V2_5 = 'deepseek/deepseek-chat-v2.5'
    MISTRALAI_MISTRAL_NEMO = 'mistralai/mistral-nemo'
    OPENAI_GPT_4_32K = 'openai/gpt-4-32k'
    QWEN_QWEN_4B_CHAT = 'qwen/qwen-4b-chat'
    META_LLAMA_LLAMA_3_70B = 'meta-llama/llama-3-70b'
    ANTHROPIC_CLAUDE_2_1_BETA = 'anthropic/claude-2.1:beta'
    LIQUID_LFM_40B = 'liquid/lfm-40b'
    META_LLAMA_LLAMA_3_3_70B_INSTRUCT = 'meta-llama/llama-3.3-70b-instruct'
    ANTHROPIC_CLAUDE_2_0 = 'anthropic/claude-2.0'
    QWEN_QWQ_32B_PREVIEW = 'qwen/qwq-32b-preview'
    GOOGLE_GEMINI_EXP_1206_FREE = 'google/gemini-exp-1206:free'
    MATT_SHUMER_REFLECTION_70B = 'mattshumer/reflection-70b'
    SAO10K_L3_STHENO_8B = 'sao10k/l3-stheno-8b'
    GOOGLE_GEMINI_FLASH_1_5_8B = 'google/gemini-flash-1.5-8b'
    LIZ_PRECIATIOR_LZLV_70B_FP16_HF = 'lizpreciatior/lzlv-70b-fp16-hf'
    NEVERSLEEP_NOROMAID_MIXTAL_8X7B_INSTRUCT = 'neversleep/noromaid-mixtral-8x7b-instruct'
    MISTRALAI_MIXTAL_8X7B_INSTRUCT_NITRO = 'mistralai/mixtral-8x7b-instruct:nitro'
    NEVERSLEEP_LLAMA_3_LUMIMAID_70B = 'neversleep/llama-3-lumimaid-70b'
    TOGETHER_COMPUTER_STRIPEDHYENA_HESSIAN_7B = 'togethercomputer/stripedhyena-hessian-7b'
    ANTHROPIC_CLAUDE_3_5_SONNET_20240620 = 'anthropic/claude-3.5-sonnet-20240620'
    NOUSRESEARCH_NOUS_HERMES_2_MISTRAL_7B_DPO = 'nousresearch/nous-hermes-2-mistral-7b-dpo'
    NOUSRESEARCH_HERMES_2_THETA_LLAMA_3_8B = 'nousresearch/hermes-2-theta-llama-3-8b'
    ALPINDALE_MAGNUM_72B = 'alpindale/magnum-72b'
    XWIN_LM_XWIN_LM_70B = 'xwin-lm/xwin-lm-70b'
    MISTRALAI_MISTRAL_7B_INSTRUCT_NITRO = 'mistralai/mistral-7b-instruct:nitro'
    ANTHROPIC_CLAUDE_3_OPUS = 'anthropic/claude-3-opus'
    MISTRALAI_MINISTRAL_3B = 'mistralai/ministral-3b'
    _01_AI_YI_34B_CHAT = '01-ai/yi-34b-chat'
    NEVERSLEEP_LLAMA_3_LUMIMAID_8B_EXTENDED = 'neversleep/llama-3-lumimaid-8b:extended'
    DATABRICKS_DBRX_INSTRUCT = 'databricks/dbrx-instruct'
    _01_AI_YI_1_5_34B_CHAT = '01-ai/yi-1.5-34b-chat'
    TEKNIUM_OPENHERMES_2_5_MISTRAL_7B = 'teknium/openhermes-2.5-mistral-7b'
    LIUHAOTIAN_LLAVA_YI_34B = 'liuhaotian/llava-yi-34b'
    META_LLAMA_LLAMA_3_8B_INSTRUCT_FREE = 'meta-llama/llama-3-8b-instruct:free'
    MISTRALAI_MINISTRAL_8B = 'mistralai/ministral-8b'
    OPENCHAT_OPENCHAT_7B = 'openchat/openchat-7b'
    RAIFLE_SORCERERLM_8X22B = 'raifle/sorcererlm-8x22b'
    QWEN_QWEN_2_7B_INSTRUCT = 'qwen/qwen-2-7b-instruct'
    JEBCARTER_PSYFIGHTER_13B = 'jebcarter/psyfighter-13b'
    NEVERSLEEP_LLAMA_3_LUMIMAID_8B = 'neversleep/llama-3-lumimaid-8b'
    OPENROUTER_AUTO = 'openrouter/auto'
    GOOGLE_GEMINI_FLASH_1_5_8B_EXP = 'google/gemini-flash-1.5-8b-exp'
    PERPLEXITY_LLAMA_3_1_SONAR_LARGE_128K_ONLINE = 'perplexity/llama-3.1-sonar-large-128k-online'
    OPENAI_GPT_4_32K_0314 = 'openai/gpt-4-32k-0314'
    PERPLEXITY_LLAMA_3_SONAR_SMALL_32K_ONLINE = 'perplexity/llama-3-sonar-small-32k-online'
    AI21_JAMBA_1_5_MINI = 'ai21/jamba-1-5-mini'
    LYNN_SOLILOQUY_V3 = 'lynn/soliloquy-v3'
    NVIDIA_NEMOTRON_4_340B_INSTRUCT = 'nvidia/nemotron-4-340b-instruct'
    DEEPSEEK_DEEPSEEK_CODER = 'deepseek/deepseek-coder'
    SAO10K_L3_LUNARIS_8B = 'sao10k/l3-lunaris-8b'
    META_LLAMA_LLAMA_2_13B_CHAT = 'meta-llama/llama-2-13b-chat'
    META_LLAMA_CODELLAMA_70B_INSTRUCT = 'meta-llama/codellama-70b-instruct'
    _01_AI_YI_34B_200K = '01-ai/yi-34b-200k'
    NOUSRESEARCH_NOUS_HERMES_2_MIXTAL_8X7B_DPO = 'nousresearch/nous-hermes-2-mixtral-8x7b-dpo'
    SAO10K_L3_EURYALE_70B = 'sao10k/l3-euryale-70b'
    META_LLAMA_LLAMA_3_1_70B_INSTRUCT_FREE = 'meta-llama/llama-3.1-70b-instruct:free'
    DEEPSEEK_DEEPSEEK_R1_NITRO = 'deepseek/deepseek-r1:nitro'
    PERPLEXITY_LLAMA_3_1_SONAR_LARGE_128K_CHAT = 'perplexity/llama-3.1-sonar-large-128k-chat'
    UNDI95_REMM_SLERP_L2_13B_EXTENDED = 'undi95/remm-slerp-l2-13b:extended'
    ALPINDALE_GOLIATH_120B = 'alpindale/goliath-120b'
    META_LLAMA_LLAMA_3_1_70B_INSTRUCT = 'meta-llama/llama-3.1-70b-instruct'
    QWEN_QWEN_2_5_CODER_32B_INSTRUCT = 'qwen/qwen-2.5-coder-32b-instruct'
    LIQUID_LFM_3B = 'liquid/lfm-3b'
    DEEPSEEK_DEEPSEEK_R1_FREE = 'deepseek/deepseek-r1:free'
    ANTHROPIC_CLAUDE_3_5_SONNET_BETA = 'anthropic/claude-3.5-sonnet:beta'
    MISTRALAI_PIXTAL_LARGE_2411 = 'mistralai/pixtral-large-2411'
    MISTRALAI_MIXTAL_8X7B = 'mistralai/mixtral-8x7b'
    GOOGLE_GEMINI_PRO_1_5_EXP = 'google/gemini-pro-1.5-exp'
    KOBAI_PSYFIGHTER_13B_2 = 'koboldai/psyfighter-13b-2'
    OPENAI_GPT_4O_2024_11_20 = 'openai/gpt-4o-2024-11-20'
    INFLATEBOT_MN_MAG_MELL_R1 = 'inflatebot/mn-mag-mell-r1'
    MICROSOFT_PHI_3_MEDIUM_128K_INSTRUCT_FREE = 'microsoft/phi-3-medium-128k-instruct:free'
    NOUSRESEARCH_HERMES_3_LLAMA_3_1_70B = 'nousresearch/hermes-3-llama-3.1-70b'
    HUGGINGFACEH4_ZEPHYR_7B_BETA_FREE = 'huggingfaceh4/zephyr-7b-beta:free'
    MISTRALAI_MISTRAL_7B_INSTRUCT_V0_3 = 'mistralai/mistral-7b-instruct-v0.3'
    GOOGLE_GEMMA_2_9B_IT = 'google/gemma-2-9b-it'
    PERPLEXITY_LLAMA_3_SONAR_SMALL_32K_CHAT = 'perplexity/llama-3-sonar-small-32k-chat'
    MICROSOFT_PHI_3_5_MINI_128K_INSTRUCT = 'microsoft/phi-3.5-mini-128k-instruct'
    META_LLAMA_LLAMA_3_2_1B_INSTRUCT = 'meta-llama/llama-3.2-1b-instruct'
    MISTRALAI_MISTRAL_7B_INSTRUCT_V0_2 = 'mistralai/mistral-7b-instruct-v0.2'
    NOUSRESEARCH_HERMES_2_PRO_LLAMA_3_8B = 'nousresearch/hermes-2-pro-llama-3-8b'
    OPENAI_GPT_3_5_TURBO_16K = 'openai/gpt-3.5-turbo-16k'
    PERPLEXITY_LLAMA_3_1_SONAR_SMALL_128K_CHAT = 'perplexity/llama-3.1-sonar-small-128k-chat'
    MICROSOFT_PHI_3_MEDIUM_128K_INSTRUCT = 'microsoft/phi-3-medium-128k-instruct'
    AMAZON_NOVA_PRO_V1 = 'amazon/nova-pro-v1'
    GOOGLE_GEMMA_2_9B_IT_FREE = 'google/gemma-2-9b-it:free'
    PYGMALIONAI_MYTHALION_13B = 'pygmalionai/mythalion-13b'
    GRYPHE_MYTHOMAX_L2_13B_EXTENDED = 'gryphe/mythomax-l2-13b:extended'
    GRYPHE_MYTHOMAX_L2_13B_NITRO = 'gryphe/mythomax-l2-13b:nitro'
    NEVERSLEEP_NOROMAID_20B = 'neversleep/noromaid-20b'
    META_LLAMA_LLAMA_3_2_3B_INSTRUCT = 'meta-llama/llama-3.2-3b-instruct'
    HUGGINGFACEH4_ZEPHYR_ORPO_141B_A35B = 'huggingfaceh4/zephyr-orpo-141b-a35b'
    DEEPSEEK_DEEPSEEK_R1_DISTILL_LLAMA_70B = 'deepseek/deepseek-r1-distill-llama-70b'
    INFLECTION_INFLECTION_3_PI = 'inflection/inflection-3-pi'
    MINIMAX_MINIMAX_01 = 'minimax/minimax-01'
    META_LLAMA_LLAMA_GUARD_2_8B = 'meta-llama/llama-guard-2-8b'
    META_LLAMA_LLAMA_3_1_8B_INSTRUCT_FREE = 'meta-llama/llama-3.1-8b-instruct:free'
    ANTHROPIC_CLAUDE_3_5_SONNET = 'anthropic/claude-3.5-sonnet'
    MICROSOFT_WIZARDLM_2_7B = 'microsoft/wizardlm-2-7b'
    GOOGLE_GEMINI_PRO = 'google/gemini-pro'
    DEEPSEEK_DEEPSEEK_R1 = 'deepseek/deepseek-r1'
    ANTHROPIC_CLAUDE_3_5_SONNET_20240620_BETA = 'anthropic/claude-3.5-sonnet-20240620:beta'
    META_LLAMA_LLAMA_3_8B_INSTRUCT = 'meta-llama/llama-3-8b-instruct'
    OPENAI_GPT_4_1106_PREVIEW = 'openai/gpt-4-1106-preview'
    FIREWORKS_FIRELLAVA_13B = 'fireworks/firellava-13b'
    SAO10K_L3_1_70B_HANAMI_X1 = 'sao10k/l3.1-70b-hanami-x1'
    GOOGLE_PALM_2_CHAT_BISON = 'google/palm-2-chat-bison'
    MANCER_WEAVER = 'mancer/weaver'
    NEVERSLEEP_LLAMA_3_1_LUMIMAID_8B = 'neversleep/llama-3.1-lumimaid-8b'
    COGNITIVECOMPUTATIONS_DOLPHIN_LLAMA_3_70B = 'cognitivecomputations/dolphin-llama-3-70b'
    GOOGLE_GEMINI_2_0_FLASH_THINKING_EXP_FREE = 'google/gemini-2.0-flash-thinking-exp:free'
    _01_AI_YI_LARGE = '01-ai/yi-large'
    COHERE_COMMAND = 'cohere/command'
    GOOGLE_GEMINI_FLASH_1_5 = 'google/gemini-flash-1.5'
    AMAZON_NOVA_MICRO_V1 = 'amazon/nova-micro-v1'
    X_AI_GROK_2_1212 = 'x-ai/grok-2-1212'
    QWEN_QWEN_110B_CHAT = 'qwen/qwen-110b-chat'
    META_LLAMA_LLAMA_3_70B_INSTRUCT = 'meta-llama/llama-3-70b-instruct'
    OPENAI_GPT_3_5_TURBO_0301 = 'openai/gpt-3.5-turbo-0301'
    MISTRALAI_MISTRAL_MEDIUM = 'mistralai/mistral-medium'
    META_LLAMA_LLAMA_3_2_11B_VISION_INSTRUCT = 'meta-llama/llama-3.2-11b-vision-instruct'
    GOOGLE_GEMINI_2_0_FLASH_EXP_FREE = 'google/gemini-2.0-flash-exp:free'
    AI21_JAMBA_1_5_LARGE = 'ai21/jamba-1-5-large'
    GOOGLE_GEMINI_EXP_1114_FREE = 'google/gemini-exp-1114:free'
    DEEPSEEK_DEEPSEEK_R1_DISTILL_QWEN_32B = 'deepseek/deepseek-r1-distill-qwen-32b'
    COHERE_COMMAND_R_PLUS_04_2024 = 'cohere/command-r-plus-04-2024'
    INFERMATIC_MN_INFEROR_12B = 'infermatic/mn-inferor-12b'
    OPENAI_O1_PREVIEW = 'openai/o1-preview'
    OPENAI_GPT_4_TURBO_PREVIEW = 'openai/gpt-4-turbo-preview'
    # 'x-ai/grok-2-vision-1212'
    # 'anthracite-org/magnum-v4-72b'
    # 'openai/gpt-4o:extended'
    # 'x-ai/grok-2-mini'
    # 'undi95/toppy-m-7b:nitro'
    # 'anthracite-org/magnum-v2-72b'
    # 'meta-llama/llama-3.2-1b-instruct:free'
    # 'openai/gpt-3.5-turbo-1106'
    # 'aetherwiing/mn-starcannon-12b'
    # 'openai/gpt-4-turbo'
    # 'google/learnlm-1.5-pro-experimental:free'
    # 'inflection/inflection-3-productivity'
    # 'cohere/command-r-plus'
    # 'meta-llama/llama-3.1-405b-instruct:free'
    # 'openchat/openchat-8b'
    # 'nousresearch/nous-hermes-2-mixtral-8x7b-sft'
    # 'openai/gpt-3.5-turbo-0613'
    # 'nousresearch/nous-capybara-34b'
    # 'qwen/qwen-2-72b-instruct'
    # 'eva-unit-01/eva-qwen-2.5-32b'
    # 'undi95/toppy-m-7b:free'
    # 'google/gemini-exp-1121:free'
    # 'cohere/command-r7b-12-2024'
    # '01-ai/yi-vision'
    # 'meta-llama/llama-3.1-70b-instruct:nitro'
    # 'liuhaotian/llava-13b'
    # 'openrouter/cinematika-7b'
    # 'openai/o1'
    # 'rwkv/rwkv-5-world-3b'
    # 'perplexity/sonar-reasoning'
    # 'deepseek/deepseek-chat'
    # 'mistralai/mistral-7b-instruct:free'
    # 'anthropic/claude-instant-1.1'
    # 'qwen/qwen-32b-chat'
    # 'lynn/soliloquy-l3'
    # 'mistralai/mixtral-8x7b-instruct'
    # 'anthropic/claude-3-haiku:beta'
    # 'mistralai/mistral-large-2407'
    # 'intel/neural-chat-7b'
    # 'anthropic/claude-3-haiku'
    # 'microsoft/phi-3-mini-128k-instruct:free'
    # 'google/gemma-2-27b-it'
    # 'jondurbin/airoboros-l2-70b'
    # 'meta-llama/llama-2-70b-chat'
    # 'gryphe/mythomax-l2-13b'
    # 'google/palm-2-codechat-bison'
    # 'qwen/qwen-7b-chat'
    # 'recursal/rwkv-5-3b-ai-town'
    # 'google/gemma-7b-it'
    # 'cohere/command-r-03-2024'
    # 'openai/o1-mini-2024-09-12'
    # 'anthropic/claude-2'
    # 'ai21/jamba-instruct'
    # 'bigcode/starcoder2-15b-instruct'
    # 'perplexity/llama-3-sonar-large-32k-online'
    # 'togethercomputer/stripedhyena-nous-7b'
    # 'sophosympatheia/midnight-rose-70b'
    # 'qwen/qwen-2.5-7b-instruct'
    # 'openchat/openchat-7b:free'
    # 'phind/phind-codellama-34b'
    # 'mistralai/codestral-mamba'
    OPENAI_CHATGPT_4O_LATEST = 'openai/chatgpt-4o-latest'
    OPENAI_O1_MINI = 'openai/o1-mini'
    EVA_UNIT_01_EVA_QWEN_2_5_14B = 'eva-unit-01/eva-qwen-2.5-14b'
    RECURSAL_EAGLE_7B = 'recursal/eagle-7b'
    QWEN_QWEN_2_7B_INSTRUCT_FREE = 'qwen/qwen-2-7b-instruct:free'


class GeminiModelNamesEnum(Enum):
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_2_0_FLASH_THINKING_EXP = "gemini-2.0-flash-thinking-exp"
    GEMINI_1_5_FLASH_LATEST = "gemini-1.5-flash-latest"
    GEMINI_1_5_FLASH_8B_LATEST = "gemini-1.5-flash-8b-latest"
    GEMINI_2_0_FLASH_THINKING_EXP_1219 = "gemini-2.0-flash-thinking-exp-1219"


class OllamaModelNamesEnum(Enum):
    QWEN2_5_7B = "qwen2.5:7b"
    LLAMA2_7B = "llama2:7b"
    LLAMA3_1_8B = "llama3.1:8b"
    LLAMA3_1_13B = "llama3.1:13b"
    MISTRAL_7B = "mistral:7b"


class AzureOpenaiModelNamesEnum(Enum):
    GPT_4O = "gpt-4o"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class KlusterAIModelNamesEnum(Enum):
    DEEPSEEK_REASONER = "deepseek-ai/DeepSeek-R1"
    META_LLAMA_3_1_405B_INSTUCT_TURBO = "klusterai/Meta-Llama-3.1-405B-Instruct-Turbo"
    META_LLAMA_3_3_70B_INSTUCT_TURBO = "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo"


class ModelGetter:
    anthropic = AnthropicModelNamesEnum
    openai = OpenaiModelNamesEnum
    azure_openai = AzureOpenaiModelNamesEnum
    deepseek = DeepseekModelNamesEnum
    kluster_ai = KlusterAIModelNamesEnum
    openrouter = OpenRouterModelNamesEnum
    gemini = GeminiModelNamesEnum
    ollama = OllamaModelNamesEnum


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_STUDIO_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_GEMINI_STUDIO_API_KEY is not set in the environment variables")

KLUSTER_AI_API_KEY = os.getenv("KLUSTER_AI_API_KEY", "")
if not KLUSTER_AI_API_KEY:
    raise ValueError(
        "KLUSTER_AI_API_KEY is not set in the environment variables")

OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
if not OPEN_ROUTER_API_KEY:
    raise ValueError(
        "OPEN_ROUTER_API_KEY is not set in the environment variables")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "DEEPSEEK_API_KEY is not set in the environment variables")


class DataPolicy(TypedDict):
    privacyPolicyURL: str
    training: bool


class Icon(TypedDict):
    url: str


class ProviderInfo(TypedDict):
    name: str
    displayName: str
    baseUrl: str
    dataPolicy: DataPolicy
    hasChatCompletions: bool
    hasCompletions: bool
    isAbortable: bool
    moderationRequired: bool
    group: str
    editors: List[str]
    owners: List[str]
    isMultipartSupported: bool
    statusPageUrl: Optional[str]
    byokEnabled: bool
    isPrimaryProvider: bool
    icon: Icon


class Model(TypedDict):
    slug: str
    hf_slug: str
    updated_at: str
    created_at: str
    hf_updated_at: Optional[str]
    name: str
    short_name: str
    author: str
    description: str
    model_version_group_id: Optional[str]
    context_length: int
    modality: str
    has_text_output: bool
    group: str
    instruct_type: Optional[str]
    default_system: Optional[str]
    default_stops: List[str]
    hidden: bool
    router: Optional[str]
    warning_message: Optional[str]
    permaslug: str


class Endpoint(TypedDict):
    id: str
    name: str
    context_length: int
    model: Model
    model_variant_slug: str
    model_variant_permaslug: str
    provider_name: str
    provider_info: ProviderInfo
    provider_display_name: str
    provider_model_id: str
    is_cloaked: bool
    quantization: Optional[str]
    variant: str
    is_self_hosted: bool
    can_abort: bool
    max_prompt_tokens: int
    max_completion_tokens: Optional[int]
    supported_parameters: List[str]
    is_byok_required: bool
    moderation_required: bool
    data_policy: DataPolicy
    pricing: Dict[str, str]
    is_hidden: bool
    is_deranked: bool
    supports_tool_parameters: bool
    supports_reasoning: bool
    limit_rpm: Optional[int]
    limit_rpd: Optional[int]
    has_completions: bool
    has_chat_completions: bool


class OpenRouterModel(TypedDict):
    slug: str
    hf_slug: str
    updated_at: str
    created_at: str
    hf_updated_at: Optional[str]
    name: str
    short_name: str
    author: str
    description: str
    model_version_group_id: Optional[str]
    context_length: int
    modality: str
    has_text_output: bool
    group: str
    instruct_type: Optional[str]
    default_system: Optional[str]
    default_stops: List[str]
    hidden: bool
    router: Optional[str]
    warning_message: Optional[str]
    permaslug: str
    endpoint: Endpoint


class DataPolicy(TypedDict):
    termsOfServiceURL: str
    privacyPolicyURL: str
    training: bool
    requiresUserIDs: bool


class Icon(TypedDict):
    url: str
    invertRequired: bool


class OpenRouterProvider(TypedDict):
    name: str
    displayName: str
    baseUrl: str
    dataPolicy: DataPolicy
    hasChatCompletions: bool
    hasCompletions: bool
    isAbortable: bool
    moderationRequired: bool
    group: str
    editors: List[str]
    owners: List[str]
    isMultipartSupported: bool
    statusPageUrl: Optional[str]
    byokEnabled: bool
    isPrimaryProvider: bool
    icon: Icon


class GetLLMModelsResponse(TypedDict):
    providers: List[OpenRouterProvider]
    models: List[OpenRouterModel]


async def get_llm_models_openrouter() -> GetLLMModelsResponse:
    # r_gui = requests.get(f"https://openrouter.ai/models")
    r_providers = requests.get(
        f"https://openrouter.ai/api/frontend/all-providers")
    r_models = requests.get("https://openrouter.ai/api/frontend/models")
    return {
        "providers": r_providers.json(),
        "models": r_models.json(),
    }


def get_llm_model_2(model_name: str | Enum, use_model_provider: MODEL_PROVIDER_NAMES_TYPE | Enum):
    if isinstance(model_name, Enum):
        model_name = model_name.value
    if isinstance(use_model_provider, Enum):
        use_model_provider = use_model_provider.value

    if use_model_provider == ModelProviderNamesEnum.OPENAI.value:
        # Create LLM with tools
        llm = ChatOpenAI(model=model_name, temperature=0,
                         api_key=SecretStr(OPENAI_API_KEY))
    elif use_model_provider == ModelProviderNamesEnum.GEMINI.value:
        # Initialize the Gemini model
        llm = ChatGoogleGenerativeAI(
            model=model_name, api_key=SecretStr(GOOGLE_API_KEY))
    elif use_model_provider == ModelProviderNamesEnum.AZURE_OPENAI.value:
        llm = AzureChatOpenAI(
            model=model_name,
            api_version='2024-10-21',
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
            api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
        )
    elif use_model_provider == ModelProviderNamesEnum.OLLAMA.value:
        llm = ChatOllama(model=(model_name or "llama3.1:8b"))
    elif use_model_provider == ModelProviderNamesEnum.ANTHROPIC.value:
        llm = ChatAnthropic(model=(model_name or "claude-3-5-sonnet-20240620"),
                            api_key=SecretStr(os.getenv('ANTHROPIC_API_KEY', '')))
    elif use_model_provider == ModelProviderNamesEnum.DEEPSEEK.value or use_model_provider == ModelProviderNamesEnum.KLUSTER_AI.value:
        if ModelGetter.deepseek.DEEPSEEK_REASONER.value == model_name:
            llm = DeepSeekR1ChatOpenAI(
                base_url=os.getenv('DEEPSEEK_ENDPOINT', ''),
                model=(model_name),
                api_key=SecretStr(os.getenv('DEEPSEEK_API_KEY', '')),
            )
        else:
            llm = ChatOpenAI(
                base_url=os.getenv('KLUSTER_AI_ENDPOINT', ''),
                model=(model_name),
                api_key=SecretStr(os.getenv('KLUSTER_AI_API_KEY', '')),
            )
    elif use_model_provider == ModelProviderNamesEnum.OPENROUTER.value:
        llm = ChatOpenAI(
            base_url=os.getenv('OPEN_ROUTER_ENDPOINT', ''),
            model=(model_name or "deepseek-chat"),
            api_key=SecretStr(os.getenv('OPEN_ROUTER_API_KEY', '')),
        )
    else:
        raise ValueError(f"Invalid model provider: {use_model_provider}")
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"Using model: {model_name} from {use_model_provider}")
        try:
            response = llm.invoke("Hello, world!")
            logging.debug(f"Response: {response}")
        except Exception as e:
            logging.error(f"Error invoking llm model {
                          model_name} from {use_model_provider}: {e}")
    return llm

# Callback to update the model name dropdown based on the selected provider
# def update_model_dropdown(llm_provider, api_key=None, base_url=None):
#     """
#     Update the model name dropdown with predefined models for the selected provider.
#     """
#     # Use API keys from .env if not provided
#     if not api_key:
#         api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
#     if not base_url:
#         base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")

#     # Use predefined models for the selected provider
#     if llm_provider in model_names:
#         return gr.Dropdown(choices=model_names[llm_provider], value=model_names[llm_provider][0], interactive=True)
#     else:
#         return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)


def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files


async def capture_screenshot(browser_context):
    """Capture and encode a screenshot"""
    # Extract the Playwright browser instance
    # Ensure this is correct.
    playwright_browser = browser_context.browser.playwright_browser

    # Check if the browser instance is valid and if an existing context can be reused
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None

    # Access pages in the context
    pages = None
    if playwright_context:
        pages = playwright_context.pages

    # Use an existing page or create a new one if none exist
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None

    # Take screenshot
    try:
        screenshot = await active_page.screenshot(
            type='jpeg',
            quality=75,
            scale="css"
        )
        encoded = base64.b64encode(screenshot).decode('utf-8')
        return encoded
    except Exception as e:
        return None
