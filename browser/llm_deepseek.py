from openai import OpenAI
import requests
import pdb
from langchain_openai import ChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)


class OpenRouterModelFetcher:
    def __init__(self, base_url: str, client: Optional[OpenAI] = None):
        self.base_url = base_url
        self.client = client

    async def list_models_openrouter_http(self):
        r = requests.get(f"{self.base_url}/models")
        return r.json()

    async def list_models_openrouter_vision_tools(self):
        r = requests.get(
            f"https://openrouter.ai/models?fmt=cards&order=newest&supported_parameters=tools&modality=text%2Bimage-%3Etext")
        return r.json()

    async def list_models_openrouter(self, mode: Literal['json', 'python'] = "python"):
        if self.client:
            return self.client.models.list().to_dict(mode=mode)
        else:
            return await self.list_models_openrouter_http()


class DeepSeekR1ChatOpenAI(ChatOpenAI):
    # Declare class attributes
    base_url: Optional[str] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Get and remove base_url from kwargs before calling super()
        base_url = kwargs.pop("base_url", None)

        # Call parent class initialization
        super().__init__(*args, **kwargs)

        # Set instance attributes
        self.base_url = base_url
        self.client: OpenAI = OpenAI(
            base_url=base_url,
            api_key=kwargs.get("api_key")
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append(
                    {"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append(
                    {"role": "assistant", "content": input_.content})
            else:
                message_history.append(
                    {"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append(
                    {"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append(
                    {"role": "assistant", "content": input_.content})
            else:
                message_history.append(
                    {"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)
