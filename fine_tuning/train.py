import json
from typing import TypedDict, Optional, List, Literal, NotRequired


class TrainingItem(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    name: NotRequired[str]
    weight: NotRequired[float | int]
    input_variables: NotRequired[dict[str, str]]


class TrainingLine(TypedDict):
    messages: List[TrainingItem]
    weight: float | int


example_train: list[TrainingLine] = [
    {
        "messages": [
            {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "Paris", "weight": 0},
            {"role": "user", "content": "Can you be more sarcastic?"},
            {"role": "assistant",
                "content": "Paris, as if everyone doesn't know that already.", "weight": 1}
        ],
        "weight": 1
    }
]

train: list[TrainingLine] = [
    {

        "messages": [
            {
                "role": "system",
                "content": """You are a job description extractor. You are given a web page that contains a job post within it. A job post contains a job name, a job description, a company name, a company domain url, a company own job posting url, a hiring manager name, a hiring manager url, a company about content, a company values, a key skills section and a keywords section."""
            },
            {
                "role": "user",
                "content": """You need to extract the job information from the this context. You should only modify the existing extracted job information if the new information is more accurate or complete.
                    The current job information is: {summarise_current_jobinfo}
                    The next chunk of job information is:""",
                "input_variables": {
                    "summarise_current_jobinfo": ...
                }
            },

        ],
        "weight": 1
    }
]


def to_json(train: list[TrainingLine]) -> str:
    output = ""
    for line in train:
        single_line = ""
        for message in line["messages"]:
            if "input_variables" in message:
                content = message["content"]
                content = content.format(message["input_variables"])
                message["content"] = content
        output += json.dumps(line) + "\n"
    return output


print(to_json(train))
