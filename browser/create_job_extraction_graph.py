from enum import Enum
import logging
import os
from typing import Literal
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBinding, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from cv_ats_assistant.browser.llm_deepseek import DeepSeekR1ChatOpenAI
from cv_ats_assistant.case_a import CHUNK_SIZE
from cv_ats_assistant.logging_config import configure_logging
from cv_ats_assistant.models import (
    JobGraphInputState, JobGraphOutputState, JobGraphOverallState, JobInfo)
import cv_ats_assistant.helpers as helpers

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


class JOB_FIELDS(Enum):
    JOB_NAME = "job_name"
    JOB_URL = "job_url"
    COMPANY_DOMAIN_URL = "company_domain_url"
    COMPANY_OWN_JOB_POSTING_URL = "company_own_job_posting_url"
    COMPANY_NAME = "company_name"
    HIRING_MANAGER_NAME = "hiring_manager_name"
    HIRING_MANAGER_URL = "hiring_manager_url"
    COMPANY_ABOUT_CONTENT = "company_about_content"
    COMPANY_VALUES = "company_values"
    KEY_SKILLS = "key_skills"
    KEYWORDS = "keywords"
    FULL_JOB_POST_ON_SOURCE = "full_job_post_on_source"


JOB_FIELDS_READABLE = {
    JOB_FIELDS.JOB_NAME: "Job Name",
    JOB_FIELDS.JOB_URL: "Job URL",
    JOB_FIELDS.COMPANY_DOMAIN_URL: "Company Domain URL",
    JOB_FIELDS.COMPANY_OWN_JOB_POSTING_URL: "Company Own Job Posting URL",
    JOB_FIELDS.COMPANY_NAME: "Company Name",
    JOB_FIELDS.HIRING_MANAGER_NAME: "Hiring Manager Name",
    JOB_FIELDS.HIRING_MANAGER_URL: "Hiring Manager URL",
    JOB_FIELDS.COMPANY_ABOUT_CONTENT: "Company About Content",
    JOB_FIELDS.COMPANY_VALUES: "Company Values",
    JOB_FIELDS.KEY_SKILLS: "Key Skills Section",
    JOB_FIELDS.KEYWORDS: "Keywords Section",
    JOB_FIELDS.FULL_JOB_POST_ON_SOURCE: "Full Job Post on Source Page",
}


def format_with_a_prefix(items: list[str]) -> str:
    return "a " + ", a ".join(items[:-1]) + " and a " + items[-1]


job_fields_list_prefix_a = format_with_a_prefix(
    list(JOB_FIELDS_READABLE.values()))


class JobExtractionPrompts:
    IMPROVE_JOB_INFO_EXTRACTOR_PROMPT = prompt_template = PromptTemplate(
        input_variables=["corrected_job_info"],
        template="""
        You are an expert AI Prompt Engineer.
        You are given a existing prompt for a job information extraction task but this prompt failed to extract the job information correctly.
        You are given a corrected job information that you need to use to update the prompt to extract the job information correctly from the page source.
        Here is the corrected job information:
        {corrected_job_info}
        You will be given the page source in chunks below:
        """
    )
    EXTRACT_JOB_INFO_FROM_CHUNKS_PROMPT = prompt_template = PromptTemplate(
        input_variables=["summarise_current_jobinfo", "next_chunk"],
        template="""
        You are a job description extractor. You are given a web page that contains a job post within it. A job post contains: 
            """
        # a job name, a job description, a company name, a company domain url, a company own job posting url, a hiring manager name, a hiring manager url, a company about content, a company values, a key skills section and a keywords section.
        + job_fields_list_prefix_a +
        """You are given a chunk of this webpage's source and a summary of the job information you have already extracted in previous steps.
            You need to extract the job information from the this context. You should only modify the existing extracted job information if the new information is more accurate or complete.
            The current job information is: {summarise_current_jobinfo}
            The next chunk of job information is:
            {next_chunk}
        """
    )
    EXTRACT_JOB_INFO_FROM_CORRECTED_JOB_INFO_PROMPT = prompt_template = PromptTemplate(
        input_variables=["chunks_to_visit"],
        template="""
            You are a job description extractor. You are given a text file that contains a job post within it. 
            A job post contains:
            """
        + job_fields_list_prefix_a +
        """
            You need to extract the job information from the this context.
            The job post text file is:
            {chunks_to_visit}
            """
    )


def create_job_extraction_graph(
        checkpointer: Checkpointer,
        llm: ChatOpenAI | ChatGoogleGenerativeAI | ChatAnthropic | ChatOllama | AzureChatOpenAI | DeepSeekR1ChatOpenAI,
) -> StateGraph:
    """Create a LangGraph for job extraction"""
    workflow = StateGraph(JobGraphOverallState,
                          input=JobGraphInputState, output=JobGraphOutputState)
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
        prompt_template = JobExtractionPrompts.IMPROVE_JOB_INFO_EXTRACTOR_PROMPT
        prompt = prompt_template.format(
            corrected_job_info=corrected_job_info,
            # page_source=state.chunks_to_visit
        )
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
        prompt_t = JobExtractionPrompts.EXTRACT_JOB_INFO_FROM_CORRECTED_JOB_INFO_PROMPT
        prompt = prompt_t.format(
            chunks_to_visit=state.chunks_to_visit[0]
        )
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
        prompt_t = JobExtractionPrompts.EXTRACT_JOB_INFO_FROM_CHUNKS_PROMPT
        prompt = prompt_t.format(
            summarise_current_jobinfo=summarise_current_jobinfo,
            next_chunk=" ".join(next_chunk)
        )
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
