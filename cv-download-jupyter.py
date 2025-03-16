from langchain.chains import LLMSearchChain, human_in_loop
from langchain.agents import LLMChainAgent
from langchain.chatbots import Messenger
import pydantic as pd
from pydantic import BaseModel
import playwright.sync_api as pws
import asyncio

# Import Pydantic model


class JobPost(pd.BaseModel):
    job_description: str = pd.Field(..., description="Job Description")
    job_name: str = pd.Field(..., description="Job Name")
    job_url: str = pd.Field(..., description="Job URL")
    company_domain_url: str = pd.Field(..., description="Company Domain URL")
    company_own_job_posting_url: str = pd.Field(
        ..., description="Company Own Job Posting URL")
    hiring_manager_name: str = pd.Field(..., description="Hiring Manager Name")
    hiring_manager_url: str = pd.Field(..., description="Hiring Manager URL")


class State(BaseModel):
    job_post_url: str = pd.Field(...)
    job_info: JobPost = pd.Field(default=None, title="Job Information")


def login_to_website(username: str, password: str, browser: pws.Browser) -> pws.Page:
    page = browser.new_page()
    page.goto("https://www.linkedin.com/login/")

    # Fill out form
    # Replace with your own API key
    page.fill("text=x-api-key", "YOUR_API_KEY")
    page.fill("text=username", username)
    page.fill("text=password", password)

    # Submit form
    page.click("text=Sign in")

    return page


async def extract_job_info(browser: pws.Browser, page: pws.Page, job_post_url: str) -> JobPost:
    await page.goto(job_post_url)
    response = await page.content()
    # Use langchain.chain.extract to extract the job info
    chain = Chain.from_language_model(
        language_model="llama",
        chain=[
            "langchain.chains.extract",
            "mания.JobPost.parse",
        ],
    )

    result = chain.run(input={"html": response})
    return result.json_output
