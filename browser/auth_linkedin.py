"""
Find and apply to jobs.

@dev You need to add OPENAI_API_KEY to your environment variables.

Also you have to install PyPDF2 to read pdf files: pip install PyPDF2
"""

from enum import Enum
import logging
from browser_use.browser.context import BrowserContext
from browser_use import ActionResult, Agent, Controller
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from typing import List, Optional
import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path

from PyPDF2 import PdfReader

from browser_use.browser.browser import Browser, BrowserConfig
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from cv_ats_assistant.browser.llm_deepseek import DeepSeekR1ChatOpenAI
from cv_ats_assistant.logging_config import configure_logging
from cv_ats_assistant.browser.browser_utils import LLM_TYPE, get_llm_model_2, model_names, MODEL_PROVIDER_NAMES_TYPE
from cv_ats_assistant.models import JOB_POST_ATTRS, JobPost

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()

logger = logging.getLogger(__name__)
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
    logger.warning(
        "USER_AGENT environment variable not set, using default USER_AGENT.")


class AuthType(Enum):
    EMAIL_PASSWORD = 'Email and Password Sign-in'
    GOOGLE = 'Google Sign-in'
    APPLE = 'Apple Sign-in'
    MICROSOFT = 'Microsoft Sign-in'
    EMAIL_LINK = 'Email Link Sign-in'


controller = Controller()


class GetCredentialsParams(BaseModel):
    site: str = Field(description='The site to get credentials for')
    auth_type: AuthType = Field(
        description='The type of authentication to use')


@controller.action(
    description='Get the credentials for a given site and auth type',
    param_model=GetCredentialsParams,
)
def get_credentials(params: GetCredentialsParams):
    secret_prefix = params.site.replace(
        "http://", "").replace("https://", "").split(".co")[0].upper()
    if params.auth_type == AuthType.GOOGLE:
        secret_prefix = 'GOOGLE'
    elif params.auth_type == AuthType.APPLE:
        secret_prefix = 'APPLE'
    elif params.auth_type == AuthType.MICROSOFT:
        secret_prefix = 'MICROSOFT'
    username = os.getenv(f'{secret_prefix}_USERNAME')
    if not username:
        raise ValueError(f'No credentials found for {params.site}')
    password = os.getenv(f'{secret_prefix}_PASSWORD')
    if not password:
        raise ValueError(f'No credentials found for {params.site}')
    return ActionResult(extracted_content=f'Username: {username}\nPassword: {password}')


@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)


browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
        chrome_instance_path=None,
        wss_url=None,
        proxy=None,
        cdp_url=None,
        extra_chromium_args=[
            f"--window-size={1280},{1024}",
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ],
    )
)

# TODO: Add the ability to ready my mail using an api connection rather than browser connection using the mail api elsewhere in this repo gmail_rule_deamon and others...

# extract_fields = ["job name", "url", "company", "location", "key_requirements_and_skills",
#                   "required_skills_experience_not_on_cv", "optional_skills_experience_not_on_cv", "full job description", "fit score"]


async def main():
    auth_task = (
        'You are tasked with signing in to {site} which is a {site_type} using {auth_type}. '
        'For any cookies popups or buttons, you should always accept all of them so that the rest of the site can be accessed and the page javascript is run. '
        'Please sign in to the site, the site\'s login page might be a popup, an iframe or an embedded form asking you to sign in. '
        'You can get any credentials needed from get_credentials function. '
        'If the site asks for a verification code, you should use the ask_human function to input the code.'
        'If the site asks for the user to click a link on email or another device, you should use the ask_human function to click the link.'
        'You should not try to navigate away from verification pages, you should stay on the page and wait for the user to input the code or click the link.'
        'If the site is using google authentication, the login page or popup or iframe will show a button with a google logo and the text like "Continue with Google" or "Sign in with Google"'
        'If the site is using apple authentication, the login page or popup or iframe will show a button with a apple logo and the text like "Continue with Apple" or "Sign in with Apple"'
        'If the site is using microsoft authentication, the login page or popup or iframe will show a button with a microsoft logo and the text like "Continue with Microsoft" or "Sign in with Microsoft"'
        'If the site is using email and password authentication, the login page or popup or iframe will show a form with an email field and a password field'
        f'If the auth_type is {
            AuthType.EMAIL_LINK}, you should input the email address and click on the link sent to your email to sign in, then use the ask_human function to input the code sent to your email to complete the sign in process'
        'If you are still not able to sign in, you should ask for help from the user using the ask_human function.'
    )
    sites_to_search = [
        ('LinkedIn', 'job search media site', AuthType.EMAIL_PASSWORD),
        # ('LinkedIn', 'job search media site', AuthType.GOOGLE),
        # ('Glassdoor', 'job search media site'),
        # ('Indeed', 'job search media site'),
    ]

    tasks = []
    tasks += [
        auth_task.format(site=site, site_type=site_type, auth_type=auth_type)
        for site, site_type, auth_type in sites_to_search
    ]

    use_model_provider: MODEL_PROVIDER_NAMES_TYPE = "gemini"
    model_name = "gemini-2.0-flash-exp"

    llm: LLM_TYPE = get_llm_model_2(model_name, use_model_provider)

    agents = []
    for task in tasks:
        agent = Agent(task=task, llm=llm,
                      controller=controller, browser=browser)
        agents.append(agent)

    await asyncio.gather(*[agent.run() for agent in agents])


if __name__ == '__main__':
    asyncio.run(main())
