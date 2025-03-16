"""
Find and apply to jobs.

@dev You need to add OPENAI_API_KEY to your environment variables.

Also you have to install PyPDF2 to read pdf files: pip install PyPDF2
"""

from dataclasses import dataclass
from datetime import datetime
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
from browser_use.browser.context import (
    BrowserContext,
    BrowserContextConfig,
    BrowserContextWindowSize
)
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from cv_ats_assistant.browser.llm_deepseek import DeepSeekR1ChatOpenAI, OpenRouterModelFetcher
from cv_ats_assistant.logging_config import configure_logging
from cv_ats_assistant.browser.browser_utils import LLM_TYPE, ModelGetter, ModelProviderNamesEnum, get_llm_model_2, model_names, MODEL_PROVIDER_NAMES_TYPE
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
# full screen mode


class AuthType(Enum):
    NONE = 'None'
    EMAIL_PASSWORD = 'Email and Password Sign-in'
    GOOGLE = 'Google Sign-in'
    APPLE = 'Apple Sign-in'
    MICROSOFT = 'Microsoft Sign-in'
    EMAIL_LINK = 'Email Link Sign-in'


class SiteType(Enum):
    JOB_SEARCH_MEDIA = 'job search media site'
    CAREERS_PAGE = 'careers page'


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


# 1. Integrate Langgraph with the controller, browser and agents by having them all
# outisde the definition of the langraph in a Services class that allows
# us unlimited agents, a single browser and a single controller. The Controller then r
# egisters its actions and tools
# outside of the defintiion of the graph within an init_controller sync function.
# The langraph is then also initialised in a init_graph function that for now is a
# simplified graph that uses the browser controller to complete a simple task but by
# taking the task as input to the graph, the nodes of the graph then call the
# browser-use agent to run tasks if the task is a browser-use task which has the
# ask_human controller action and the other functions to save job information
# embedded within the agent.run code before control is then
# passed back into the graph flow? This may break the graph as the graph does not
# allow inputs to run in the function nodes,

# the alternative for 1. is to do the same, but instead of using langgraph, using
# langchain llms instead to begin
# with within the browser-use agents and
# then adjusting once we have that first bit working.

# 2. Integrate the captcha solving controller action to this action and describe the
# action so that the controller
# understands when to use the captcha,
# understands to zoom in and to use a larger LLM to via api for solving the captcha
# and smaller LLMs for the other
# tasks if we still get the same performance.

# 3. Integrate the current functionality that we have in the langraph node in case_a
# into this script. by calling
# the functions without the nodes. Can we
# reuse the same functions, using the state input? but not within a graph?


# NOTE: This is the path to your cv file
CV = Path('/Users/joey/Documents/CVs/Joey_Dwonczyk_CV.pdf')

if not CV.exists():
    raise FileNotFoundError(
        f'You need to set the path to your cv file in the CV variable. CV file not found at {CV}')


@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)


def register_site_controller_actions(llm: LLM_TYPE):
    class CoveringLetter(BaseModel):
        job_title: str
        company_name: str
        covering_letter: str
        job_post_url: str
        date: str

    @controller.action(
        description='Save jobs to file - with a score how well it fits to my profile and other scores as defined in the JobPost parameter model',
        param_model=JobPost,
    )
    def save_jobs(job: JobPost):
        # first check if the file exists, if it does not, create it and init the header row
        dir = Path.cwd() / 'cv_ats_assistant' / 'browser'
        if not os.path.exists(dir / 'jobs.csv'):
            with open(dir / 'jobs.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(JOB_POST_ATTRS)
                writer.writerow([str(getattr(job, attr, ""))
                                for attr in JOB_POST_ATTRS])
        else:
            with open(dir / 'jobs.csv', 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                if header != JOB_POST_ATTRS:
                    raise ValueError(f'The headers in the file are not correct.\nShould be:\n{
                        JOB_POST_ATTRS} but got:\n{header}')

            with open(dir / 'jobs.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([getattr(job, attr, "")
                                for attr in JOB_POST_ATTRS])

        with open(dir / 'jobs.json', 'a', newline='') as f:
            # insert this job object to end of list in json file
            json.dump(job, f)

        # TODO: add to a sqlite db

        return 'Saved job to file'

    @controller.action('Read jobs from file')
    def read_jobs():
        dir = Path.cwd() / 'cv_ats_assistant' / 'browser'
        with open(dir / 'jobs.csv', 'r') as f:
            return f.read()

    @controller.action('Read saved covering letters from file as a json list of strings')
    def read_covering_letters():
        dir = Path.cwd() / 'cv_ats_assistant' / 'browser'
        if not os.path.exists(dir / 'covering_letters.json'):
            with open(dir / 'covering_letters.json', 'w') as f:
                json.dump([], f)
            return []
        covering_letters: list[CoveringLetter] = []
        with open(dir / 'covering_letters.json', 'r') as f:
            covering_letters_json: list[dict] = json.load(f)
            for covering_letter in covering_letters_json:
                covering_letters.append(
                    CoveringLetter.model_construct(**covering_letter))
        return covering_letters

    @controller.action('Read my cv for context to fill forms')
    def read_cv():
        pdf = PdfReader(CV)
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
        logger.info(f'Read cv with {len(text)} characters')
        return ActionResult(extracted_content=text, include_in_memory=True)

    def summarise_job_post(job: JobPost):
        """Summarise the job information"""
        summarised_job_info = f"""
        ----START----
        Job Name: {job.title}
        Job URL: {job.url}
        Company Domain URL: {job.company_domain_url}
        Company Own Job Posting URL: {job.company_domain_url}
        Company Name: {job.company_name}
        Hiring Manager Name: {job.hiring_manager_name}
        Hiring Manager URL: {job.hiring_manager_url}
        Company About Content: {job.company_about_content}
        Company Values: {job.company_values}
        Key Skills: {job.key_words}
        Missing Skills: {job.missing_key_words_in_cv}
        Matching Skills: {job.matching_key_words_in_cv}
        Full Job Post: {job.full_job_post_on_source}
        Description: {job.description}
        Location: {job.location}
        Salary: {job.salary}
        Fit Score: {job.fit_score}
        Required Skills Score: {job.cv_to_job_name_score}
        CV K Score: {job.cv_K_score}
        ----END----
        """
        return summarised_job_info

    @controller.action('Write a covering letter to the hiring manager', param_model=JobPost)
    def write_covering_letter(job: JobPost):
        covering_letter_xml_template = """<CoveringLetter>
        <Header>
            <YourName example="John Doe">[Your Name]</YourName>
            <EmailAddress example="john.doe@example.com">[Your Email Address]</EmailAddress>
            <LinkedIn example="https://linkedin.com/in/johndoe">[LinkedIn Profile or Portfolio Link]</LinkedIn>
            <Date example="2025-01-24">[Date]</Date>
        </Header>
        <EmployerDetails>
            <RecruitmentFirmName example="Elite Tech Recruiters">[Recruitment Firm Name]</RecruitmentFirmName>
            <HiringManagerName example="Alex Taylor">[Recruiter's Name or Team]</HiringManagerName>
        </EmployerDetails>
        <Salutation>
            Dear <Recruiter example="Alex">[Recruiter's Name]</Recruiter>,
        </Salutation>
        <OpeningParagraph>
            <Purpose>
                I am writing to apply for the <JobTitle example="Senior Data Analyst">[Job Title]</JobTitle> position in the <Industry example="financial technology">[Industry]</Industry> sector, as advertised by <RecruitmentFirmNameReference example="Elite Tech Recruiters">[Recruitment Firm Name]</RecruitmentFirmNameReference>.
            </Purpose>
            <Hook>
                With <ExperienceYears example="7">[X years]</ExperienceYears> of experience in <RelevantField example="data analytics and visualization">[relevant field/role]</RelevantField>, I am eager to leverage my expertise to contribute to an organization at the forefront of <IndustryFocus example="fintech innovation">[specific industry focus]</IndustryFocus>.
            </Hook>
        </OpeningParagraph>
        <MiddleParagraphs>
            <KeySkills>
                <Skill example="Advanced data modeling">
                    I have demonstrated expertise in <SpecificAchievement example="building predictive models">[specific accomplishment]</SpecificAchievement>, resulting in <Result example="a 15% increase in forecasting accuracy">[result]</Result>. This reflects my ability to <SkillAbility example="analyze and solve complex business problems">[relevant skill or responsibility]</SkillAbility>.
                </Skill>
                <Skill example="Cross-departmental collaboration">
                    My experience in <FieldOrTechnology example="SQL and Tableau">[specific field or technology]</FieldOrTechnology> has equipped me to effectively tackle challenges such as <Challenge example="integrating data from multiple departments">[specific challenge mentioned in job description]</Challenge>.
                </Skill>
            </KeySkills>
        </MiddleParagraphs>
        <IndustryAlignment>
            <IndustryKnowledge>
                What draws me to this opportunity is the chance to apply my skills within the <IndustryReference example="fintech">[Industry]</Industry>, an area where innovation and impact intersect.
            </IndustryKnowledge>
            <Contribution>
                I am excited to bring my expertise in <KeySkill example="streamlining analytics processes">[key skill/experience]</KeySkill> to support the success of <TargetCompanyType example="an industry-leading organization">[target company type]</TargetCompanyType> and align with their strategic objectives.
            </Contribution>
        </IndustryAlignment>
        <ClosingParagraph>
            <Interest>
                Thank you for considering my application. I am enthusiastic about the opportunity to bring value to the right organization in this role.
            </Interest>
            <CallToAction>
                I would welcome the chance to discuss how my skills and experience align with the needs of the position and am happy to provide additional details or arrange an interview.
            </CallToAction>
        </ClosingParagraph>
        <SignOff>
            <Closing example="Yours sincerely">[Yours sincerely]</Closing>,
            <FullName example="John Doe">[Your Name]</FullName>
        </SignOff>
    </CoveringLetter>"""

        covering_letter_xml_prompt = f"""I need to write a covering letter to the hiring manager for my application to this role.
        Here is the template I would like to follow, but please respond in text format so that I can copy it into a form.

        {covering_letter_xml_template}

        Please write this in plain text using my CV and the context of the job post:\n{summarise_job_post(job)}
        to fill in the template.
        """

        covering_letter_written_llm = llm.invoke(covering_letter_xml_prompt)

        dir = Path.cwd() / 'cv_ats_assistant' / 'browser'
        if not os.path.exists(dir / 'covering_letters.json'):
            with open(dir / 'covering_letters.json', 'w') as f:
                json.dump([], f)
        covering_letters: list[CoveringLetter] = []
        with open(dir / 'covering_letters.json', 'r') as f:
            covering_letters_json: list[dict] = json.load(f)
            for covering_letter in covering_letters_json:
                covering_letters.append(
                    CoveringLetter.model_construct(**covering_letter))
        covering_letters.append(CoveringLetter(
            job_title=job.title,
            company_name=job.company_name,
            covering_letter=covering_letter_written_llm.content,
            job_post_url=job.url,
            date=datetime.now().strftime('%Y-%m-%d'),
        ))
        with open(dir / 'covering_letters.json', 'w') as f:
            json.dump(covering_letters, f)

        return 'Wrote covering letter to file'

    @controller.action(
        'Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element',
        requires_browser=True,
    )
    async def upload_cv(index: int, browser: BrowserContext):
        path = str(CV.absolute())
        dom_el = await browser.get_dom_element_by_index(index)

        if dom_el is None:
            return ActionResult(error=f'No element found at index {index}')

        file_upload_dom_el = dom_el.get_file_upload_element()

        if file_upload_dom_el is None:
            logger.info(f'No file upload element found at index {index}')
            return ActionResult(error=f'No file upload element found at index {index}')

        file_upload_el = await browser.get_locate_element(file_upload_dom_el)

        if file_upload_el is None:
            logger.info(f'No file upload element found at index {index}')
            return ActionResult(error=f'No file upload element found at index {index}')

        try:
            await file_upload_el.set_input_files(path)
            msg = f'Successfully uploaded file to index {index}'
            logger.info(msg)
            return ActionResult(extracted_content=msg)
        except Exception as e:
            logger.debug(f'Error in set_s_files: {str(e)}')
            return ActionResult(error=f'Failed to upload file to index {index}')

    return {
        'upload_cv': upload_cv,
        'read_cv': read_cv,
        'save_jobs': save_jobs,
        'read_jobs': read_jobs,
        'write_covering_letter': write_covering_letter,
        'read_covering_letters': read_covering_letters,
    }


@dataclass
class CreateBrowserResult:
    browser: BrowserContext
    browser_context: BrowserContext


async def create_browser(headless: bool, window_w: int, window_h: int):
    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            disable_security=True,
            chrome_instance_path=None,
            wss_url=None,
            proxy=None,
            cdp_url=None,
            extra_chromium_args=[
                f"--window-size={window_w},{window_h}",
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
    )

    # Create browser context
    browser_context = await browser.new_context(
        config=BrowserContextConfig(
            no_viewport=False,
            browser_window_size=BrowserContextWindowSize(
                width=window_w, height=window_h
            ),
            disable_security=True,
            cookies_file=None,
            trace_path=None,
            save_recording_path="./tmp/recordings" if not headless else None,
        )
    )
    return CreateBrowserResult(
        browser=browser,
        browser_context=browser_context,
    )

# TODO: Add the ability to ready my mail using an api connection rather than browser connection using the mail api elsewhere in this repo gmail_rule_deamon and others...

# extract_fields = ["job name", "url", "company", "location", "key_requirements_and_skills",
#                   "required_skills_experience_not_on_cv", "optional_skills_experience_not_on_cv", "full job description", "fit score"]


async def run_job_crawl(
    headless: bool = False,
    window_w: int = 1280,
    window_h: int = 1024,
    use_vision: bool = False,
):
    llm: LLM_TYPE = get_llm_model_2(
        (model_name := ModelGetter.openrouter.ANTHROPIC_CLAUDE_3_OPUS),
        (use_model_provider := ModelProviderNamesEnum.OPENROUTER),
    )

    def get_agent(llm: LLM_TYPE, controller: Controller, browser: BrowserContext, browser_context: BrowserContext, use_vision: bool, site: str, task_type: str, task: str):
        return Agent(
            task=task,
            llm=llm,
            controller=controller,
            browser=browser,
            browser_context=browser_context,
            use_vision=use_vision,
            # max_actions_per_step=5,
            tool_call_in_content=True,
            max_failures=1,
            retry_delay=2,
            save_conversation_path="./tmp/conversations",
            generate_gif=(
                f'cv_ats_assistant/browser/'
                f'{task_type}_task_{site}_agent_history.gif'
            )
        )
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
        ('LinkedIn', 'https://www.linkedin.com',
         SiteType.JOB_SEARCH_MEDIA, AuthType.EMAIL_PASSWORD),
        ('ai.gov.uk', 'https://ai.gov.uk/join',
         SiteType.CAREERS_PAGE, AuthType.NONE),
        # ('LinkedIn', 'job search media site', AuthType.GOOGLE),
        # ('Glassdoor', 'job search media site'),
        # ('Indeed', 'job search media site'),
    ]
    job_extraction_task = (
        'You are a professional job finder. '
        '1. Read my cv with read_cv and '
        '2. Read the saved jobs file '
        '3. Find AI engineering roles using python in the UK on {site_type} {site} at url: [{site_url}]({site_url}) that are actively hiring and save them to the saved jobs file if they are not already in the file '
        ' When on a job page, you should get the link to the company page from the link text containing the company name and open the link in a new tab to extract company information and the company\'s website url from the company page which on linkedin is often in the "About" section, '
        ' you can also query a search engine for the company\'s website to get the company\'s website url, "About" page contents and "Values".'
        ' You may then also be able to get the find the same job posting on the company\'s website and extract the company\'s website url from the job posting page. '
    )
    cover_letter_task = (
        'Read the saved jobs file, read my cv using the read_cv function and read the '
        'saved covering letters file using the read_covering_letters function and write covering '
        'letters for each job post that does not have a covering letter already written '
        'using the write_covering_letter function'
    )
    auth_tasks = [
        (auth_task.format(site=site, site_type=site_type.value,
                          auth_type=auth_type.value), site)
        for site, site_url, site_type, auth_type in sites_to_search if auth_type != AuthType.NONE
    ]
    tasks = []
    tasks += [
        (job_extraction_task.format(
            site=site, site_type=site_type.value, site_url=site_url), site)
        for site, site_url, site_type, auth_type in sites_to_search
    ]
    logging.info(f'Using model: {model_name} from {use_model_provider}')
    agents: list[Agent] = []
    browser_result = await create_browser(headless=False, window_w=1280, window_h=1024)
    browser = browser_result.browser
    browser_context = browser_result.browser_context
    for index, (task, site) in enumerate(auth_tasks):
        print(f'Running auth task {index + 1} of {len(auth_tasks)}')
        agent = get_agent(llm=llm, controller=controller, browser=browser, browser_context=browser_context,
                          use_vision=True, site=site, task_type='auth', task=task)
        agents.append(agent)

    auth_results = await asyncio.gather(*[agent.run() for agent in agents])
    auth_results_dict = {site[0]: auth_result for site,
                         auth_result in zip(sites_to_search, auth_results)}
    for site, auth_result in auth_results_dict.items():
        logging.debug(f'Authentication result for {site}: {auth_result}')
        if auth_result.errors():
            logging.error(f'Authentication failed for {
                          site}: {auth_result.errors()}')
            # raise ValueError(f'Authentication failed for {site}: {auth_result.errors()}')
            # sites_to_search.remove(
            #     next(s for s in sites_to_search if s[0] == site))
            if isinstance(llm, DeepSeekR1ChatOpenAI):
                models = await llm.client.models.list().to_dict(mode='python')
                logging.info([m for m in models])
            try:
                openrouter_model_fetcher = OpenRouterModelFetcher(
                    base_url='https://openrouter.ai', client=llm.client)
                models = await openrouter_model_fetcher.list_models_openrouter_vision_tools()
                logging.info([m for m in models])
            except Exception as e:
                logging.error(f'Error fetching models: {str(e)}')

    logging.info("Authentication tasks complete")

    registered_actions_dict = register_site_controller_actions(llm=llm)
    logging.debug(f"Registered actions: {registered_actions_dict.keys()}")
    agents = []
    for index, task in enumerate(tasks):
        print(f'Running task {index + 1} of {len(tasks)}')
        agent = get_agent(llm=llm, controller=controller, browser=browser, browser_context=browser_context,
                          use_vision=False, site=site, task_type='job_extraction', task=task)
        agents.append(agent)

    await asyncio.gather(*[agent.run() for agent in agents])


if __name__ == '__main__':
    asyncio.run(run_job_crawl())
