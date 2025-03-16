from datetime import datetime
import getpass
import os
import re
from typing import Optional, TypeVar
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup, ResultSet, NavigableString, Tag
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables import RunnableBinding, RunnableConfig
from typing import Literal, Type, TypedDict
from pydantic import BaseModel, Field
from cv_ats_assistant.case_a import WebScraperManager
from selenium.webdriver.remote.webdriver import WebDriver
from urllib.parse import urlparse
from dotenv import load_dotenv
from logging_config import configure_logging, logging

configure_logging()

load_dotenv()


T = TypeVar("T", bound=BaseModel, contravariant=True)


def get_css_path(element: Tag | None) -> str:
    if element is None:
        return ""
    path = []
    while element is not None and element.name != '[document]':
        siblings = element.find_previous_siblings(element.name)
        index = len(siblings) + 1
        path.append(f"{element.name}:nth-of-type({index})")
        if element and element.name == 'html':
            break
        element = element.parent
        if element and element.name == '[document]':
            break
    return " > ".join(reversed(path))


def get_elements_by_css_path(soup: BeautifulSoup, css_path: str):
    """
    Get all elements from a BeautifulSoup object using a CSS path string.

    :param soup: BeautifulSoup object containing the parsed HTML.
    :param css_path: CSS path string to search for elements.
    :return: List of elements matching the CSS path.
    """
    return soup.select(css_path)


class LLMCallTD(TypedDict):
    content: str
    role: Literal["system", "user", "assistant"]
    confidence: float


class AuthAreasOfInterest(BaseModel):
    areas_of_interest: list[str] = Field(...,
                                         description="A list of css paths that are likely to contain login elements")


class PageOverview(TypedDict):
    title: str
    main_sections: list[str]
    forms: int
    inputs: int
    occurences_of_key_word: dict[str, dict[str, int | list[str]]]


class PythonCommands(BaseModel):
    commands: list[str] = Field(
        default_factory=list, description="A list of valid python commands that can be executed with exec()")


class AIAuthenticator:
    def __init__(self):
        self._web_scraper_manager = WebScraperManager()
        self.driver: WebDriver = self._web_scraper_manager.driver  # type: ignore
        self.initialize_llm()  # Placeholder for LLM initialization

    def initialize_llm(self):
        # Initialize your chosen LLM here
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._llms: dict[str, RunnableBinding] = {}

    @property
    def llm(self):
        return self._llm

    def get_page_overview(self, html_content: str) -> PageOverview:
        soup = BeautifulSoup(html_content, 'html.parser')

        def get_count_and_css_paths(soup: BeautifulSoup, keyword: str):
            found: ResultSet = soup.find_all(
                string=re.compile(keyword, re.IGNORECASE))
            ps: list[NavigableString] = [p for p in found]
            pcss = [get_css_path(p.parent) for p in ps]
            return {
                'count': len(ps),
                'css_paths': pcss
            }
        overview: PageOverview = {
            'title': (soup.title.string if soup.title and soup.title.string else '') or '',
            'main_sections': [tag.name for tag in soup.find_all(['header', 'nav', 'main', 'footer'])],
            'forms': len(soup.find_all('form')),
            'inputs': len(soup.find_all('input')),
            'occurences_of_key_word': {
                'login': get_count_and_css_paths(soup, r'login'),
                'sign_in': get_count_and_css_paths(soup, r'sign in'),
                'sign_up': get_count_and_css_paths(soup, r'sign up'),
                'sign_in_with': get_count_and_css_paths(soup, r'sign in with'),
                'sign_up_with': get_count_and_css_paths(soup, r'sign up with'),
            }
        }
        return overview

    def analyze_page_structure(self, overview: PageOverview, html_content: str):
        prompt = f"""
        Analyze this page structure: {overview}
        and then this html_content: {html_content}
        and identify potential login areas as a list of css paths"""
        return self.llm_api_call_with_structured_output(AuthAreasOfInterest, prompt)

    def extract_relevant_sections(self, html_content: str, areas_of_interest: list[str]):
        """
        html_content: str
        areas_of_interest: list[str] is the list of css paths that are likely to contain login elements
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        relevant_html: list[Tag] = []
        for area in areas_of_interest:
            # TODO: Use the huggingface example notebooks multimodal emb edding to embed images of screenshots of webpages to get login sections based on the screenshot and then add this analysis in context window to text langchani.
            sections = soup.select(area)
            if sections:
                relevant_html.extend(sections)
            else:
                logging.warning(f"No sections [css path] found for '{area}'")
        return relevant_html

    def analyze_focused_sections(self, relevant_html):
        prompt = f"Analyze these HTML sections for login elements: {
            relevant_html}"
        return self.llm_api_call(prompt)

    def multi_turn_analysis(self, html_content):
        conversation = [
            {"role": "system", "content": "You are analyzing a webpage for authentication elements."},
            {"role": "user", "content": f"Here's the page overview: {
                self.get_page_overview(html_content)}"}
        ]

        while True:
            response = self.llm_api_call(conversation)
            conversation.append(
                {"role": "assistant", "content": response["content"]})

            if "ANALYSIS_COMPLETE" in response["content"]:
                break

            if "REQUEST_SECTION" in response["content"]:
                section = self.extract_section(
                    html_content, response["content"])
                conversation.append(
                    {"role": "user", "content": f"Here's the requested section: {section}"})

        return self.parse_final_analysis(conversation)

    def generate_selenium_commands(self, analysis: LLMCallTD):
        selenium_version = "4.27.1"
        prompt = f"""
            Generate Selenium Python commands to interact with these elements: {analysis}.

            Each command should be a valid executable python command that I can call exec() on.
            The WebDriver is already initialized and available as self.driver, do not instantiate your own WebDriver.
            You will have access to local variables for username and password when the commands are executed. Please put placeholders for them using squrly parentheses if you need to input them into the webpage using the selenium commands.
            The WebDriver is using version "{selenium_version}" and uses find_element syntax: `self.driver.find_element(By.ID, ...)`, `self.driver.find_element_by_id` is no longer supported.
            """
        return self.llm_api_call_with_structured_output(PythonCommands, prompt)

    def regenerate_generate_selenium_commands(self, i: int, analysis: LLMCallTD, commands: list[str], erroneous_commands: list[tuple[int, str]], html_content: str):
        selenium_version = "4.27.1"
        prompt = f"""
            Generate Selenium Python commands to interact with these elements: {analysis}.

            Each command should be a valid executable python command that I can call exec() on.
            The WebDriver is already initialized and available as self.driver, do not instantiate your own WebDriver.
            You will have access to local variables for username and password when the commands are executed. Please put placeholders for them using squrly parentheses if you need to input them into the webpage using the selenium commands.
            The WebDriver is using version "{selenium_version}" and uses find_element syntax: `self.driver.find_element(By.ID, ...)`, `self.driver.find_element_by_id` is no longer supported.
            Previously, I asked you to perform the same task {i} times and you generated the following commands: {commands}
            However, the following commands failed to execute: {erroneous_commands}
            Please fix the errors and ensure that all commands can be executed successfully for this page: {html_content}
            """
        return self.llm_api_call_with_structured_output(PythonCommands, prompt)

    def execute_commands(self, commands: list[str], final_analysis: LLMCallTD):
        erroneous_commands: list[tuple[int, str]] = []
        all_errors: list[str] = []
        for i, command in enumerate(commands):
            try:
                exec(command)
            except Exception as e:
                # self.driver.find_element(By.ID, "login-email").send_keys(os.getenv("LINKEDIN_USERNAME"))
                error_str = f"Error executing command [{i}]: `{
                    command}` with error:\n{e}\nPlease fix this error."
                erroneous_commands.append((i, error_str))
                all_errors.append(error_str)
                logging.error(error_str)
                return erroneous_commands
                if i == len(commands) - 1:
                    raise Exception("\n".join(all_errors))
        return erroneous_commands

    def get_user_credentials(self):
        if domain := self.get_url_domain(self.driver.current_url) == "www.linkedin.com":
            username = os.getenv("LINKEDIN_USERNAME")
            password = os.getenv("LINKEDIN_PASSWORD")
        else:
            logging.info(f"No credentials found for {domain}")
            username = input("Enter username: ")
            password = getpass.getpass("Enter password: ")
        return username, password

    def get_url_domain(self, url: str):
        return urlparse(url).netloc

    def authenticate(self, url: str):
        self.driver.get(url)
        html_content = self.driver.page_source

        domain = self.get_url_domain(url).replace("www.", "")
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        screenshot = self.driver.get_screenshot_as_png()
        with open(f"./cv_ats_assistant/screenshots/screenshot_{domain}_{ts}.png", "wb") as f:
            f.write(screenshot)
        overview = self.get_page_overview(html_content)
        # TODO: Can we pass the screenshot into the LLM Input for multimodal input to help it extract the job description that is visible only.
        structure_analysis = self.analyze_page_structure(
            overview, html_content)

        relevant_sections = self.extract_relevant_sections(
            html_content, structure_analysis.areas_of_interest)
        focused_analysis = self.analyze_focused_sections(relevant_sections)

        if focused_analysis['confidence'] < 0.8:
            final_analysis = self.multi_turn_analysis(html_content)
        else:
            final_analysis = focused_analysis

        commands_to_execute = self.generate_selenium_commands(final_analysis)

        username, password = self.get_user_credentials()

        erroneous_commands = self.execute_commands(
            commands_to_execute.commands, final_analysis)
        i = 1
        while erroneous_commands:
            if i > 10:
                raise Exception("Failed to execute commands after 10 attempts")
            if html_content != self.driver.page_source:
                raise Exception("Page source changed while executing commands")
            commands_to_execute = self.regenerate_generate_selenium_commands(
                i, final_analysis, commands_to_execute.commands, erroneous_commands, html_content)
            erroneous_commands = self.execute_commands(
                commands_to_execute.commands, final_analysis)

        # Wait for successful login
        WebDriverWait(self.driver, 10).until(EC.url_changes(url))

    def llm_api_call(self, prompt: LanguageModelInput) -> LLMCallTD:
        # Implement your LLM API call here
        result = self._llm.invoke(prompt)
        if isinstance(result.content, str):
            return LLMCallTD(content=result.content, role="assistant", confidence=1.0)
        elif isinstance(result.content, list):
            return LLMCallTD(content=result.content[0] if isinstance(result.content[0], str) else result.content[0]["choices"][0]["message"]["content"], role="assistant", confidence=1.0)
        elif isinstance(result.content, dict):
            return LLMCallTD(content=result.content["choices"][0]["message"]["content"], role="assistant", confidence=1.0)
        else:
            return LLMCallTD(content="", role="assistant", confidence=0.0)

    def llm_api_call_with_structured_output[T](self, structured_type: Type[T], prompt: LanguageModelInput) -> T:
        # Implement your LLM API call here
        if structured_type.__name__ not in self._llms:
            x = self._llm.with_structured_output(structured_type)
            self._llms[structured_type.__name__] = x  # type: ignore
        result: structured_type = self._llms[structured_type.__name__].invoke(
            prompt)
        return result

    def parse_final_analysis(self, conversation: LanguageModelInput):
        # Implement parsing of the final analysis from the conversation
        raise NotImplementedError

    def extract_section(self, html_content, response):
        # Implement extraction of specific sections based on LLM response
        raise NotImplementedError


if __name__ == "__main__":
    authenticator = AIAuthenticator()
    try:
        authenticator.authenticate("https://linkedin.com")
        print("Authentication successful")
    except Exception as e:
        print(f"Authentication failed: {e}")
    finally:
        authenticator.driver.quit()
