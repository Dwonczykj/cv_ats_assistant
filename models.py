from dataclasses import asdict
import operator
from typing import Annotated, Dict, List, Literal, Optional, Protocol, Tuple, Any
from typing_extensions import TypedDict
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from selenium.webdriver.remote.webdriver import WebDriver
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pprint import pformat

from cv_ats_assistant.type_to_typeddict import create_typed_dict_from_class, keys_of


class WebDriverProtocol(Protocol):
    """Protocol defining the required WebDriver interface."""

    def get(self, url: str) -> None: ...
    def quit(self) -> None: ...
    def refresh(self) -> None: ...
    def execute_script(self, script: str) -> Any: ...
    def find_element(self, by: str, value: str) -> Any: ...
    @property
    def current_url(self) -> str: ...
    @property
    def page_source(self) -> str: ...
    @property
    def title(self) -> str: ...
    def get_cookies(self) -> List[Dict[str, Any]]: ...
    def add_cookie(self, cookie_dict: Any) -> None: ...


class JobInfo(BaseModel):
    job_name: str = Field(
        default="", description="Job title/name")
    full_job_post_on_source: str = Field(
        default="", description="The full job description & specifications text from the job post")
    job_url: str = Field(
        default="", description="Original job posting URL")
    company_domain_url: str = Field(
        default="", description="Company's main website URL")
    company_own_job_posting_url: str = Field(
        default="", description="Direct link to job on company website")
    company_name: str = Field(
        default="", description="Name of the company")
    hiring_manager_name: str = Field(
        default="", description="Name of hiring manager if available")
    hiring_manager_url: str = Field(
        default="", description="LinkedIn/contact URL of hiring manager")
    company_about_content: str = Field(
        default="", description="Content from company's about page")
    company_values: List[str] = Field(
        default=[], description="Company's core values")
    key_skills: List[str] = Field(
        default=[], description="Required skills in order of importance")
    keywords: List[str] = Field(
        default=[], description="Key terms from job description")

    @classmethod
    def default(cls):
        return cls(
            job_name="",
            full_job_post_on_source="",
            job_url="",
            company_domain_url="",
            company_own_job_posting_url="",
            company_name="",
            hiring_manager_name="",
            hiring_manager_url="",
            company_about_content="",
            company_values=[],
            key_skills=[],
            keywords=[],
        )


class JobPost(BaseModel):
    """Model representing a job posting with essential information."""
    title: str = Field(...,
                       description="The title of the job posting")
    url: str = Field(...,
                         description="The full URL of the job posting")
    company_name: Optional[str] = Field(default=None,
                                        description="The name of the company")
    company_domain_url: Optional[str] = Field(default=None,
                                                  description="The domain URL of the company that is hiring for the job")
    company_own_job_posting_url: Optional[str] = Field(default=None,
                                                           description="The URL of the company's own job posting")
    hiring_manager_name: Optional[str] = Field(default=None,
                                               description="The name of the hiring manager")
    hiring_manager_url: Optional[str] = Field(default=None,
                                                  description="The profile URL of the hiring manager if they have one.")
    company_about_content: Optional[str] = Field(default=None,
                                                 description="The about content of the company")
    company_values: Optional[List[str]] = Field(default_factory=list,
                                                description="The values of the company")
    description: Optional[str] = Field(default=None,
                                       description="Brief description or snippet of the job posting")
    full_job_post_on_source: Optional[str] = Field(default=None,
                                                   description="The full contents of the job post")
    fit_score: float = Field(default=0.0,
                             description="The score of how well the job fits to my profile")
    location: Optional[str] = Field(default=None,
                                    description="The location of the job")
    salary: Optional[str] = Field(default=None,
                                  description="The salary of the job")
    key_words: List[str] = Field(
        default_factory=list, description="The key words from the job description in order of importance, most important first dictated by semantic similarity to the job name and number of occurrences in the job description and whether the key word appears in the required skills section of the job description.")
    matching_key_words_in_cv: List[str] = Field(
        default_factory=list, description="The key words from the job description that also appear in my CV (or have a semantically very similar token)")
    missing_key_words_in_cv: List[str] = Field(
        default_factory=list, description="The key words from the job description that do not appear in my CV in order of importance same as the key_words field.")
    cv_to_job_name_score: int = Field(
        default=0, description="The score of how well my CV semantically aligns to the job name as score between 0 and 100")
    cv_headline_to_job_name_score: int = Field(
        default=0, description="The semantic similarity score between my headline title role on my CV at the top of the summary section at the beginning of the document and the job name as score between 0 and 100")
    cv_K_score: int = Field(
        default=0, description="a score out of K where K:= is defined as the number of required skills appearing in the job description (which does not include optional or nice-to-have skills) in the Job Description. Give my CV 1 point from K possible points for each skill that I also mention in my CV from the required skills only. Do not hallucinate. Sum the points and return a response as '{i}/K'.format(my_skills_required)")


JOB_POST_ATTRS = [
    'title',
    'url',
    'company_name',
    'company_domain_url',
    'description',
    'full_job_post_on_source',
    'fit_score',
    'location',
    'salary',
    'company_own_job_posting_url',
    'hiring_manager_name',
    'hiring_manager_url',
    'company_about_content',
    'company_values',
    'key_words',
    'matching_key_words_in_cv',
    'missing_key_words_in_cv',
    'cv_to_job_name_score',
    'cv_headline_to_job_name_score',
    'cv_K_score'
]
assert all(attr in JobPost.model_fields for attr in JOB_POST_ATTRS), f"The attributes {
    [str(attr) for attr in JOB_POST_ATTRS if attr not in JobPost.model_fields]} are not all present in the JobPost class"
assert all(field in JOB_POST_ATTRS for field in JobPost.model_fields), f"The fields {
    [str(field) for field in JobPost.model_fields if field not in JOB_POST_ATTRS]} are not all present in the JobPost class"


class CVScores(BaseModel):
    experience_score: int = Field(
        0, description="Score for relevant experience")
    grammar_score: int = Field(0, description="Score for CV grammar")
    skills_score: int = Field(0, description="Score for matching skills")
    ats_score: int = Field(0, description="ATS compatibility score")
    clarity_score: int = Field(0, description="Score for CV clarity")
    role_fit_score: int = Field(
        0, description="Overall fit score for the role")
    overall: int = Field(0, description="Average of all scores")


class KeywordAnalysis(BaseModel):
    all_keywords: List[str] = Field(...,
                                    description="All keywords in order of importance")
    matching_keywords: List[str] = Field(...,
                                         description="Keywords found in CV")
    missing_keywords: List[str] = Field(...,
                                        description="Important keywords missing from CV")
    required_skills_score: Tuple[int, int] = Field(
        ..., description="Score of matching required skills")
    semantic_scores: Tuple[int, int] = Field(
        ..., description="Job and title semantic similarity scores")


class JobGraphOutputState(BaseModel):
    """The output state of the job graph"""
    job_info: JobInfo = Field(
        default=..., description="Job information")
    error: Optional[str] = Field(
        default=None, description="Error message if job information extraction fails")


class JobGraphInputState(BaseModel):
    """The input state for the job graph"""
    job_post_url: str = Field(..., description="URL of the job posting")
    chunks_to_visit: List[str] = Field(
        [], description="Chunks to visit")
    job_info_summary: str = Field(
        default="", description="Summary of job information")


class JobGraphOverallState(JobGraphInputState, JobGraphOutputState, BaseModel):
    """The overall state of the job graph"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    job_info: JobInfo = Field(
        default=JobInfo.default(), description="Job information")
    error: Optional[str] = Field(
        default=None, description="Error message if job information extraction fails")
    prompt_job_info_extraction_template: str = Field(
        default="", description="The prompt for the job information extraction")
    incorrect_job_info_extraction: bool = Field(
        default=False, description="Whether the job information extraction is incorrect")


AuthActionActionEnum = Literal["wait", "reload", "execute_js",
                               "click_button", "sign_in_with_credentials", "solve_captcha", "dont_know"]


class AuthAction(BaseModel):
    """The action to take to bypass the challenge page"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    action: AuthActionActionEnum = Field(
        ...,
        description="The action to take to bypass the challenge page if it is a challenge page or to authenticate if it is a sign in page or captcha page"
    )
    details: Dict[str, Any] = Field(
        ...,
        description="Required details for the action. For 'wait': {'duration': seconds}, for 'execute_js': {'script': js_code}, for 'click_button': {'selector': css_selector}, for 'sign_in_with_credentials': {'username': username, 'password': password}, for 'solve_captcha': {'captcha_type': captcha_type, 'captcha_id': captcha_id}, for 'dont_know': {'reasoning': reasoning}"
    )
    action_history: Annotated[List[str], operator.add] = Field(
        default=[], description="The history of the actions taken")


class LinkedInCredentials(TypedDict):
    username: str
    password: str


class WebDocument(BaseModel):
    """The document of the web page"""
    url: str = Field(..., description="The URL of the web page")
    page_content: str = Field(...,
                              description="The page source of the web page")
    markdownified_page_source: str = Field(
        "", description="The markdownified web page source")
    chunks: list[str] = Field(
        [], description="The chunks of the markdownified web page source")
    title: str = Field("", description="The title of the web page")

    preprocessed_document_str: str = Field(
        "", description="The preprocessed page source of the web page")
    webpage_category: str = Field(
        "", description="The classification of the web page such as authentication, authentication_challenege, job_description, job_posting, job_listings, news_feed, blog, etc.")


class WebContentActioniserInputState(BaseModel):
    """The state of the web content"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field("",
                       description="The query to search the vectorstore for")
    predefined_actions: list[AuthActionActionEnum] = Field(
        [], description="The predefined actions to classify the web page content into")
    uri: str = Field("", description="The URI or file path of the web page")
    uri_type: Literal["url",
                      "file_path",
                      ] = Field("url", description="The type of the URI")


class WebContentActioniserOverallState(WebContentActioniserInputState, BaseModel):
    """The state of the web content"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore_path: Optional[str] = Field(
        default=None, description="The path to the vectorstore of the web page")
    documents: list[WebDocument] = Field(
        default=[], description="The documents of the web page")
    chunk_size: int = Field(
        default=1000, description="The size of the chunks to split the web page content into")
    chunk_overlap: int = Field(
        default=200, description="The overlap of the chunks to split the web page content into")
    top_k: int = Field(
        default=5, description="The number of similar chunks to retrieve from the vectorstore")
    k_similar_chunks: list[Document] = Field(
        default=[], description="The similar chunks to the query retrieved from the vectorstore")
    summary: str = Field(default="", description="The summary of the web page")
    action: Optional[AuthAction] = Field(
        default=None, description="The action to take to bypass the authentication challenge")


class AuthenticationGraphOutputState(BaseModel):
    """Output from authentication workflow"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    current_url: str = Field(default="",
                             description="The current url of the web driver")
    url_history: Annotated[list[str], operator.add] = Field(default=[],
                                                            description="The history of the urls visited")
    current_title: str = Field("",
                               description="The current title of the web driver")
    current_page_source: str = Field(
        "", description="The contents of the most current page on the web driver")
    domain: str = Field(
        "", description="The domain name of the initial_url obtaining after parsing the initial_url")


class AuthenticationGraphInputState(BaseModel):
    """State for authentication workflow"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    job_post_url: str = Field(
        ..., description="The initial url to GET and watch for redirects to authentication pages.")
    test_mode: bool = Field(
        default=False, description="Whether to run in test mode")


class AuthenticationGraphOverallState(AuthenticationGraphInputState, AuthenticationGraphOutputState, WebContentActioniserOverallState, BaseModel):
    """State for authentication workflow"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # driver: WebDriver | WebDriverProtocol = Field(
    #     ..., description="The web driver instance")
    challenge_probability: float = Field(default=1.0, description="The probability is the probability that the page is a challenge page, security verification, or 2FA page. If the page is likely a challenge page, security verification, or 2FA page, then the probability should be close to 1. If the page is not likely a challenge page, security verification, or 2FA page, then the probability should be close to 0.")
    credentials: Optional[Dict[str, str] | LinkedInCredentials] = Field(
        default=None, description="credentials required for the requested domain")
    error: Optional[str] = Field(
        default=None, description="Errors encountered whilst running the authentication graph")
    authenticated: bool = Field(
        default=False, description="Whether the domain is authenticated")
    auth_challenge_page_action_type: Optional[str] = Field(
        None, description="The type of action the user is being asked to do. This should be one of the following: login, sign in, challenge, 2FA, captcha, confirm_using_mobile_device or other")
    auth_challenge_page_action_reasoning: Optional[str] = Field(
        None, description="Explanation of the auth_challenge_page_action_type")
    auth_challenge_page_action_reason_source: Optional[str] = Field(
        None, description="The section of the page that the reasoning is based on including the markdown the reasoning was taken from")
    auth_challenge_page_action_list_summary: Optional[str] = Field(
        None, description="An ordered list of points summarizing each part of the page")
    require_human_auth: bool = Field(
        False, description="Whether the user needs to authenticate to continue")

    # initial_url: str = Field(description="Initial url to GET and watch for redirects to authentication pages.")
    # current_url: Optional[str] = Field(default=None, description="The current url of the web driver")
    # current_page_source: Optional[str] = Field(default=None, description="The contents of the most current page on the web driver")
    # driver: Optional[WebDriverProtocol] = Field(default=None, description="The web driver instance")
    # challenge_probability: float = Field(default=1.0, description="The probability is the probability that the page is a challenge page, security verification, or 2FA page. If the page is likely a challenge page, security verification, or 2FA page, then the probability should be close to 1. If the page is not likely a challenge page, security verification, or 2FA page, then the probability should be close to 0.")
    # credentials: Optional[Dict[str, str] | LinkedInCredentials] = Field(default=None, description="credentials required for the requested domain")
    # error: Optional[str] = Field(default=None, description="Errors encountered whilst running the authentication graph")
    # # Use field with default_factory to avoid initialization issues
    # domain: str = Field(default="", description="The domain name of the initial_url obtaining after parsing the initial_url")

    def __post_init__(self):
        """Initialize derived fields after dataclass initialization."""
        if self.job_post_url:
            self.domain = urlparse(self.job_post_url).netloc
        else:
            self.domain = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary, excluding the driver."""
        state_dict = self.model_dump(exclude={"driver"})
        return state_dict


class GraphInputState(AuthenticationGraphInputState, BaseModel):
    cv_path: Optional[str] = Field(None, description="Local path to CV file")


def add_error_reducer(error: Optional[str], new_error: Optional[str]) -> Optional[str]:
    if error is None:
        return new_error
    elif new_error is None:
        return error
    else:
        return error + "\n" + new_error


class GraphOutputState(BaseModel):
    job_info: Optional[JobInfo] = Field(
        default=None, description="Job information")
    error: Annotated[Optional[str], add_error_reducer] = Field(
        default=None, description="Error encountered in the graph")


class GraphState(GraphInputState, GraphOutputState, AuthenticationGraphOverallState, JobGraphOverallState, BaseModel):
    """State for the graph"""
    credentials: Optional[Dict[str, Any]] = Field(
        default=None, description="Credentials for authentication")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # auth_graph_state: Optional[AuthenticationGraphOverallState] = Field(
    #     default=None)
    # job_graph_state: Optional[JobGraphOverallState] = Field(
    #     default=None)
    cv_embedding: Optional[str] = Field(
        None, description="CV document embeddings cached path")
    cv_scores: Optional[CVScores] = Field(
        None, description="CV scoring results")
    keyword_analysis: Optional[KeywordAnalysis] = Field(
        None, description="Keyword analysis results")
    altered_cv_path: Optional[str] = Field(
        None, description="Path to modified CV")
    suggested_changes: Annotated[Optional[List[Dict[str, Any]]], operator.add] = Field(
        [], description="Suggested CV modifications")
    user_approved_changes: Optional[List[Dict[str, Any]]] = Field(
        [], description="User-approved modifications")
    user_approved_changes_str: Optional[str] = Field(
        None, description="User-approved modifications in JSON format")

    def __repr__(self):
        return pformat(self.to_dict())

    def __str__(self):
        return self.__repr__()


# GraphStateDict = create_typed_dict_from_class(GraphState)
GraphStateKeys = keys_of(GraphState)

GraphStateDict = dict[GraphStateKeys, Any]
