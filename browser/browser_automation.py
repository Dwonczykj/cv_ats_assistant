import os
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContext,
    BrowserContextConfig,
    BrowserContextWindowSize
)
from cv_ats_assistant.browser.browser_utils import get_llm_model
from browser_use.agent.views import AgentHistoryList
from cv_ats_assistant.browser.browser_utils import MODEL_PROVIDER_NAMES_TYPE, model_names
load_dotenv()

# Configure logging based on environment variable
log_level = os.getenv("BROWSER_USE_LOGGING_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


def get_appropriate_model(use_vision: bool) -> tuple[str, str]:
    """
    Get the appropriate model name based on whether vision is required
    Returns: (provider, model_name)
    """
    model_provider = input("Enter the model provider: \n")
    while model_provider not in model_names.keys():
        model_provider = input(f"model_provider must be one of: {
                               model_names.keys()}\nEnter the model provider: \n")
        if not model_provider:
            model_provider = "openai"
    model_name = ""
    while model_name not in model_names[model_provider]:
        model_name = input(f"model_name must be one of: {
                           model_names[model_provider]}\nEnter the model name: \n")
        if not model_name:
            model_name = "gpt-4o"
    return model_provider, model_name


async def run_browser_task(
    task: str,
    llm_provider: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    llm_temperature: float = 0.7,
    headless: bool = False,
    window_w: int = 1280,
    window_h: int = 1024,
    use_vision: bool = True,
    max_steps: int = 20
) -> AgentHistoryList:
    """
    Run a browser automation task using the browser_use package
    """
    browser = None
    browser_context = None

    try:
        # Get appropriate model if not specified
        if not llm_provider or not llm_model_name:
            llm_provider, llm_model_name = get_appropriate_model(use_vision)
            logger.info(f"Using model: {llm_provider}/{llm_model_name}")

        # Initialize browser with proper BrowserConfig
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

        # Get LLM model
        llm = get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            temperature=llm_temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Create and run agent
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            use_vision=use_vision,
            max_actions_per_step=5,
            tool_call_in_content=True,
            max_failures=3,
            retry_delay=2,
            save_conversation_path="./tmp/conversations",
            generate_gif='cv_ats_assistant/browser/agent_history.gif'
        )

        logger.info(f"Starting task: {task}")
        history = await agent.run(max_steps=max_steps)
        return history

    except Exception as e:
        logger.error(f"Error during browser task: {str(e)}")
        raise
    finally:
        # Cleanup
        if browser_context:
            try:
                await browser_context.close()
            except Exception as e:
                logger.warning(f"Error closing browser context: {str(e)}")
        if browser:
            try:
                await browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {str(e)}")


async def handle_linkedin_auth(task: str) -> None:
    """
    Special handler for LinkedIn authentication tasks
    """
    username = os.getenv("LINKEDIN_USERNAME")
    password = os.getenv("LINKEDIN_PASSWORD")

    if not username or not password:
        logger.error("LinkedIn credentials not found in environment variables")
        return

    # Modify task to include login credentials
    auth_task = f"""
    {task}
    Use these credentials when asked to login, if you see a sign-in button, or a sign-in with email button click it to get to the login screen:
    Username/Email: {username}
    Password: {password}

    If you encounter a two-factor authentication request:
    1. Notify the user by printing "2FA Required - Please check your device"
    2. Wait for user to complete 2FA
    3. After authentication is complete, continue with the original task.
    """

    try:
        # For LinkedIn, we want to use vision but with a non-headless browser
        history = await run_browser_task(
            auth_task,
            headless=False,  # LinkedIn often needs visual verification
            max_steps=30,  # Increase steps for auth workflow
            use_vision=True,  # LinkedIn needs vision for CAPTCHA/verification
            llm_provider="openai",
            llm_model_name="gpt-4o",
            
        )

        # Print results
        if history:
            print(f"Final Result: {history.final_result()}")
            if history.errors():
                print(f"Errors: {history.errors()}")
    except Exception as e:
        logger.error(f"LinkedIn automation failed: {str(e)}")
