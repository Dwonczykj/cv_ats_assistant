import asyncio
import argparse
from typing import Optional
from cv_ats_assistant.browser.browser_automation import (
    run_browser_task,
    handle_linkedin_auth
)


async def main(task: Optional[str] = None):
    if not task:
        task = input("Enter the task to perform: ")

    # Check if this is a LinkedIn task requiring authentication
    if "linkedin.com" in task.lower():
        await handle_linkedin_auth(task) # TODO: update this function to be a combination of the @controller function calls tools registered ot handle the captcha, the ask_human controller action and the other functions to save job information and can we combine this controller within langgraph agentic flow?
        """
        ```python
from browser_use import Controller, ActionResult
# Initialize the controller
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)
```
"""
    else:
        history = await run_browser_task(task)

        # Print results
        if history:
            print(f"Final Result: {history.final_result()}")
            if history.errors():
                print(f"Errors: {history.errors()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browser Automation Task Runner")
    parser.add_argument("--task", type=str, help="Task to perform")
    args = parser.parse_args()

    asyncio.run(main(args.task))
