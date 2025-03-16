import asyncio
import json
from typing import Any, AsyncIterator, Callable, Literal, overload
from cv_ats_assistant.case_a import create_CV_curation_graph
from cv_ats_assistant.models import GraphState
import cv_ats_assistant.helpers as helpers
from logging_config import logging
from PIL import Image
import io
from langgraph.graph import Graph, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from cv_ats_assistant.helpers import draw_graph
from langgraph_sdk.schema import Thread
from langgraph.types import StreamMode, StateSnapshot
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid
import sqlite3
from pprint import pprint, pformat

logger = logging.getLogger(__name__)


async def view_graph_state_at_interrupt(checkpointer: AsyncSqliteSaver | SqliteSaver, graph: CompiledStateGraph, thread: RunnableConfig = {"configurable": {"thread_id": "1"}}):
    if isinstance(checkpointer, AsyncSqliteSaver):
        state = await graph.aget_state(thread)
    else:
        state = graph.get_state(thread)
    pprint(f"State at interrupt:\n{state}\n\nwith next node: {state.next}")


PERSISTENT_CHECKPOINT_DB_PATH = "cv_ats_assistant/case_a_checkpoints.db"

if __name__ == "__main__":

    async def get_graph_state(thread_id: str):
        db_path = PERSISTENT_CHECKPOINT_DB_PATH
        async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
            cv_curator_graph, configure_graph, interruption_handler, compiler = create_CV_curation_graph(
                memory)
            # ONLY USE Interrupt_before when using streaming so that we can restart the graph from the checkpoint using None as input trick.
            graph = cv_curator_graph.compile(
                checkpointer=memory,
                interrupt_before=["human_interrupt", "suggest_cv_changes"])
            graph_config: RunnableConfig = {"configurable": {}}
            thread: Thread | dict[str, Any] = {"thread_id": thread_id}
            graph_config = {"configurable": thread}

            try:
                new_graph_config: RunnableConfig = {}
                if isinstance(memory, AsyncSqliteSaver):
                    existing_state = await graph.aget_state(graph_config)
                    existing_checkpoint = await memory.aget(graph_config)
                    new_graph_config = {
                        "configurable": {
                            "thread_id": thread['thread_id'],
                            "checkpoint_ns": "",
                            "checkpoint_id": existing_checkpoint.get('id', "") if existing_checkpoint else "",

                        }
                    }
                    logging.info(f"Using memory checkpoint with \n\tthread_id: {
                        thread['thread_id']} and \n\tcheckpoint_id: {existing_checkpoint.get('id', "") if existing_checkpoint else ""} and \n\tgraph_state: {existing_state}")
                elif isinstance(memory, SqliteSaver):
                    existing_state = await graph.aget_state(graph_config)
                    existing_checkpoint = memory.get(graph_config)
                    new_graph_config = {
                        "configurable": {
                            "thread_id": thread['thread_id'],
                            "checkpoint_ns": "",
                            "checkpoint_id": existing_checkpoint.get('id', "") if existing_checkpoint else "",

                        }
                    }
                    logging.info(f"Using memory checkpoint with \n\tthread_id: {
                        thread['thread_id']} and \n\tcheckpoint_id: {existing_checkpoint.get('id', "") if existing_checkpoint else ""} and \n\tgraph_state: {existing_state}")
                if input("Use existing state? y/n\n").lower() == 'y':
                    graph_config = new_graph_config or graph_config

            except Exception as e:
                logging.error(
                    f"Error getting state, consider that the graph_state may be empty: {e}")

    async def main(checkpoint_id: str, thread_id: str, fork_graph_state: bool, stream_mode: bool, use_last_thread: bool):
        if not thread_id:
            thread_id = str(uuid.uuid4())
            print(f"Using new thread_id: {thread_id}")
        linkedin_jobs_listings_url = "https://www.linkedin.com/jobs/"

        test_mode = False
        if test_mode:
            job_post_url = "./page_source/www.linkedin.com_20250113_140134_07c34036_page_source.html"
        else:
            job_post_urls = [
                "https://www.linkedin.com/jobs/collections/similar-jobs/?currentJobId=4116841841&originToLandingJobPostings=4116841841"
                "https://www.linkedin.com/jobs/search/?currentJobId=4126139445&f_TPR=a1736601023-&geoId=101165590&origin=JOB_ALERT_EMAIL&savedSearchId=7529070914",
                "https://cord.co/u/granola/jobs/203886-ai-engineer?listing_id=203886"
            ]
            job_post_url = job_post_urls[-1]
        STREAM_MODE: bool = True or checkpoint_id is not None

        USE_LANGGRAPH_STUDIO: bool = False
        USE_CHECKPOINTS_MEMORY: bool = True
        USE_MEMORY = False
        graph_args = {
            "job_post_url": job_post_url,
            "cv_path": "https://dwonczykj.github.io/assets/pdfs/Joey%20Dwonczyk%20CV.pdf",
            "test_mode": test_mode
        }

        async def run_graph():
            db_path = PERSISTENT_CHECKPOINT_DB_PATH if not USE_MEMORY else ":memory:"
            checkpointer: AsyncSqliteSaver | SqliteSaver
            async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
                cv_curator_graph, configure_graph, interruption_handler, compiler = create_CV_curation_graph(
                    checkpointer=checkpointer)
                graph = compiler(cv_curator_graph, checkpointer)
                graph_config, stream_graph_fn = await configure_graph(
                    graph=graph,
                    graph_args=graph_args,
                    checkpointer=checkpointer,
                    checkpoint_id=checkpoint_id,
                    thread_id=thread_id,
                    use_last_thread=use_last_thread,
                )
                latest_state = await graph.aget_state(graph_config)
                next_node = next(iter(latest_state.next), "")
                while next_node != "END":
                    try:
                        async for event in stream_graph_fn(runnable=None):
                            logger.debug(pformat(event))
                        latest_state = await graph.aget_state(graph_config)
                        next_node = next(iter(latest_state.next), "")
                        stream_graph_fn = await interruption_handler(graph, graph_config, checkpointer=checkpointer)
                    except Exception as e:
                        logger.error(f"Error interrupting graph: {e}")
                        raise e

        async def run_graph2(
            fork_graph_state: bool,
            stream_mode: bool,
            use_last_thread: bool,
        ):
            try:
                if STREAM_MODE:
                    # In memory
                    db_path = PERSISTENT_CHECKPOINT_DB_PATH if not USE_MEMORY else ":memory:"
                    # conn = sqlite3.connect(db_path, check_same_thread=False)
                    # memory = MemorySaver()
                    checkpointer: AsyncSqliteSaver | SqliteSaver
                    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
                        # ONLY USE Interrupt_before when using streaming so that we can restart the graph from the checkpoint using None as input trick.
                        cv_curator_graph, configure_graph, interruption_handler, compiler = create_CV_curation_graph(
                            checkpointer=checkpointer)
                        graph = compiler(cv_curator_graph, checkpointer)
                        graph_config: RunnableConfig = {"configurable": {}}
                        thread: Thread | dict[str, Any] = {
                            "thread_id": thread_id}
                        graph_config = {"configurable": thread}

                        if USE_CHECKPOINTS_MEMORY and checkpointer:
                            if checkpoint_id:
                                graph_config = {
                                    "configurable": {
                                        "thread_id": thread_id,
                                        "checkpoint_ns": "",
                                        "checkpoint_id": checkpoint_id}}
                                previous_state, checkpoint = await helpers.get_state_history_checkpoint_with_id(checkpointer, graph, graph_config, checkpoint_id)
                                if checkpoint:
                                    # graph_config = {
                                    #     'configurable': checkpoint.config['configurable']}
                                    graph_config = checkpoint.config
                                    print(f"Using checkpoint with timestamp {
                                          checkpoint.checkpoint['ts']} and graph_config: {graph_config}")
                                else:
                                    raise ValueError(
                                        f"Checkpoint with id {checkpoint_id} not found")
                            else:
                                if fork_graph_state:
                                    to_fork, last_state = await helpers.get_last_graph_state(
                                        checkpointer, graph, graph_config)
                                    if not to_fork:
                                        raise ValueError(
                                            f"No checkpoint found for graph_config: {graph_config}")
                                    if not last_state:
                                        raise ValueError(
                                            f"No state found for graph_config: {graph_config}")
                                    current_state = to_fork.checkpoint
                                    input(
                                        "Please add any changes to state to fork_state.json and press Enter to continue...")
                                    with open("fork_state.json", "r") as f:
                                        new_state = json.load(f)

                                    if input(f"You current have the following updates to state:\n{new_state}\n\nPress Enter to continue or q to quit...\n").lower() == "q":
                                        exit(0)
                                    updated_state = {
                                        **current_state, **new_state}
                                    graph_config = await helpers.fork_graph(
                                        checkpointer, graph, to_fork.config, updated_state)
                                else:
                                    try:
                                        new_graph_config: RunnableConfig = {}
                                        if isinstance(checkpointer, AsyncSqliteSaver):
                                            # type: ignore
                                            existing_state = await checkpointer.aget(graph_config)
                                            if existing_state:
                                                new_graph_config = {
                                                    "configurable": {
                                                        "thread_id": thread['thread_id'],
                                                        "checkpoint_ns": "",
                                                        "checkpoint_id": existing_state.get('id', ""),

                                                    }
                                                }
                                            else:
                                                new_graph_config = graph_config
                                        elif isinstance(checkpointer, SqliteSaver):
                                            existing_state = checkpointer.get(
                                                graph_config)
                                            if existing_state:
                                                new_graph_config = {
                                                    "configurable": {
                                                        "thread_id": thread['thread_id'],
                                                        "checkpoint_ns": "",
                                                        "checkpoint_id": existing_state.get('id', ""),

                                                    }
                                                }
                                            else:
                                                new_graph_config = graph_config
                                        if use_last_thread:
                                            graph_config = new_graph_config or graph_config
                                            logging.info(f"Using memory checkpoint with thread_id: {
                                                thread['thread_id']} and graph_state: {existing_state}")
                                        else:
                                            logging.info(f"Using new graph state with graph_config: {
                                                graph_config}")
                                    except Exception as e:
                                        logging.error(
                                            f"Error getting state, consider that the graph_state may be empty: {e}")

                        graph.debug = True

                        async def stream_graph(runnable: Callable[[], AsyncIterator[dict[str, Any] | Any]] | None = None):
                            stream_mode: Literal["values", "updates", "debug", "messages",
                                                 "custom"] | StreamMode | list[StreamMode] | None = "values"
                            # BUG: Exception has occurred: TypeError
                            # Object of type FAISS is not serializable
                            async for event in (runnable() if runnable else graph.astream(
                                input=None if graph_config.get('configurable', {}).get(
                                    'checkpoint_id') else graph_args,
                                config=graph_config,
                                stream_mode=stream_mode,
                            )):
                                # BUG: the issue is that our sub graphs dont have checkpoints and we therefore have to start any subgraph that we were half way through again as the subgraph will not have been saved.
                                if stream_mode == "values":
                                    pprint(event)
                                elif stream_mode == 'messages':
                                    event['messages'][-1].pretty_print()
                                elif stream_mode == 'updates':
                                    pprint(event)
                                elif stream_mode == 'debug':
                                    pprint(event)
                                elif stream_mode == 'custom':
                                    pprint(event)

                        # Replace this with the URL of your own deployed graph
                        if USE_LANGGRAPH_STUDIO:
                            URL = "http://localhost:56091"
                            from langgraph_sdk import get_client
                            client = get_client(url=URL)

                            # Search all hosted graphs
                            assistants = await client.assistants.search()
                            # We create a thread for tracking the state of our run
                            client_thread: Thread = await client.threads.create()
                            # input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

                            # async def stream_graph():
                            #     async for chunk in client.runs.stream(
                            #         thread.__dict__['thread_id'],
                            #         "agent",
                            #         input=graph_args,
                            #         config=graph_config,
                            #         stream_mode="values",
                            #     ):
                            #         if chunk.data and chunk.event != "metadata":
                            #             print(chunk.data['messages'][-1])
                            async def stream_client_runs_runnable() -> AsyncIterator[dict[str, Any] | Any]:
                                return client.runs.stream(
                                    client_thread.__dict__['thread_id'],
                                    "agent",
                                    input=graph_args,
                                    config=graph_config,
                                    stream_mode="values",
                                )
                            await stream_graph(runnable=stream_client_runs_runnable)

                            suggest_cv_changes_input = input(
                                "Check the graph state at suggest_cv_changes. would you like to continue? y/n\n")
                            await view_graph_state_at_interrupt(checkpointer, graph, client_thread)
                            if suggest_cv_changes_input.lower() == "y":
                                await stream_graph()
                            else:
                                print("Exiting...")
                                return
                            human_interrupt_input = input(
                                "Check the graph state at interrupt. Press Enter to continue.\n")
                            if human_interrupt_input == "":
                                await view_graph_state_at_interrupt(checkpointer, graph, client_thread)
                                await stream_graph()
                            else:
                                print("Exiting...")
                                return

                        else:

                            await stream_graph()

                            await view_graph_state_at_interrupt(checkpointer, graph, graph_config)
                            suggest_cv_changes_input = input(
                                "Check the graph state at suggest_cv_changes. would you like to continue? y/n\n")
                            if suggest_cv_changes_input.lower() == "y":
                                await stream_graph()
                            else:
                                print("Exiting...")
                                return

                            await view_graph_state_at_interrupt(checkpointer, graph, graph_config)
                            human_interrupt_input = input(
                                "Check the graph state at interrupt. Press Enter to continue.\n")
                            if human_interrupt_input == "":
                                await stream_graph()
                            else:
                                print("Exiting...")
                                return

                else:
                    graph = cv_curator_graph.compile(
                        checkpointer=checkpointer
                    )
                    if checkpoint_id:
                        graph_config = {"configurable": {
                            "thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": checkpoint_id}}
                    graph.debug = True
                    final_state = await graph.ainvoke(
                        input=graph_args,
                        config=graph_config,
                    )
                    pprint(final_state)
                    return final_state

            except Exception as e:
                logger.error(f"Error running workflow at step {graph}: {e}")
                raise e

        # await run_graph2(
        #     fork_graph_state=fork_graph_state,
        #     stream_mode=stream_mode,
        #     use_last_thread=use_last_thread,
        # )
        await run_graph()

    # asyncio.run(get_graph_state())
    # asyncio.run(main(checkpoint_id='1efd64f0-0663-6e12-8004-7029ab79c95a'))
    asyncio.run(main(
        checkpoint_id='',
        thread_id=str(uuid.uuid4()),
        # replace state at checkpoint_id with new state in fork_state.json
        fork_graph_state=False,
        stream_mode=True,  # stream the graph
        use_last_thread=True,  # use the existing state at checkpoint_id
    ))
