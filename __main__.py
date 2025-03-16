import argparse
from pathlib import Path
from .case_a import create_CV_curation_graph
from .case_b import create_case_b_graph
from .models import GraphState


def main():
    # TODO: Add a --debug flag to print the graph state at each step
    # TODO: Add a --verbose flag to print the verbose output of the graph
    # TODO: Add a --checkpoint flag to save the graph state to a checkpoint file
    # TODO: Add an interactive cli so that any flags that are not set are prompted for

    parser = argparse.ArgumentParser(description="CV Analysis Tool")
    parser.add_argument("--case", choices=['a', 'b'], required=True)
    parser.add_argument("--job-url", required=True)
    parser.add_argument("--cv-path", required=True)
    parser.add_argument("--debug", choices=['True', 'False'], required=False)
    parser.add_argument("--verbose", choices=['True', 'False'], required=False)
    parser.add_argument(
        "--checkpoint", choices=['True', 'False'], required=False)

    args = parser.parse_args()
    if not args.case:
        args.case = input("Enter the case (a or b): ")
        if args.case not in ['a', 'b']:
            raise ValueError("Invalid case. Please enter 'a' or 'b'.")
    if not args.job_url:
        args.job_url = input("Enter the job url: ")
        if not args.job_url.startswith("http"):
            raise ValueError(
                "Invalid job url. Please enter a valid url starting with http or https.")

    if not args.cv_path:
        args.cv_path = input("Enter the cv path: ")
        if not args.cv_path:
            args.cv_path = "/Users/joey/Documents/CVs/Joey Dwonczyk CV.pdf"
        elif not Path(args.cv_path).exists():
            raise ValueError(
                "Invalid cv path. Please enter a valid path to a file.")

    # TODO: Set the following flags using a checkbox list in the cli
    if not args.debug:
        args.debug = input("Enter the debug flag (True or False): ")
    if not args.verbose:
        args.verbose = input("Enter the verbose flag (True or False): ")
    if not args.checkpoint:
        args.checkpoint = input("Enter the checkpoint flag (True or False): ")

    initial_state = GraphState(
        job_post_url=args.job_url,
        cv_path=args.cv_path,
        cv_embedding=None,
        cv_scores=None,
        keyword_analysis=None,
        altered_cv_path=None,
        suggested_changes=None,
        user_approved_changes=None
    )

    if args.case == 'a':
        graph = create_CV_curation_graph()
    else:
        graph = create_case_b_graph()

    result = graph.ainvoke(initial_state)

    # Print results based on case
    if args.case == 'a':
        print(f"CV Scores: {result.cv_scores}")
    else:
        print(f"Keyword Analysis: {result.keyword_analysis}")


if __name__ == "__main__":
    main()
