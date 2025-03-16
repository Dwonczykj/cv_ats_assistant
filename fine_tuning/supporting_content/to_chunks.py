from cv_ats_assistant.case_a import chunk_html_content_by_tagname
import os
import json


def get_source_file_names() -> dict[str, list[str]]:
    """
    Get the source file names and their corresponding file paths as dict of extension to file path.
    """
    def group_by_extension(file_paths: list[str]) -> dict[str, list[str]]:
        extensions = set(os.path.splitext(fp)[1] for fp in file_paths)
        return {extension: [fp for fp in file_paths if os.path.splitext(fp)[1] == extension] for extension in extensions}
    return group_by_extension(os.listdir("cv_ats_assistant/fine_tuning/supporting_content/page_source"))


def to_chunks(html_content: str) -> list[str]:
    source_file_names = get_source_file_names()
    if "html" in source_file_names:
        for fp in source_file_names["html"]:
            with open(fp, "r") as f:
                html_content = f.read()
            chunks = chunk_html_content_by_tagname(html_content, "body")
            for chunk in chunks:
                yield chunk
