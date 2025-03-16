

from typing import Optional
from cv_ats_assistant.models import JobPost


def merge_job_info(existing: Optional[JobPost], new_info: JobPost) -> JobPost:
    """Merge new job information with existing information."""
    if not existing:
        return new_info

    merged_dict = existing.model_dump()
    new_dict = new_info.model_dump()

    # Merge logic for each field
    for field, value in new_dict.items():
        if value is not None and (field not in merged_dict or merged_dict[field] is None):
            merged_dict[field] = value
        elif field == 'description' and value:
            # Concatenate descriptions
            existing_desc = merged_dict.get('description', '')
            if existing_desc and value not in existing_desc:
                merged_dict['description'] = f"{existing_desc}\n{value}"

    return JobPost.model_validate(merged_dict)