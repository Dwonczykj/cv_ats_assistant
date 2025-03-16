#! /bin/bash

if [ -f "cv_ats_assistant_venv/bin/activate" ]; then
    source cv_ats_assistant_venv/bin/activate
fi

# assert that which pip contains cv_ats_assistant_venv/bin/ in the path
if ! which pip | grep -q "cv_ats_assistant_venv/bin/"; then
    echo "Error: pip does not contain cv_ats_assistant_venv/bin/ in the path"
    exit 1
fi

python -m spacy download en_core_web_lg
