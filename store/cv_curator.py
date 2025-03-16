from cv_ats_assistant.case_a import create_CV_curation_graph


cv_curator_graph = create_CV_curation_graph(checkpointer=None)
graph = cv_curator_graph.compile()
