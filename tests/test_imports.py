def test_imports():
    import app.server
    import evaluation.dataset_stats
    import llm_review_analysis
    import llm_review_analysis.agents
    import llm_review_analysis.analytics
    import llm_review_analysis.db

    assert llm_review_analysis.__version__
    assert callable(app.server.create_app)
