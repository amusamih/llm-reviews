from pathlib import Path

from llm_review_analysis.agents import ReviewOrchestrator


def test_orchestrator_direct_sql_semantics_and_analytics(settings, provider, sample_db):
    conn, table = sample_db
    orchestrator = ReviewOrchestrator(settings, provider)

    direct = orchestrator.answer(conn, "How many reviews for sample product?", product_table=table)
    assert direct["type"] == "text"
    assert "2 reviews" in direct["message"]

    semantic = orchestrator.answer(conn, "Why are users unhappy about sample product?", product_table=table)
    assert semantic["type"] == "text"
    assert "Relevant review evidence" in semantic["message"]

    analytics = orchestrator.answer(conn, "Show the rating distribution for sample product", product_table=table)
    assert analytics["type"] == "chart"
    assert Path(analytics["path"]).exists()
