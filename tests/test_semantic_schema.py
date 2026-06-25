from llm_review_analysis.agents.semantic_tagger import SemanticTagger, SemanticTaxonomy


def test_taxonomy_matches_paper_labels_without_low_effort():
    taxonomy = SemanticTaxonomy()
    assert "positive" in taxonomy.all_labels
    assert "negative" in taxonomy.all_labels
    assert "helpful" in taxonomy.all_labels
    assert "vague" in taxonomy.all_labels
    assert "contradictory" in taxonomy.all_labels
    assert "no justification" in taxonomy.all_labels
    assert "duplicate" in taxonomy.all_labels
    assert "potentially misleading" in taxonomy.all_labels
    assert "low-effort" not in taxonomy.all_labels


def test_semantic_tagger_can_assign_multiple_dimensions():
    tags = SemanticTagger().tag_text("Great product but bad delivery and not as advertised.")
    assert "positive" in tags
    assert "negative" in tags
    assert "contradictory" in tags
    assert "potentially misleading" in tags
