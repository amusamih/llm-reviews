from __future__ import annotations

from pathlib import Path

from llm_review_analysis.agents.topic_assignment_agent import TopicAssignmentAgent, load_topic_vocabulary
from llm_review_analysis.llm import LLMResponse


VOCABULARY_PATH = Path("evaluation/configs/product_generic_topics.json")


def test_product_generic_topic_vocabulary_has_expected_public_shape():
    labels = load_topic_vocabulary(VOCABULARY_PATH)

    assert len(labels) == 14
    assert len(set(labels)) == 14
    assert "other" not in {label.lower() for label in labels}
    assert "delivery and packaging" in labels
    assert "overall satisfaction" in labels


def test_product_generic_topic_vocabulary_loads_into_topic_assignment_agent():
    labels = load_topic_vocabulary(VOCABULARY_PATH)
    provider = OneTopicProvider('{"topic": "power battery and charging"}')
    agent = TopicAssignmentAgent(provider)

    topic = agent.assign_topic("The charger overheats and the battery drains quickly.", labels)

    assert topic == "power battery and charging"
    assert provider.calls == 1


class OneTopicProvider:
    model = "unit-provider"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        self.calls += 1
        assert purpose == "topic_assign"
        assert response_format == "json"
        assert "power battery and charging" in prompt
        return LLMResponse(content=self.response, model=self.model)
