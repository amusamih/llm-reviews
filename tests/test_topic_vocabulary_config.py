from __future__ import annotations

from pathlib import Path

from llm_review_analysis.agents.topic_assignment_agent import (
    DEFAULT_TOPIC_DEFINITIONS,
    TopicAssignmentAgent,
    load_topic_vocabulary,
    load_topic_vocabulary_config,
)
from llm_review_analysis.llm import LLMResponse


VOCABULARY_PATH = Path("evaluation/configs/product_generic_topics.json")


def test_product_generic_topic_vocabulary_has_expected_public_shape():
    vocabulary = load_topic_vocabulary_config(VOCABULARY_PATH)
    labels = list(vocabulary.labels)

    assert len(labels) == 14
    assert len(set(labels)) == 14
    assert "other" not in {label.lower() for label in labels}
    assert labels == [
        "delivery and packaging",
        "product quality and durability",
        "price and value",
        "usability and setup",
        "performance and functionality",
        "design and appearance",
        "compatibility and fit",
        "comfort and ergonomics",
        "content and media quality",
        "taste scent and consumable experience",
        "power battery and charging",
        "customer support returns and warranty",
        "safety and reliability",
        "overall satisfaction",
    ]
    assert set(vocabulary.definitions) == set(labels)
    assert all(vocabulary.definitions[label].strip() for label in labels)
    assert vocabulary.name == "product_generic_topics"
    assert vocabulary.version == 1


def test_product_generic_topic_vocabulary_loads_labels_for_legacy_callers():
    assert load_topic_vocabulary(VOCABULARY_PATH) == list(load_topic_vocabulary_config(VOCABULARY_PATH).labels)


def test_product_generic_topic_vocabulary_definitions_reach_assignment_prompt():
    vocabulary = load_topic_vocabulary_config(VOCABULARY_PATH)
    provider = OneTopicProvider('{"topic": "power battery and charging"}')
    agent = TopicAssignmentAgent(provider)

    topic = agent.assign_topic("The charger overheats and the battery drains quickly.", vocabulary)

    assert topic == "power battery and charging"
    assert provider.calls == 1
    assert "Battery duration, charging speed, power use, power stability, or charging compatibility." in provider.prompts[0]
    assert "The review is primarily about power battery and charging" not in provider.prompts[0]


def test_existing_application_default_topic_definitions_remain_available():
    assert "battery life" in DEFAULT_TOPIC_DEFINITIONS
    assert "delivery and packaging" not in DEFAULT_TOPIC_DEFINITIONS


def test_legacy_list_based_topic_input_remains_supported():
    provider = OneTopicProvider('{"topic": "battery life"}')
    agent = TopicAssignmentAgent(provider)

    assert agent.assign_topic("The battery drains quickly.", ["battery life", "delivery"]) == "battery life"
    assert "Battery duration, charging endurance, or power drain during use." in provider.prompts[0]


class OneTopicProvider:
    model = "unit-provider"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        self.calls += 1
        self.prompts.append(prompt)
        assert purpose == "topic_assign"
        assert response_format == "json"
        return LLMResponse(content=self.response, model=self.model)
