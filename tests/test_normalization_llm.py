"""Normalization LLM calls: Claude plain-text transport, output shape, AICache reuse."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Normalization"))

import merge_mentee_skills_enhanced as mentee
import merge_mentor_skills_enhanced as mentor
import normalization
import shared_utils
from llm_client import LLMError
from shared_utils import AICache


class FakeClaude:
    """Stands in for ClaudeClient: returns canned text and records calls."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete_text(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return f"  {self.reply}\n"


def test_education_level_shape_and_budget():
    c = FakeClaude("Masters")
    assert mentee.normalize_education_level("MSc", c, AICache(), 1) == "Masters"
    assert c.calls[0][2] == 10


def test_program_split_parses_json_text():
    c = FakeClaude('{"level": "PhD", "specialization": "Computer Science"}')
    assert mentee.split_current_education_program("PhD CS", c, AICache(), 1) == (
        "PhD",
        "Computer Science",
    )
    assert c.calls[0][2] == 100


def test_broad_field_strips_punctuation():
    c = FakeClaude('"Computer Science".')
    assert mentee.determine_broad_field_of_expertise("cs", "ml", c, AICache(), 1) == "Computer Science"
    assert c.calls[0][2] == 20


def test_mentee_skill_merge_is_one_line_keyword_list():
    c = FakeClaude("biomedicine,\npcr, python")
    out = mentee.merge_and_normalize_skills("Bio", "x", "y", "pcr", c, AICache(), 1)
    assert out == "biomedicine, pcr, python"
    assert c.calls[0][2] == 500


def test_mentor_education_and_field():
    c = FakeClaude('{"level": "Masters", "field": "Law"}')
    assert mentor.extract_education_level_and_field("LLM in Law", c, AICache(), 1) == ("Masters", "Law")
    assert c.calls[0][2] == 150


def test_mentor_skill_merge():
    c = FakeClaude("law, contracts")
    out = mentor.merge_and_normalize_skills("Law", "x", "y", "contracts", c, AICache(), 1)
    assert out == "law, contracts"
    assert c.calls[0][2] == 500


def test_mentor_field_classification():
    c = FakeClaude("Quantum Optics")
    cat, terms = normalization.normalize_mentor_field_of_expertise(
        "zzqq quasar widgets", client=c, cache=AICache(), row_num=1
    )
    assert cat == "Quantum Optics"
    assert c.calls[0][2] == 20
    assert "quasar" in terms


def test_cache_prevents_second_call():
    c, cache = FakeClaude("PhD"), AICache()
    assert mentee.normalize_education_level("Doctorate", c, cache, 1) == "PhD"
    assert mentee.normalize_education_level("Doctorate", c, cache, 2) == "PhD"
    assert len(c.calls) == 1


def test_hard_failure_falls_back_without_raising(monkeypatch):
    monkeypatch.setattr(mentee.time, "sleep", lambda s: None)

    class Down:
        def complete_text(self, *a, **k):
            raise LLMError("unreachable")

    assert mentee.normalize_education_level("MSc", Down(), AICache(), 1) == "MSc"
    assert mentee.merge_and_normalize_skills("Bio", "", "", "PCR", Down(), AICache(), 1) == "bio, pcr"


def test_setup_claude_uses_haiku_and_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = shared_utils.setup_claude()
    assert client.model == "claude-haiku-5-5"
    assert client.api_key == "k"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        shared_utils.setup_claude()


def test_program_split_accepts_fenced_json():
    c = FakeClaude('```json\n{"level": "Masters", "specialization": "Data Science"}\n```')
    assert mentee.split_current_education_program("MSc DS", c, AICache(), 1) == (
        "Masters",
        "Data Science",
    )
    assert len(c.calls) == 1  # parsed first try, no retry


def test_mentor_education_accepts_fenced_json():
    c = FakeClaude('```json\n{"level": "PhD", "field": "Physics"}\n```')
    assert mentor.extract_education_level_and_field("PhD Physics", c, AICache(), 1) == (
        "PhD",
        "Physics",
    )
    assert len(c.calls) == 1
