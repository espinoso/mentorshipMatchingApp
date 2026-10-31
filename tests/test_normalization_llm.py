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


MENTEE_JSON = (
    '{"education_level": "Masters", "program_level": "PhD", "field_of_expertise": "Computer Science", '
    '"specialization_extracted": "Machine Learning", "skills": "computer science,\\nmachine learning, python"}'
)


def test_mentee_row_single_call_shape_and_budget():
    c = FakeClaude(MENTEE_JSON)
    out = mentee.normalize_mentee_row("MSc", "PhD ML", "cs", "ml", "python", c, AICache(), 1)
    assert out == {
        "education_level": "Masters",
        "program_level": "PhD",
        "field_of_expertise": "Computer Science",
        "skills": "computer science, machine learning, python",
    }
    assert len(c.calls) == 1
    assert c.calls[0][2] == 600


def test_mentee_row_all_empty_makes_no_call():
    c = FakeClaude(MENTEE_JSON)
    out = mentee.normalize_mentee_row(float("nan"), "N/A", "", None, " ", c, AICache(), 1)
    assert set(out.values()) == {"N/A"}
    assert c.calls == []


def test_mentee_row_accepts_fenced_json():
    c = FakeClaude("```json\n" + MENTEE_JSON + "\n```")
    out = mentee.normalize_mentee_row("MSc", "PhD ML", "cs", "ml", "python", c, AICache(), 1)
    assert out["program_level"] == "PhD"
    assert len(c.calls) == 1  # parsed first try, no retry


def test_mentee_row_unexpected_education_keeps_original():
    c = FakeClaude('{"education_level": "Licenciatura", "program_level": "N/A", "field_of_expertise": "Law", "skills": "law"}')
    out = mentee.normalize_mentee_row("Licenciatura", "", "Law", "", "", c, AICache(), 1)
    assert out["education_level"] == "Licenciatura"


MENTOR_JSON = (
    '{"education_level": "Masters", "education_field": "Law", "field_of_expertise": "Law", '
    '"skills": "law,\\ncontracts"}'
)


def test_mentor_row_single_call_shape_and_budget():
    c = FakeClaude(MENTOR_JSON)
    out = mentor.normalize_mentor_row("LLM in Law", "", "", "", "y", "contracts", c, AICache(), 1)
    assert out == {
        "education_level": "Masters",
        "education_field": "Law",
        "field_of_expertise": "Law",
        "skills": "law, contracts",
    }
    assert len(c.calls) == 1
    assert c.calls[0][2] == 600


def test_mentor_row_all_empty_makes_no_call():
    c = FakeClaude(MENTOR_JSON)
    out = mentor.normalize_mentor_row("N/A", "", "", "", None, float("nan"), c, AICache(), 1)
    assert set(out.values()) == {"N/A"}
    assert c.calls == []


def test_mentor_row_accepts_fenced_json():
    c = FakeClaude("```json\n" + MENTOR_JSON + "\n```")
    out = mentor.normalize_mentor_row("PhD Physics", "", "", "", "", "", c, AICache(), 1)
    assert out["education_level"] == "Masters"
    assert len(c.calls) == 1


def test_mentor_field_classification():
    c = FakeClaude("Quantum Optics")
    cat, terms = normalization.normalize_mentor_field_of_expertise(
        "zzqq quasar widgets", client=c, cache=AICache(), row_num=1
    )
    assert cat == "Quantum Optics"
    assert c.calls[0][2] == 20
    assert "quasar" in terms


def test_cache_prevents_second_call():
    c, cache = FakeClaude(MENTEE_JSON), AICache()
    first = mentee.normalize_mentee_row("MSc", "PhD ML", "cs", "ml", "python", c, cache, 1)
    second = mentee.normalize_mentee_row("MSc", "PhD ML", "cs", "ml", "python", c, cache, 2)
    assert first == second
    assert len(c.calls) == 1
    c2 = FakeClaude(MENTOR_JSON)
    mentor.normalize_mentor_row("LLM", "", "", "", "", "", c2, cache, 1)
    mentor.normalize_mentor_row("LLM", "", "", "", "", "", c2, cache, 2)
    assert len(c2.calls) == 1


def test_hard_failure_falls_back_without_raising(monkeypatch):
    monkeypatch.setattr(mentee.time, "sleep", lambda s: None)
    monkeypatch.setattr(mentor.time, "sleep", lambda s: None)

    class Down:
        def complete_text(self, *a, **k):
            raise LLMError("unreachable")

    out = mentee.normalize_mentee_row("MSc", "PhD ML", "Bio", "", "PCR", Down(), AICache(), 1)
    assert out == {
        "education_level": "MSc",
        "program_level": "PhD ML",
        "field_of_expertise": "Bio",
        "skills": "bio, pcr",
    }
    out = mentor.normalize_mentor_row("LLM", "", "Law", "", "x", "PCR", Down(), AICache(), 1)
    assert out["skills"] == "law, x, pcr"
    assert out["education_level"] == "N/A"


def test_unparseable_json_retries_then_falls_back(monkeypatch):
    monkeypatch.setattr(mentee.time, "sleep", lambda s: None)
    c = FakeClaude("not json at all")
    out = mentee.normalize_mentee_row("MSc", "", "Bio", "", "", c, AICache(), 1)
    assert len(c.calls) == mentee.MAX_RETRIES
    assert out["education_level"] == "MSc"


def test_setup_claude_uses_haiku_and_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = shared_utils.setup_claude()
    assert client.model == "claude-haiku-5-5"
    assert client.api_key == "k"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        shared_utils.setup_claude()


def test_process_mentee_file_one_call_and_progress_pass_per_row(tmp_path, capsys):
    import pandas as pd

    path = tmp_path / "mentees.xlsx"
    pd.DataFrame({
        "Highest educational level completed": ["MSc", "BSc"],
        "Current education program": ["PhD ML", "Law"],
        "Field of expertise": ["cs", "law"],
        "Specialization": ["ml", "contracts"],
        "Specific hard skills": ["python", "writing"],
    }).to_excel(path, index=False)
    c = FakeClaude(MENTEE_JSON)
    df, _, _ = mentee.process_mentee_file(str(path), dry_run=True, client=c, cache=AICache())
    assert len(c.calls) == 2
    assert df["Mentee Highest educational level completed"].tolist() == ["Masters", "Masters"]
    assert df["Mentee Current education program"].tolist() == ["PhD", "PhD"]
    assert df["Mentee Field of expertise"].tolist() == ["Computer Science"] * 2
    out = capsys.readouterr().out
    assert out.count("[1/2]") == 1 and out.count("[2/2]") == 1  # a single progress pass


def test_process_mentor_file_one_call_per_row_and_ai_field_only_when_unmapped(tmp_path, capsys):
    import pandas as pd

    path = tmp_path / "mentors.xlsx"
    pd.DataFrame({
        "Type of institution": ["University", "University"],
        "Highest educational level completed": ["LLM in Law", "PhD"],
        "Field of expertise": ["Law", "zzqq quasar widgets"],
        "Specialization": ["contracts", "optics"],
        "Specific hard skills": ["drafting", "lasers"],
    }).to_excel(path, index=False)
    c = FakeClaude('{"education_level": "Masters", "education_field": "Law", '
                   '"field_of_expertise": "Quantum Optics", "skills": "law, optics"}')
    result = mentor.process_mentor_file(str(path), dry_run=True, client=c, cache=AICache())
    df, unmapped = result[0], result[-1]
    assert len(c.calls) == 2
    assert df["Mentor Field of expertise"].tolist()[1] == "Quantum Optics"
    assert df["Mentor Field of expertise"].tolist()[0] != "Quantum Optics"  # dictionary-mapped, not AI
    assert df["Mentor Highest Educational Level"].tolist() == ["Masters", "Masters"]
    assert len(unmapped) == 1 and "AI classified" in unmapped[0]
    out = capsys.readouterr().out
    assert out.count("[1/2]") == 1 and out.count("[2/2]") == 1
