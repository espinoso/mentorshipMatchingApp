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


@pytest.mark.parametrize("text,expected", [
    ("Ph.D. in Computer Science", "PhD"),
    ("Doctorado en Física", "PhD"),
    ("Máster en Datos", "Masters"),
    ("MSc Biology", "Masters"),
    ("Maestría en Educación", "Masters"),
    ("Posgrado en Derecho", "Masters"),  # must not hit "grado" -> Bachelors
    ("Licenciatura en Derecho", "Bachelors"),
    ("Grado en Ingeniería", "Bachelors"),
    ("BSc and Master's", "Masters"),  # highest level wins
    ("Bachillerato", "High School"),
    ("something else", "N/A"),
    ("", "N/A"),
    (None, "N/A"),
    (float("nan"), "N/A"),
])
def test_classify_education_level(text, expected):
    assert shared_utils.classify_education_level(text) == expected


def test_extract_program_specialization():
    f = shared_utils.extract_program_specialization
    assert f("PhD in Computer Science") == "Computer Science"
    assert f("Máster en Ciencia de Datos") == "Ciencia de Datos"
    assert f("Computer Science") == "Computer Science"  # level-free text is the specialization
    assert f("PhD") == ""
    assert f(None) == ""


def test_merge_skill_sources_splits_dedupes_and_drops_junk():
    out = shared_utils.merge_skill_sources(
        "Biology; PCR", "N/A", None, "pcr, Python/R\nERROR: no field", float("nan"), " biology "
    )
    assert out == "biology, pcr, python, r"
    assert shared_utils.merge_skill_sources("", "N/A", None) == "N/A"


def test_unrecognized_level_falls_back_to_llm():
    # A present-but-unusual level the keyword list doesn't know -> LLM fallback classifies it.
    c, cache = FakeClaude("PhD"), AICache()
    assert shared_utils.classify_level_llm("quux doctoral fellowship", c, cache) == "PhD"
    assert len(c.calls) == 1
    # Blank input never calls the model.
    assert shared_utils.classify_level_llm("", c, cache) == "N/A"
    assert shared_utils.classify_level_llm(float("nan"), c, cache) == "N/A"
    assert len(c.calls) == 1
    # An invalid model reply degrades to N/A, not garbage.
    bad = FakeClaude("not a level")
    assert shared_utils.classify_level_llm("mystery credential", bad, AICache()) == "N/A"


def test_mentee_field_single_call_shape_and_budget():
    c = FakeClaude('"Computer Science".')
    assert mentee.classify_mentee_field("cs", "PhD ML", c, AICache(), 1) == "Computer Science"
    assert len(c.calls) == 1
    assert c.calls[0][2] == 20
    assert "PhD ML" in c.calls[0][1]  # program is given as context


def test_mentee_field_empty_makes_no_call():
    c = FakeClaude("x")
    assert mentee.classify_mentee_field(float("nan"), "N/A", c, AICache(), 1) == "N/A"
    assert c.calls == []


def test_mentor_field_classification():
    c = FakeClaude("Quantum Optics")
    cat, terms = normalization.normalize_mentor_field_of_expertise(
        "zzqq quasar widgets", client=c, cache=AICache(), row_num=1
    )
    assert cat == "Quantum Optics"
    assert c.calls[0][2] == 20
    assert "quasar" in terms


def test_cache_prevents_second_call():
    c, cache = FakeClaude("Computer Science"), AICache()
    assert mentee.classify_mentee_field("cs", "PhD", c, cache, 1) == "Computer Science"
    assert mentee.classify_mentee_field("cs", "PhD", c, cache, 2) == "Computer Science"
    assert len(c.calls) == 1


def test_hard_failure_falls_back_to_original_field(monkeypatch):
    monkeypatch.setattr(mentee.time, "sleep", lambda s: None)

    class Down:
        def complete_text(self, *a, **k):
            raise LLMError("unreachable")

    assert mentee.classify_mentee_field("Bio", "PhD", Down(), AICache(), 1) == "Bio"
    assert mentee.classify_mentee_field("", "PhD", Down(), AICache(), 1) == "N/A"


def test_empty_response_retries_then_falls_back(monkeypatch):
    monkeypatch.setattr(mentee.time, "sleep", lambda s: None)
    c = FakeClaude("")
    assert mentee.classify_mentee_field("Bio", "", c, AICache(), 1) == "Bio"
    assert len(c.calls) == mentee.MAX_RETRIES


def test_setup_claude_uses_haiku_and_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = shared_utils.setup_claude()
    assert client.model == "claude-haiku-5-5"
    assert client.api_key == "k"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        shared_utils.setup_claude()


def test_process_mentee_file_one_field_call_and_progress_pass_per_row(tmp_path, capsys):
    import pandas as pd

    path = tmp_path / "mentees.xlsx"
    pd.DataFrame({
        "Highest educational level completed": ["MSc", "Licenciatura"],
        "Current education program": ["PhD in Machine Learning", "PhD"],
        "Field of expertise": ["cs", "law"],
        "Specialization": ["ml; NLP", "contracts"],
        "Specific hard skills": ["python", "writing, Law"],
    }).to_excel(path, index=False)
    c = FakeClaude("Computer Science")
    df, _, _ = mentee.process_mentee_file(str(path), dry_run=True, client=c, cache=AICache())
    assert len(c.calls) == 2  # all levels keyword-recognized -> one FIELD call per row, no level LLM
    assert df["Mentee Highest educational level completed"].tolist() == ["Masters", "Bachelors"]
    assert df["Mentee Current education program"].tolist() == ["PhD", "PhD"]
    assert df["Mentee Field of expertise"].tolist() == ["Computer Science"] * 2
    assert df["Mentee Specialization"].tolist() == [
        "cs, machine learning, ml, nlp, python",
        "law, contracts, writing",
    ]
    out = capsys.readouterr().out
    assert out.count("[1/2]") == 1 and out.count("[2/2]") == 1  # a single progress pass


def test_process_mentor_file_llm_only_for_unmapped_field(tmp_path, capsys):
    import pandas as pd

    path = tmp_path / "mentors.xlsx"
    pd.DataFrame({
        "Type of institution": ["University", "University"],
        "Highest educational level completed": ["Master in Law", "PhD"],
        "Field of expertise": ["Law", "zzqq quasar widgets"],
        "Specialization": ["contracts", "optics"],
        "Specific hard skills": ["drafting", "lasers"],
    }).to_excel(path, index=False)
    c = FakeClaude("Quantum Optics")
    result = mentor.process_mentor_file(str(path), dry_run=True, client=c, cache=AICache())
    df, unmapped = result[0], result[-1]
    assert len(c.calls) == 1  # only the unmapped field hits the model
    assert df["Mentor Field of expertise"].tolist()[1] == "Quantum Optics"
    assert df["Mentor Highest Educational Level"].tolist() == ["Masters", "PhD"]
    assert "law" in df["Mentor Specialization"].tolist()[0]
    assert df["Mentor Specialization"].tolist()[0].endswith("contracts, drafting")
    assert len(unmapped) == 1 and "AI classified" in unmapped[0]
    out = capsys.readouterr().out
    assert out.count("[1/2]") == 1 and out.count("[2/2]") == 1
