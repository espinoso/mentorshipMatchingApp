"""Jev matrix scorer: 0-100 rescaling, matrix assembly, partial/total failure, progress."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jev_scoring
from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from jev_scoring import JevError, score_matrix_jev


def _ee():
    return pd.DataFrame({MENTEE_COLUMNS['id']: [1, 2], MENTEE_COLUMNS['field']: ["Bio", "CS"]})


def _mo():
    return pd.DataFrame({MENTOR_COLUMNS['id']: [10, 11], MENTOR_COLUMNS['field']: ["Bio", "CS"]})


class _Answer:
    def __init__(self, score):
        self.score = score


class _Resp:
    def __init__(self, scores):
        self.scores = scores  # {question_name: _Answer}


class FakeClient:
    """Answers every requested question. `score` is a constant or a {dim: value} map (0..4);
    raises when fail_fn(state) is True. Records the questions asked and the states seen."""

    def __init__(self, score=4.0, fail_fn=None):
        self.score, self.fail_fn, self.states, self.questions_seen = score, fail_fn, [], []

    def system_one(self, state, questions, **kwargs):
        self.states.append(state)
        self.questions_seen.append(sorted(questions))
        if self.fail_fn and self.fail_fn(state):
            raise RuntimeError("jev down")
        def val(dim):
            return self.score[dim] if isinstance(self.score, dict) else self.score
        return _Resp({d: _Answer(val(d)) for d in questions})

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(jev_scoring.time, "sleep", lambda s: None)


def test_profile_includes_goals_needs_and_seniority():
    """Jev must see mentee career goals + guidance needs and mentor seniority, not just field."""
    ee = pd.DataFrame({
        MENTEE_COLUMNS['id']: [1],
        MENTEE_COLUMNS['field']: ["Bio"],
        MENTEE_COLUMNS['career_goals']: ["move into industry R&D"],
        MENTEE_COLUMNS['guidance_areas']: ["networking"],
    })
    mo = pd.DataFrame({
        MENTOR_COLUMNS['id']: [10],
        MENTOR_COLUMNS['field']: ["Bio"],
        MENTOR_COLUMNS['experience_years']: ["15 years"],
        MENTOR_COLUMNS['current_position']: ["Professor"],
    })
    client = FakeClient()
    score_matrix_jev(ee, mo, "k", client=client)
    state = client.states[0]
    assert "move into industry R&D" in state   # mentee career goal reached Jev
    assert "networking" in state                # mentee guidance need reached Jev
    assert "15 years" in state                  # mentor seniority reached Jev
    assert "Professor" in state                 # mentor position reached Jev


def test_assembles_full_matrix_and_rescales_top():
    m, unscored = score_matrix_jev(_ee(), _mo(), "k", client=FakeClient(score=4.0))
    assert sorted(m.index) == ["10", "11"]
    assert sorted(m.columns) == ["1", "2"]
    assert (m == 100).all().all()  # score 4 of top 4 -> 100
    assert unscored == 0


@pytest.mark.parametrize("score,pct", [(2.0, 50), (0.83, 21), (0.0, 0)])
def test_rescale(score, pct):
    m, _ = score_matrix_jev(_ee(), _mo(), "k", client=FakeClient(score=score))
    assert (m == pct).all().all()


def test_partial_failure_filled_zero_not_raised():
    # Fail every pair whose MENTOR half is CS (mentor 11); the rest still score.
    def fail(s):
        return "CS" in s.split("MENTOR:")[-1]
    m, unscored = score_matrix_jev(_ee(), _mo(), "k", client=FakeClient(score=4.0, fail_fn=fail))
    assert m.loc["11", "1"] == 0 and m.loc["11", "2"] == 0  # failed mentor -> 0
    assert m.loc["10", "1"] == 100 and m.loc["10", "2"] == 100  # others scored
    assert unscored == 2  # the two failed pairs are reported, not hidden


def test_total_failure_raises():
    with pytest.raises(JevError):
        score_matrix_jev(_ee(), _mo(), "k", client=FakeClient(fail_fn=lambda s: True))


def test_progress_cb_reaches_total():
    seen = []
    score_matrix_jev(_ee(), _mo(), "k", client=FakeClient(), progress_cb=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (4, 4)  # 2x2 pairs, final tick is done==total


def _rich_ee():
    return pd.DataFrame({
        MENTEE_COLUMNS['id']: [1], MENTEE_COLUMNS['field']: ["Biology"],
        MENTEE_COLUMNS['specialization']: ["Genetics"], MENTEE_COLUMNS['education_level']: ["Master's"],
        MENTEE_COLUMNS['guidance_areas']: ["career paths"], MENTEE_COLUMNS['career_goals']: ["industry"],
    })


def _rich_mo():
    return pd.DataFrame({
        MENTOR_COLUMNS['id']: [10], MENTOR_COLUMNS['field']: ["Biology"],
        MENTOR_COLUMNS['specialization']: ["Genetics"], MENTOR_COLUMNS['education_level']: ["PhD"],
        MENTOR_COLUMNS['experience_years']: ["10"], MENTOR_COLUMNS['current_position']: ["Professor"],
        MENTOR_COLUMNS['career_dev_experience']: ["mentoring students"],
    })


def test_weighted_blend_over_active_dimensions():
    # This data activates 4 dims (availability/style have no columns -> dropped).
    scores = {"field_specialization": 4, "goals_mentoring": 2, "skills": 0, "level_seniority": 4}
    client = FakeClient(score=scores)
    m, _ = score_matrix_jev(_rich_ee(), _rich_mo(), "k", client=client)
    assert client.questions_seen[0] == sorted(scores)  # only dims with data are asked
    # defaults 30/20/20/15 over the 4 active dims, renormalized /85:
    # (100*30 + 50*20 + 0*20 + 100*15)/85 = 64.7 -> 65
    assert m.loc["10", "1"] == 65


def test_weights_change_the_blend():
    scores = {"field_specialization": 4, "goals_mentoring": 2, "skills": 0, "level_seniority": 4}
    # put all weight on skills (scored 0) -> blended 0, proving the sliders actually drive the score
    m, _ = score_matrix_jev(_rich_ee(), _rich_mo(), "k",
                            client=FakeClient(score=scores), weights={"skills": 100})
    assert m.loc["10", "1"] == 0
