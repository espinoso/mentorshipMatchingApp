"""Shared synthetic fixtures. All names, IDs and values are invented; no real applicant data."""

import pandas as pd
import pytest

from info import MENTEE_COLUMNS, MENTOR_COLUMNS

MENTOR_IDS = ["1", "2", "3"]
MENTEE_IDS = ["101", "102", "103", "104"]


@pytest.fixture
def matrix_df():
    """Match scores (0-100): mentors as rows, mentees as columns."""
    data = [
        [90.0, 40.0, 75.0, 60.0],
        [35.0, 85.0, 50.0, 70.0],
        [60.0, 55.0, 95.0, 30.0],
    ]
    return pd.DataFrame(data, index=MENTOR_IDS, columns=MENTEE_IDS)


@pytest.fixture
def mentees_df():
    rows = [
        ("101", "Institution A", "Country A", "Bachelor", "Program A", "Field A",
         "Specialization A", "English", "Guidance A", "Goal A", "Note A"),
        ("102", "Institution B", "Country B", "Master", "Program B", "Field B",
         "Specialization B", "French", "Guidance B", "Goal B", "Note B"),
        ("103", "Institution C", "Country A", "Bachelor", "Program C", "Field A",
         "Specialization C", "English", "Guidance C", "Goal C", "Note C"),
        ("104", "Institution D", "Country C", "Master", "Program D", "Field C",
         "Specialization D", "Spanish", "Guidance D", "Goal D", "Note D"),
    ]
    columns = [
        MENTEE_COLUMNS[k]
        for k in ("id", "affiliation", "country", "education_level", "current_program",
                  "field", "specialization", "languages", "guidance_areas", "career_goals",
                  "other_info")
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def mentors_df():
    rows = [
        ("1", "Organization A", "Country A", "Professor", "University", "Country B",
         "University X", "PhD", "Field A", "Specialization A", "10", "Yes"),
        ("2", "Organization B", "Country B", "Engineer", "Company", "Country C",
         "University Y", "Master", "Field B", "Specialization B", "7", "No"),
        ("3", "Organization C", "Country C", "Researcher", "Research Institute", "Country A",
         "University Z", "PhD", "Field C", "Specialization C", "15", "Yes"),
    ]
    columns = [
        MENTOR_COLUMNS[k]
        for k in ("id", "affiliation", "country", "current_position", "institution_type",
                  "country_of_origin", "alma_mater", "education_level", "field",
                  "specialization", "experience_years", "career_dev_experience")
    ]
    return pd.DataFrame(rows, columns=columns)
