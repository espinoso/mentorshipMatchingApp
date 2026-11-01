"""Match-card HTML is built with unsafe_allow_html, so survey-derived text must be escaped."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ui_processing as u


def test_field_label_escapes_html():
    out = u._field_label("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_field_label_blank_for_missing():
    assert u._field_label(None) == ""
    assert u._field_label("N/A") == ""


def test_country_token_escapes_unknown_name():
    out = u._country_token("<b>Narnia</b>")
    assert "<b>" not in out and "Narnia" in out


def test_country_token_flag_for_known():
    assert u._country_token("Spain") == "🇪🇸"
