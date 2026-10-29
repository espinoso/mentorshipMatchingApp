from matching_core import MatrixParseReport, parse_matrix_response


def test_parses_clean_matrix():
    txt = (
        '{"matrix":[{"mentor_id":"1","scores":'
        '[{"mentee_id":"1","percentage":80},{"mentee_id":"2","percentage":40}]}]}'
    )
    r = parse_matrix_response(txt, ["1"], ["1", "2"])
    assert isinstance(r, MatrixParseReport)
    assert r.error is None
    assert list(r.df.columns) == ["1", "2"]
    assert r.df.loc["1", "1"] == 80
    assert r.completion_rate == 100.0


def test_strips_markdown_fences():
    txt = '```json\n{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percentage":90}]}]}\n```'
    r = parse_matrix_response(txt, ["1"], ["1"])
    assert r.df.loc["1", "1"] == 90


def test_missing_mentee_filled_with_30_and_reported():
    txt = '{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percentage":70}]}]}'
    r = parse_matrix_response(txt, ["1"], ["1", "2"])
    assert r.df.loc["1", "2"] == 30
    assert "2" in r.missing_mentees_filled
    assert r.completion_rate == 50.0


def test_wrong_mentor_id_rejected():
    txt = '{"matrix":[{"mentor_id":"99","scores":[{"mentee_id":"1","percentage":70}]}]}'
    r = parse_matrix_response(txt, ["1"], ["1"])
    assert "99" in r.wrong_mentors
    assert r.df is None


def test_truncated_json_returns_error():
    txt = '{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percent'
    r = parse_matrix_response(txt, ["1"], ["1"])
    assert r.df is None
    assert r.error is not None


def test_wrong_mentee_and_missing_mentor_reported():
    txt = (
        '{"matrix":[{"mentor_id":"1","scores":'
        '[{"mentee_id":"1","percentage":70},{"mentee_id":"zz","percentage":10}]}]}'
    )
    r = parse_matrix_response(txt, ["1", "2"], ["1"])
    assert r.wrong_mentees == ["zz"]
    assert r.missing_mentors == ["2"]
    assert list(r.df.index) == ["1"]


def test_text_around_json_is_extracted():
    txt = 'Sure!\n{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percentage":55}]}]}\nDone.'
    r = parse_matrix_response(txt, ["1"], ["1"])
    assert r.error is None
    assert r.df.loc["1", "1"] == 55


def test_integer_ids_stay_strings_and_fill_works():
    txt = '{"matrix":[{"mentor_id":1,"scores":[{"mentee_id":1,"percentage":70}]}]}'
    r = parse_matrix_response(txt, [1], [1, 2])
    assert list(r.df.columns) == ["1", "2"]
    assert r.df.loc["1", "2"] == 30


def test_reasoning_model_config_repr_is_an_error():
    r = parse_matrix_response("ResponseTextConfig(format=ResponseFormatText(type='text'))", ["1"], ["1"])
    assert r.df is None
    assert r.error is not None
