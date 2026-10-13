from xai_lab.console import _table, collect_checks


def test_table_alignment_and_width():
    lines = _table(["Clase", "Prob."], [["perro", "0.5"], ["gato", "0.25"]], "lr", indent=0)
    assert len({len(l) for l in lines}) == 1  # every row has the same width
    assert lines[0].startswith("Clase") and lines[2].endswith("0.5")


def test_checks_flag_failures():
    base = {"abs_error_logit": 1e-6, "max_abs_error_ZL_vs_M": 1e-6, "max_abs_error_normalised": 1e-6}
    res = {"images": {"a": {"t1_3": base, "t1_4a": [{"rel_error": 1e-8}]}}}
    assert all(ok for *_, ok in collect_checks(res))
    res["images"]["a"]["t1_3"] = {**base, "abs_error_logit": 1.0}
    assert not all(ok for *_, ok in collect_checks(res))
