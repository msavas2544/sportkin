import json

import numpy as np
import pytest

from conftest import make_cmj
from sportkin import cli, stats
from sportkin.io import read_columns, write_table
from sportkin.report import (
    agreement_table,
    fmt_ci,
    fmt_p,
    fmt_r,
    reliability_table,
    table_to_markdown,
)


def test_write_table_turkish_excel(tmp_path):
    p = write_table([{"a": 1.5, "b": float("nan")}, {"a": 2, "b": 3.25}], tmp_path / "t.csv",
                    decimal=",", digits=2)
    text = p.read_text(encoding="utf-8-sig").splitlines()
    assert text == ["a;b", "1,50;", "2;3,25"]
    a, b = read_columns(p, "a", "b")
    assert a == [1.5, 2.0] and np.isnan(b[0]) and b[1] == 3.25


def test_read_columns_missing(tmp_path):
    p = write_table({"x": [1.0, 2.0]}, tmp_path / "t.csv")
    with pytest.raises(KeyError):
        read_columns(p, "y")


def test_apa_number_formats():
    assert fmt_r(0.853) == ".85"
    assert fmt_r(-0.2) == "−.20"
    assert fmt_p(0.0004) == "< .001"
    assert fmt_p(0.0312) == "= .031"
    assert fmt_ci(0.81, 0.95, r_style=True) == "[.81, .95]"
    assert fmt_ci(0.81, 0.95, decimal=",", r_style=True) == "[0,81; 0,95]"
    assert fmt_r(0.853, decimal=",") == "0,85"


def _data():
    rng = np.random.default_rng(4)
    crit = rng.normal(34, 5, 20)
    return crit + rng.normal(0.4, 1.0, 20), crit


def test_tables_markdown_tr_en():
    video, crit = _data()
    t = agreement_table([("CMJ (cm)", stats.agreement(video, crit))], name_b="Mat", lang="tr",
                        decimal=",")
    md = table_to_markdown(t)
    assert "Tablo 1" in md and "SİKK(2,1)" in md and "Not." in md
    assert len(t["rows"][0]) == len(t["header"])
    r = reliability_table([("CMJ (cm)", stats.reliability(crit, crit + 0.3))])
    assert "Table 1" in table_to_markdown(r)


def test_docx_and_figure(tmp_path):
    docx = pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from sportkin.report import bland_altman_plot, table_to_docx

    video, crit = _data()
    t = agreement_table([("CMJ height (cm)", stats.agreement(video, crit))],
                        title="Agreement between video and contact mat")
    path = table_to_docx(t, tmp_path / "t.docx")
    doc = docx.Document(path)
    assert doc.paragraphs[0].text == "Table 1"
    assert doc.tables[0].cell(1, 0).text == "CMJ height (cm)"
    png = bland_altman_plot(stats.bland_altman(video, crit), tmp_path / "ba.png")
    assert png.stat().st_size > 10_000


def test_cli_cmj_from_pose_csv(tmp_path, capsys):
    pose, truth = make_cmj()
    csv_path = pose.to_csv(tmp_path / "trial_pose.csv")
    cli.main(["cmj", "--pose", str(csv_path), "--fps", "240", "--out", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out" / "trial_summary.json").read_text())
    assert len(summary["jumps"]) == 1
    assert (tmp_path / "out" / "trial_jumps.csv").exists()
    assert "jump 1" in capsys.readouterr().out


def test_cli_agreement(tmp_path):
    pytest.importorskip("docx")
    video, crit = _data()
    data = write_table({"video_cm": video, "mat_cm": crit}, tmp_path / "d.csv")
    cli.main(["agreement", str(data), "--a", "video_cm", "--b", "mat_cm", "--lang", "tr",
              "--out", str(tmp_path / "o")])
    out = json.loads((tmp_path / "o" / "agreement.json").read_text())
    assert out["icc"]["value"] > 0.9
    assert (tmp_path / "o" / "agreement_table.docx").exists()
    assert (tmp_path / "o" / "agreement_bland_altman.png").exists()
