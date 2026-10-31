"""Gin parser. The V3 schema (three keys, rejections) is tested in tests_v3/test_gin_reader.py."""


def test_last_key_wins(tmp_path):
    from SCAPES.data.config_loader import parse_gin_config
    p = tmp_path / "x.gin"
    p.write_text('model.size = "large"\ntraining.max_steps = 300\nmodel.size = "small"\ntraining.max_steps = 20\n')
    cfg = parse_gin_config(p)
    assert cfg["model"]["size"] == "small"
    assert cfg["training"]["max_steps"] == 20
