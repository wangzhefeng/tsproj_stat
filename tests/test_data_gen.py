from eda.data_gen import generate_model_test_data


def test_generate_model_test_data_smoke():
    df = generate_model_test_data(n_points=60, season_period=7)
    assert list(df.columns) == ["ds", "y"]
    assert len(df) == 60
