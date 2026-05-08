def test_suma_simple():
    assert 1 + 1 == 2

def test_modelo_existe():
    import os
    assert os.path.exists("model.joblib")