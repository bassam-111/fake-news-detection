import pytest

import app as app_module


class _FakeClassifier:
    def decision_function(self, _texts):
        return [1.5]


class _FakeModel:
    def __init__(self, prediction=1):
        self.prediction = prediction
        self.named_steps = {'clf': _FakeClassifier()}

    def predict(self, _texts):
        return [self.prediction]

    def decision_function(self, _texts):
        return [1.5]


@pytest.fixture
def client():
    original_model = app_module.model
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as test_client:
        yield test_client
    app_module.model = original_model


def test_health_not_ready_without_model(client):
    app_module.model = None
    response = client.get('/health')

    assert response.status_code == 200
    assert response.get_json() == {'status': 'not_ready', 'model_loaded': False}


def test_predict_requires_loaded_model(client):
    app_module.model = None
    response = client.post('/predict', json={'text': 'sample text'})

    assert response.status_code == 400
    assert 'Model not loaded' in response.get_json()['error']


def test_predict_rejects_missing_text_field(client):
    app_module.model = _FakeModel()
    response = client.post('/predict', json={})

    assert response.status_code == 400
    assert 'Missing "text" field' in response.get_json()['error']


def test_predict_returns_fake_label_and_score(client):
    app_module.model = _FakeModel(prediction=1)
    response = client.post('/predict', json={'text': 'Breaking story'})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['prediction'] == 'FAKE'
    assert payload['score'] == pytest.approx(1.5)
    assert payload['confidence'] == pytest.approx(1.5)


def test_predict_returns_real_label(client):
    app_module.model = _FakeModel(prediction=0)
    response = client.post('/predict', json={'text': 'Verified report'})

    assert response.status_code == 200
    assert response.get_json()['prediction'] == 'REAL'
