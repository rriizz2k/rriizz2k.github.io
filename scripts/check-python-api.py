import json
from pathlib import Path
from fastapi.testclient import TestClient
from python_api.app import app

root = Path(__file__).resolve().parents[1]
client = TestClient(app)
cases = json.loads((root/'scripts/parity-cases.json').read_text())
max_error = 0
for c in cases:
    response=client.post('/api/predict',json={'application':c['input']})
    assert response.status_code==200,response.text
    max_error=max(max_error,abs(response.json()['probability']-c['probability']))
    assert abs(response.json()['probability']-c['probability'])<1e-12
for value in [{'duration':0},{'duration':True},{'duration':'24'},{'credit_amount':-1},{'checking_status':'wrong'},{'unknown':1}]:
    assert client.post('/api/predict',json={'application':value}).status_code==422
assert len(client.post('/api/predict',json={'application':{'age':30}}).json()['imputed_fields'])==19
print(json.dumps({'python_api_parity_cases':len(cases),'max_absolute_error':max_error,'invalid_inputs_rejected':True}))
