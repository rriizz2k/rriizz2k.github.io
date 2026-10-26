"""Python alternative to the website API, using the same exported original model.

uvicorn python_api.app:app --host 0.0.0.0 --port 8000
No pickle deserialization and no training at startup.
"""
import json
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / 'lib/model.json').read_text())
BOUNDS = MODEL['schema']['numeric']


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    application: dict[str, StrictFloat | StrictInt | StrictStr | None]


def predict(application):
    values = {**application,
              'credit_amount_log': math.log1p(application['credit_amount']),
              'credit_amount_per_month': application['credit_amount'] / application['duration']}
    vector = [(values[k]-m)/s for k,m,s in zip(MODEL['numeric_columns'], MODEL['numeric_means'], MODEL['numeric_scales'])]
    for column, categories in zip(MODEL['categorical_columns'], MODEL['one_hot_categories']):
        vector.extend(int(application[column] == category) for category in categories)
    z = MODEL['intercept'] + sum(v*c for v,c in zip(vector, MODEL['coefficients']))
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def validate(application):
    values = dict(MODEL['defaults'])
    issues = []
    for k,v in application.items():
        if k not in values:
            issues.append(f'Unknown field: {k}')
        elif v is None:
            continue
        elif k in BOUNDS:
            b = BOUNDS[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not b['min'] <= v <= b['max'] or (k!='credit_amount' and not float(v).is_integer()):
                issues.append(f'{k}: expected a number from {b["min"]} to {b["max"]}')
            else: values[k] = v
        elif not isinstance(v,str) or v not in MODEL['schema']['categorical'][k]:
            issues.append(f'Unknown category for {k}')
        else: values[k] = v
    if issues: raise HTTPException(422,detail='; '.join(issues))
    return values


app = FastAPI(title='Credit Risk Scoring', version=MODEL['model_version'],
              description='Original educational German Credit model; not a credit decision service.')
app.add_middleware(CORSMiddleware,allow_origins=['https://rriizz2k.github.io'],
                   allow_methods=['GET','POST'],allow_headers=['Content-Type'])


@app.get('/api/health')
def health(): return {'status':'ok','model_version':MODEL['model_version']}


@app.post('/api/predict')
def score(request: ScoreRequest):
    values = validate(request.application)
    p = predict(values)
    return {'probability':p,'prediction':int(p>=MODEL['threshold']),
            'threshold':MODEL['threshold'],'model_version':MODEL['model_version'],
            'imputed_fields':[k for k in values if request.application.get(k) is None],
            'purpose':'educational_demo'}


@app.get('/api/model')
def report(): return json.loads((ROOT/'lib/report.json').read_text())
