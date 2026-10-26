"""Package the ORIGINAL model for the website: one fit, no tuning or improvement.

Usage: PYTHONPATH=<original project>/src <python> scripts/export-baseline.py <project>
The split, estimator, features and class weights match original train.py (seed 42).
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from credit_risk.features.prepare import add_engineered_features, split_features_target
from credit_risk.models.baseline import build_baseline_pipeline

PROJECT = Path(sys.argv[1]).resolve()
SITE = Path(__file__).resolve().parents[1]
source = PROJECT / 'data/raw/german_credit.csv'
raw = pd.read_csv(source)
prepared = add_engineered_features(raw)
X, y = split_features_target(prepared)
train, test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
numeric = train.select_dtypes(include='number').columns.tolist()
categorical = train.select_dtypes(include='object').columns.tolist()
with threadpool_limits(limits=1):
    pipeline = build_baseline_pipeline(numeric, categorical)
    pipeline.fit(train, y_train)
    probabilities = pipeline.predict_proba(test)[:, 1]

def plain(v):
    if isinstance(v, dict): return {k:plain(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    if isinstance(v, np.ndarray): return plain(v.tolist())
    if isinstance(v, np.generic): return v.item()
    return v

raw_columns = [c for c in raw.columns if c != 'class']
raw_numeric = [c for c in numeric if c in raw_columns]
defaults = {c: float(train[c].median()) for c in raw_numeric}
defaults.update({c: str(train[c].mode().iloc[0]) for c in categorical})
schema = {'numeric': {c:{'min':float(raw[c].min()),'max':float(raw[c].max())} for c in raw_numeric},
          'categorical': {c:sorted(raw[c].unique().tolist()) for c in categorical}}
prep = pipeline.named_steps['preprocessor']
model = {
    'format_version': 1, 'model_version': 'credit-g-original-42-v1',
    'kind': 'logistic', 'feature_mode': 'original', 'threshold': .5,
    'numeric_columns': numeric, 'categorical_columns': categorical,
    'numeric_means': prep.named_transformers_['num'].mean_.tolist(),
    'numeric_scales': prep.named_transformers_['num'].scale_.tolist(),
    'one_hot_categories': [x.tolist() for x in prep.named_transformers_['cat'].categories_],
    'coefficients': pipeline.named_steps['model'].coef_[0].tolist(),
    'intercept': float(pipeline.named_steps['model'].intercept_[0]),
    'defaults': defaults, 'schema': schema,
    'training_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'sklearn_version': sklearn.__version__,
}
predictions = probabilities >= .5
metric = {'roc_auc':float(roc_auc_score(y_test,probabilities)),
          'average_precision':float(average_precision_score(y_test,probabilities)),
          'brier_score':float(brier_score_loss(y_test,probabilities)),
          'recall':float(recall_score(y_test,predictions)),
          'precision':float(precision_score(y_test,predictions)),
          'threshold':.5,'confusion_matrix':confusion_matrix(y_test,predictions).tolist()}
fpr,tpr,_=roc_curve(y_test,probabilities)
indices=np.unique(np.linspace(0,len(fpr)-1,min(45,len(fpr))).astype(int))
report = {
    'version':model['model_version'], 'status':'original_baseline_reproduced',
    'model':{'algorithm':'Logistic Regression','version':model['model_version'],
             'description':"Исходная конфигурация: C=1, class_weight='balanced', два производных признака."},
    'data':{'name':'German Credit / OpenML credit-g v1','rows':1000,'features':20,'good':700,'bad':300,
            'source':'https://www.openml.org/d/31',
            'original_source':'https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data',
            'attribution':'Hofmann, H. (1994). Statlog (German Credit Data). UCI. DOI:10.24432/C5NC77',
            'license':'CC BY 4.0','sha256':model['training_data_sha256']},
    'protocol':{'seed':42,'train_rows':800,'test_rows':200,'test_positive':int(y_test.sum()),
                'note':'Исходный split 80/20, stratify, seed 42. В исходном проекте test уже использовался для сравнения моделей. Это не независимая внешняя проверка.'},
    'selected':metric,'baseline':metric,
    'curves':{'roc':[[float(fpr[i]),float(tpr[i])] for i in indices]},
    'limitations':['1000 исторических заявок; результаты не относятся к современному банковскому портфелю.',
                   'Одна проверочная выборка; независимая валидация и улучшение модели — следующий этап.',
                   'Выход predict_proba — оценка модели; её калибровка не подтверждена.',
                   'Есть исторические социально-демографические признаки; fairness-аудит не проводился.',
                   'Демо не принимает реальные решения о выдаче кредита.'],
    'reproduction':{'original_repository':'https://github.com/rriizz2k/credit-risk-scoring',
                    'original_commit':'d350a28e5b131ca1b69414a2664d1a9f98c26e51',
                    'purpose':'Воспроизведение исходной модели для демо, без подбора новых настроек.',
                    'sklearn_version':sklearn.__version__},
}
order=np.argsort(probabilities)
chosen=[order[int((len(order)-1)*q)] for q in [.08,.5,.9]]
examples=[{'id':['low','medium','high'][i],'label':['Пример A','Пример B','Пример C'][i],
           'features':raw.loc[test.index[j],raw_columns].to_dict(),
           'probability':float(probabilities[j]),'actual_class':int(y_test.iloc[j])} for i,j in enumerate(chosen)]
cases=[{'input':raw.loc[test.index[j],raw_columns].to_dict(),'probability':float(probabilities[j])}
       for j in range(len(test))]
(SITE/'lib').mkdir(exist_ok=True)
(SITE/'public/downloads').mkdir(exist_ok=True,parents=True)
payloads={'model.json':model,'report.json':report,'examples.json':examples,
          'model-public.json':{k:model[k] for k in ['model_version','kind','feature_mode','threshold','defaults','schema']}}
for name,payload in payloads.items():
    (SITE/'lib'/name).write_text(json.dumps(plain(payload),ensure_ascii=False,indent=2))
(SITE/'public/downloads/evaluation.json').write_text(json.dumps(plain(report),ensure_ascii=False,indent=2))
(SITE/'scripts/parity-cases.json').write_text(json.dumps(plain(cases),ensure_ascii=False,indent=2))
print(json.dumps({'metrics':metric,'example_scores':[x['probability'] for x in examples],
                  'model_version':model['model_version']},ensure_ascii=False,indent=2))
