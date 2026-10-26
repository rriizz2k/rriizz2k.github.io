# Руслан Мирзоев — Data Science portfolio

Одностраничная визитка с четырьмя простыми экранами: «Визитка», «Проект», «О проекте», «Связаться». Работает демо исходного credit-risk-scoring.

- Страница: https://rriizz2k.github.io/
- API: https://ruslan-mirzoev-credit-risk.zestyaphid4.chatgpt.site/api/predict
- Исходный ML-проект: https://github.com/rriizz2k/credit-risk-scoring

## Текущая модель

В демо используется исходная Logistic Regression с `C=1`, `class_weight='balanced'`, двумя производными признаками, StandardScaler и OneHotEncoder. Конфигурация воспроизведена из коммита `d350a28e5b131ca1b69414a2664d1a9f98c26e51`: **подбор улучшений для этой версии не проводился**.

Датасет German Credit / OpenML `credit-g` v1: 1000 заявок, 20 исходных признаков, 700 good / 300 bad. Разбиение 80/20, stratify, seed 42. На тех же 200 test-заявках:

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.8147619048 |
| Recall bad | 0.8166666667 |
| Precision bad | 0.5697674419 |
| Average precision | 0.6637614606 |
| Brier score | 0.1779062625 |
| Порог | 0.5 |

Матрица ошибок (строки факт good/bad, столбцы прогноз good/bad): `[[103, 37], [11, 49]]`.

Это исторический учебный бенчмарк. В исходном проекте test использовался при сравнении моделей, поэтому он не является независимой внешней оценкой. Калибровка вероятностей и fairness-аудит не подтверждены. Демо не принимает реальные решения о выдаче кредита. Улучшение модели остаётся отдельной учебной работой автора.

## Архитектура

`Python / scikit-learn → версия модели в JSON → API → React-интерфейс`

- `app/page.tsx`, `app/globals.css` — одностраничное портфолио и форма.
- `lib/model.json` — именно обученные веса, порядок признаков, средние/масштабы и one-hot категории.
- `lib/inference.ts` — серверный расчёт по экспортированной модели; без случайных чисел и ручного скоринга.
- `app/api/` — API для публикации в Workers; ввод проверяется, запросы не сохраняются.
- `python_api/app.py` — альтернативный FastAPI-сервис с тем же контрактом и моделью.
- `web/` и `vite.pages.config.ts` — статическая сборка для бесплатного GitHub Pages. Она обращается к API, а не подставляет готовые ответы.
- `lib/report.json`, `public/downloads/evaluation.json` — фактические метрики и протокол.

Экспорт не меняет предсказания: проверка на 200 случаях сравнивает TypeScript и Python API с сохранёнными прогнозами исходного sklearn-пайплайна с точностью `1e-12`.

## Запуск страницы

Node.js 22.13+ (проверено с Node 24):

```sh
npm ci
npm run build:pages
```

Файлы для GitHub Pages находятся в `dist-pages/`. Для разработки статической страницы: `npx vite --config vite.pages.config.ts`. Адрес API задаётся в `web/index.html` через `window.__CREDIT_API_ORIGIN__`.

## Python API

```sh
python -m venv .venv
.venv/bin/pip install -r python_api/requirements.txt
.venv/bin/uvicorn python_api.app:app --port 8000
```

Документация FastAPI — `/docs`. Для размещения на Render предусмотрен `render.yaml`; автоматического создания платных сервисов нет.

### Контракт

`POST /api/predict`:

```json
{"application":{"duration":24,"credit_amount":3000,"age":30}}
```

Для неполной заявки используются медианы/моды, рассчитанные **только по train**. `imputed_fields` явно перечисляет дополненные поля. Интерфейс отправляет все 20 признаков из выбранного примера. Неизвестные категории, лишние поля, неподходящие типы и выход за поддерживаемые пределы отклоняются с HTTP 422. Валидация — часть веб-обёртки, а не улучшение модели.

`GET /api/health` — версия и статус; `GET /api/model` — отчёт. API разрешает обращения со страницы `https://rriizz2k.github.io`.

## Воспроизведение экспорта

Положите исходный проект и его `data/raw/german_credit.csv` рядом. Установите его зависимости. Скрипт выполняет один fit исходной конфигурации, не ищет параметры:

```sh
PYTHONPATH=/path/to/credit-risk-scoring/src python scripts/export-baseline.py /path/to/credit-risk-scoring
npm run check:model
PYTHONPATH=. python scripts/check-python-api.py
```

## Данные и атрибуция

Hofmann, H. (1994). *Statlog (German Credit Data).* UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5NC77. Данные: https://www.openml.org/d/31. Лицензия датасета: CC BY 4.0. Сумма кредита отображается в исторической шкале DM (немецкие марки); это не современные рубли или евро, пересчёт не производится. Справочник единиц: https://posit-academy.quarto.pub/data-gallery/academy-datasets/credit-risk/. В интерфейсе используется один заполненный пример, остальные 17 полей можно раскрыть. Скор демо = выход модели × 100; это не калиброванная вероятность невозврата.

## Автор и контакты

Руслан Мирзоев — студент 3 курса РЭУ им. Г. В. Плеханова, бизнес-информатика. Самостоятельно изучает DS, ищет стажировку / junior-позицию.

- Почта: ruslanmirzoev60@gmail.com
- Telegram: https://t.me/mirzoevXO
- GitHub: https://github.com/rriizz2k
