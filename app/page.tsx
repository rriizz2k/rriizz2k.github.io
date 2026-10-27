'use client';

import { useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowDown, ArrowRight, ArrowUpRight, Check, ChevronDown, Code2, Copy, Database, Download, ExternalLink, GitBranch as Github, GraduationCap, Layers3, LoaderCircle, Mail, RotateCcw, Send, SlidersHorizontal, Terminal, TrendingUp } from 'lucide-react';
import rawReport from '../lib/report.json';
import rawModel from '../lib/model-public.json';
import rawExamples from '../lib/examples.json';
import {predictApplication, type PredictionPhase} from '../lib/prediction-client';

type Application = Record<string, string | number>;
type ModelPublic = {model_version: string; threshold: number; defaults: Application; schema: {numeric: Record<string, {min: number; max: number}>; categorical: Record<string, string[]>}};
type Score = {probability: number; prediction: number; threshold: number; model_version: string; imputed_fields: string[]; factors?: {feature: string; label: string; delta: number}[]};
type Report = {selected: {roc_auc: number; average_precision: number; brier_score: number; recall: number; precision: number; threshold: number; confusion_matrix: number[][]}; data: {rows: number; features: number; good: number; bad: number; source: string; original_source: string; attribution: string; license: string}; protocol: {seed: number; train_rows: number; test_rows: number; test_positive: number; note?: string}; curves: {roc: number[][]}; model?: {algorithm: string; version: string; description: string}; limitations: string[]};
const report = rawReport as unknown as Report;
const model = rawModel as unknown as ModelPublic;
const examples = rawExamples as unknown as {id: string; label: string; features: Application; probability: number}[];
const metric = report.selected;
const number = (n: number) => new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(n);
const pct = (n: number, digits = 1) => new Intl.NumberFormat('ru-RU', {style: 'percent', maximumFractionDigits: digits}).format(n);
const decimal = (n: number) => n.toFixed(3).replace('.', ',');
const apiBase = () => ((globalThis as typeof globalThis & {__CREDIT_API_ORIGIN__?: string}).__CREDIT_API_ORIGIN__ || '').replace(/\/$/, '');
const initial = {...model.defaults, ...(examples[0]?.features || {})};

const labels: Record<string, string> = {
  duration: 'Срок кредита', credit_amount: 'Сумма кредита', age: 'Возраст',
  checking_status: 'Остаток на текущем счёте', credit_history: 'Кредитная история', employment: 'Стаж на текущей работе',
  purpose: 'Цель кредита', savings_status: 'Сбережения', installment_commitment: 'Платёжная нагрузка · категория',
  personal_status: 'Личный статус · категория', other_parties: 'Созаявитель или поручитель', residence_since: 'Срок проживания · категория',
  property_magnitude: 'Имущество', other_payment_plans: 'Другие рассрочки', housing: 'Жильё', existing_credits: 'Кредитов в этом банке',
  job: 'Занятость · категория', num_dependents: 'Людей на содержании', own_telephone: 'Телефон на имя заявителя', foreign_worker: 'Иностранный работник',
};
const categoryLabels: Record<string, Record<string, string>> = {
  checking_status: {'<0': 'Отрицательный остаток', '0<=X<200': 'От 0 до 200 DM', '>=200': 'От 200 DM и выше', 'no checking': 'Нет текущего счёта'},
  credit_history: {'all paid': 'Все кредиты погашены в этом банке', 'critical/other existing credit': 'Критический счёт / другие кредиты', 'delayed previously': 'Были задержки платежей', 'existing paid': 'Платежи по кредитам вовремя', 'no credits/all paid': 'Нет кредитов / все погашены'},
  employment: {'unemployed': 'Без работы', '<1': 'Меньше года', '1<=X<4': 'От 1 до 4 лет', '4<=X<7': 'От 4 до 7 лет', '>=7': '7 лет и больше'},
  purpose: {'business': 'Бизнес', 'domestic appliance': 'Бытовая техника', 'education': 'Образование', 'furniture/equipment': 'Мебель / оборудование', 'new car': 'Новый автомобиль', 'other': 'Другое', 'radio/tv': 'Радио / телевизор', 'repairs': 'Ремонт', 'retraining': 'Переобучение', 'used car': 'Подержанный автомобиль'},
  savings_status: {'100<=X<500': 'От 100 до 500 DM', '500<=X<1000': 'От 500 до 1 000 DM', '<100': 'Меньше 100 DM', '>=1000': 'От 1 000 DM', 'no known savings': 'Нет данных о сбережениях'},
  personal_status: {'female div/dep/mar': 'Женщина · разведена / замужем', 'male div/sep': 'Мужчина · разведён / раздельное проживание', 'male mar/wid': 'Мужчина · женат / вдовец', 'male single': 'Мужчина · не женат'},
  other_parties: {'none': 'Нет', 'co applicant': 'Созаявитель', 'guarantor': 'Поручитель'},
  property_magnitude: {'car': 'Автомобиль / другое', 'life insurance': 'Страхование жизни / накопления', 'no known property': 'Нет данных об имуществе', 'real estate': 'Недвижимость'},
  other_payment_plans: {'bank': 'Банк', 'none': 'Нет', 'stores': 'Магазины'}, housing: {'for free': 'Без платы за жильё', 'own': 'Собственное', 'rent': 'Аренда'},
  job: {'high qualif/self emp/mgmt': 'Высокая квалификация / свой бизнес', 'skilled': 'Квалифицированный сотрудник', 'unemp/unskilled non res': 'Без работы / неквалифицированный нерезидент', 'unskilled resident': 'Неквалифицированный резидент'},
  own_telephone: {'none': 'Нет', 'yes': 'Да'}, foreign_worker: {'no': 'Нет', 'yes': 'Да'},
};
const primary = ['credit_amount', 'duration', 'age'];
const allFields = [...Object.keys(model.schema.numeric), ...Object.keys(model.schema.categorical)];
const advanced = allFields.filter(k => !primary.includes(k));

function RocChart({large = false}: {large?: boolean}) {
  const pts = report.curves.roc;
  const x = (n: number) => 42 + n * 356;
  const y = (n: number) => 206 - n * 170;
  const path = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p[0])},${y(p[1])}`).join(' ');
  return <svg className={`roc-chart ${large ? 'roc-chart-large' : ''}`} viewBox="0 0 438 244" role="img" aria-label={`ROC-кривая исходной модели. ROC-AUC ${decimal(metric.roc_auc)} на проверочной выборке из ${report.protocol.test_rows} заявок.`}>
    <defs><linearGradient id={large ? 'roc-fill-large' : 'roc-fill'} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="currentColor" stopOpacity=".22"/><stop offset="100%" stopColor="currentColor" stopOpacity=".01"/></linearGradient></defs>
    {[0, .25, .5, .75, 1].map(v => <g key={v}><line x1="42" y1={y(v)} x2="398" y2={y(v)} className="plot-grid"/><text x="31" y={y(v) + 4} textAnchor="end">{v === 0 ? '0' : String(v).replace('0.', ',')}</text></g>)}
    <line x1="42" y1="206" x2="398" y2="36" className="plot-reference"/>
    <path d={`${path} L398,206 L42,206 Z`} fill={`url(#${large ? 'roc-fill-large' : 'roc-fill'})`}/>
    <path d={path} className="roc-line"/>
    {[0, .5, 1].map(v => <text key={v} x={x(v)} y="223" textAnchor="middle">{String(v).replace('0.', ',')}</text>)}
    <text x="220" y="242" textAnchor="middle">Доля ложных срабатываний</text>
    <text x="5" y="13">Доля найденных рисковых заявок</text>
  </svg>;
}

function Field({name, value, change, compact = false}: {name: string; value: string | number; change: (key: string, value: string | number) => void; compact?: boolean}) {
  const numeric = model.schema.numeric[name];
  const id = `application-${name}`;
  return <div className={`field ${compact ? 'field-compact' : ''}`}>
    <label htmlFor={id}>{labels[name] || name}{name === 'credit_amount' && <span>немецкие марки · DM</span>}{name === 'duration' && <span>месяцев</span>}{name === 'age' && <span>лет</span>}</label>
    {numeric ? <><div className="number-field"><input id={id} type="number" inputMode="numeric" min={numeric.min} max={numeric.max} step="1" required value={value} onChange={e => change(name, e.target.value === '' ? '' : Number(e.target.value))}/>{!compact && <span aria-hidden="true">{name === 'duration' ? 'мес.' : name === 'age' ? 'лет' : 'DM'}</span>}</div>{!compact && <input className="range-input" aria-label={`${labels[name]} — ползунок`} type="range" min={numeric.min} max={numeric.max} step="1" value={Number(value) || numeric.min} onChange={e => change(name, Number(e.target.value))}/>}</> : <div className="select-wrap"><select id={id} value={String(value)} onChange={e => change(name, e.target.value)}>{(model.schema.categorical[name] || []).map(v => <option value={v} key={v}>{categoryLabels[name]?.[v] || v}</option>)}</select><ChevronDown size={15} aria-hidden="true"/></div>}
  </div>;
}

function Demo({active}: {active: boolean}) {
  const [application, setApplication] = useState<Application>(initial);
  const [result, setResult] = useState<Score | null>(null);
  const [submitted, setSubmitted] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [previous, setPrevious] = useState<number | null>(null);
  const [source, setSource] = useState<'server' | 'browser' | null>(null);
  const [phase, setPhase] = useState<PredictionPhase>('server');
  const abort = useRef<AbortController | null>(null);
  const resultRef = useRef<Score | null>(null);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState('');
  const stale = result !== null && submitted !== JSON.stringify(application);
  const runModel = async (values: Application) => {
    abort.current?.abort();
    const controller = new AbortController(); abort.current = controller;
    setLoading(true); setError('');
    try {
      const {score, source: calculatedBy} = await predictApplication(values, apiBase(), {
        signal: controller.signal,
        onPhase: next => {if(abort.current===controller && !controller.signal.aborted)setPhase(next);},
      });
      if (abort.current!==controller || controller.signal.aborted) return;
      setPrevious(resultRef.current?.probability ?? null); resultRef.current=score;
      setResult(score); setSource(calculatedBy); setSubmitted(JSON.stringify(values));
    } catch(e) {
      if (abort.current!==controller || controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : 'Не удалось выполнить расчёт. Повторите попытку.');
    } finally {if(abort.current===controller)setLoading(false);}
  };
  useEffect(()=>{if(!active)return;if(!resultRef.current)void runModel({...application});return()=>abort.current?.abort();},[active]);
  const change=(key:string,value:string|number)=>{abort.current?.abort();setLoading(false);setError('');setApplication(old=>({...old,[key]:value}));};
  const submit=(event:FormEvent)=>{event.preventDefault();void runModel({...application});};
  const reset=()=>{setApplication({...initial});void runModel({...initial});};
  const copyRequest=async()=>{try{await navigator.clipboard.writeText(JSON.stringify({application},null,2));setCopyError('');setCopied(true);setTimeout(()=>setCopied(false),2000);}catch{setCopyError('Можно выделить и скопировать JSON в блоке ниже.');}};
  const showResult=result!==null&&!stale&&!loading&&!error;
  const probability=showResult?result.probability:0;
  const circumference=2*Math.PI*72;
  return <div className="demo-shell">
    <div className="demo-top"><div><h3>Попробуйте изменить заявку</h3><p>Поля заполнены одной обезличенной заявкой из German Credit.</p></div></div>
    <div className="demo-unit-note"><strong>DM — немецкая марка.</strong> Это денежная шкала исторического датасета, не рубли и не евро. Суммы не пересчитываются в современные деньги.</div>
    <div className="demo-body"><form onSubmit={submit} className="application-form"><div className="fields-grid">{primary.map(name=><Field key={name} name={name} value={application[name]} change={change}/>)}</div>
      <details className="advanced-fields"><summary><SlidersHorizontal size={16}/>Остальные {advanced.length} параметров заявки<ChevronDown size={16}/></summary><p className="field-help">Модель использует все 20 параметров. Здесь указаны остальные значения из того же примера; их тоже можно менять.</p><div className="fields-grid">{advanced.map(name=><Field key={name} name={name} value={application[name]} change={change} compact/>)}</div></details>
      <div className="form-actions"><button className="button primary-button" type="submit" disabled={loading}>{loading?<LoaderCircle size={17} className="spinner"/>:<ArrowRight size={17}/>} {loading?'Считаем…':'Рассчитать риск'}</button><button className="reset-button" type="button" disabled={loading} onClick={reset}><RotateCcw size={15}/>Вернуть пример</button></div><p className="field-help">При изменении трёх полей остальные 17 остаются прежними.</p>
    </form>
    <div className={`prediction-panel ${stale?'is-stale':''}`} aria-busy={loading}><div className="result-heading"><span className="tiny-label">СКОР МОДЕЛИ</span><span className={`result-chip ${stale||error?'stale-chip':''}`}>{loading?(phase==='retry'?'Повтор запроса':'Расчёт'):error?'Ошибка':stale?'Нужен пересчёт':result?'Готово':'Ожидание'}</span></div><div className="risk-gauge"><svg viewBox="0 0 180 180" aria-hidden="true"><circle cx="90" cy="90" r="72" className="gauge-track"/><circle cx="90" cy="90" r="72" className="gauge-value" strokeDasharray={circumference} strokeDashoffset={circumference*(1-probability)}/></svg><div><strong>{showResult?number(Math.round(probability*100)):'—'}</strong><span>из 100 баллов</span></div></div>
      <div className="result-explanation" aria-live="polite"><h4>{loading?(phase==='retry'?'Восстанавливаем связь':phase==='browser'?'Считаем в браузере':'Считаем оценку'):error?'Расчёт не выполнен':stale?'Заявка изменена':result?result.prediction?'Повышенный риск по модели':'Пониженный риск по модели':'Результат появится здесь'}</h4><p>{loading?'Если сервер не ответит, та же модель выполнит расчёт в браузере.':error?'Проверьте заявку и нажмите «Рассчитать риск».':stale?'Нажмите «Рассчитать риск», чтобы получить новую оценку.':result?`Чем выше скор, тем выше оценённый моделью риск. Граница классов этой версии — ${number(result.threshold*100)} баллов.`:'Нажмите «Рассчитать риск».'}</p></div>
      {previous!==null&&showResult&&result&&<div className="previous-score"><span>К предыдущему расчёту</span><strong>{result.probability-previous>=0?'+':'−'}{new Intl.NumberFormat('ru-RU',{maximumFractionDigits:1}).format(Math.abs(result.probability-previous)*100)} балла</strong></div>}
      {error&&<p className="error-message" role="alert">{error}</p>}{showResult&&source==='browser'&&<p className="fallback-note" role="status">Сервер недоступен. Та же модель выполнила расчёт в браузере.</p>}<div className="result-bottom"><Code2 size={14}/><span>{showResult?(source==='browser'?'Расчёт в браузере':'Расчёт на сервере'):'Учебная модель'} · Logistic Regression</span></div></div></div>
    <div className="model-confidence-note"><strong>Как понимать результат?</strong> Это учебный скор, а не подтверждённая вероятность невозврата. Модель может ошибаться и не принимает решений о выдаче кредита. <a href="#about-project">Данные, проверка и ограничения →</a></div>
    <details className="request-details"><summary><Terminal size={16}/>Для разработчиков: запрос к API<ChevronDown size={16}/></summary><div className="request-heading"><code>POST /api/predict</code><button type="button" className="text-button" onClick={copyRequest}>{copied?<Check size={15}/>:<Copy size={15}/>} {copied?'Скопировано':'Скопировать JSON'}</button></div><pre>{JSON.stringify({application},null,2)}</pre>{copyError&&<p role="status">{copyError}</p>}<p>Ответ содержит оценку, класс, порог и версию модели. <a href={`${apiBase()}/api/model`} target="_blank" rel="noreferrer">Технический отчёт <ArrowUpRight size={13}/></a></p></details>
  </div>;
}

type SectionName = 'visitka' | 'project' | 'about-project' | 'contact';
const navigation: {id: SectionName; label: string}[] = [
  {id:'visitka',label:'Визитка'}, {id:'project',label:'Проект'},
  {id:'about-project',label:'О проекте'}, {id:'contact',label:'Связаться'},
];

export default function Home() {
  const [section, setSection] = useState<SectionName>('visitka');
  useEffect(() => {
    const readHash = () => {
      const hash = window.location.hash.slice(1);
      const normalized = hash==='demo' ? 'project' : hash==='about' || hash==='results' || hash==='process' ? 'about-project' : hash;
      setSection(navigation.some(x=>x.id===normalized) ? normalized as SectionName : 'visitka');
      window.scrollTo({top:0,behavior:'instant'});
    };
    readHash(); window.addEventListener('hashchange',readHash);
    return () => window.removeEventListener('hashchange',readHash);
  },[]);
  const navigate = (e: {preventDefault():void}, next: SectionName) => {
    e.preventDefault(); setSection(next); window.history.pushState(null,'',`#${next}`);
    window.scrollTo({top:0,behavior:'instant'});
  };
  const [tn,fp] = metric.confusion_matrix[0];
  const [fn,tp] = metric.confusion_matrix[1];
  return <>
    <a className="skip-link" href="#main">Перейти к содержанию</a>
    <header className="site-header"><div className="header-inner">
      <a className="brand" href="#visitka" onClick={e=>navigate(e,'visitka')}><span className="monogram">РМ</span><span>Руслан Мирзоев</span></a>
      <nav aria-label="Разделы сайта">{navigation.map(item=><a key={item.id} href={`#${item.id}`} aria-current={section===item.id?'page':undefined} onClick={e=>navigate(e,item.id)}>{item.label}</a>)}</nav>
    </div></header>
    <main id="main">
      <section id="visitka" className="view visitka-view" hidden={section!=='visitka'}>
        <div className="availability"><span/>Ищу стажировку / Junior Data Scientist</div>
        <h1>Руслан<br className="mobile-break"/> Мирзоев.</h1>
        <p className="hero-subtitle">Data Science.<br/>Финансовые задачи.</p>
        <p className="hero-description">Учусь на бизнес-информатике в РЭУ им. Плеханова. Изучаю машинное обучение через проекты и хочу применять его к задачам бизнеса.</p>
        <div className="hero-actions"><a className="button primary-button" href="#project" onClick={e=>navigate(e,'project')}>Посмотреть проект <ArrowRight size={17}/></a><a className="button secondary-button" href="#contact" onClick={e=>navigate(e,'contact')}>Связаться <ArrowUpRight size={17}/></a></div>
        <div className="profile-facts"><span>Python · SQL · Machine Learning</span><span>РЭУ им. Плеханова · 3 курс</span></div>
        <a className="featured-project" href="#project" onClick={e=>navigate(e,'project')}><div><span className="eyebrow">ПРОЕКТ, КОТОРЫЙ МОЖНО ПОПРОБОВАТЬ</span><h2>Credit Risk Scoring.</h2><p>Данные заявки → модель → оценка риска.</p></div><ArrowUpRight size={26}/></a>
        
      </section>

      <section id="project" className="view project-view" hidden={section!=='project'}>
        <div className="page-intro"><span className="eyebrow">ИНТЕРАКТИВНЫЙ ПРОЕКТ</span><h1>Credit Risk Scoring.</h1><p>Измените заявку. Посмотрите, как меняется оценка модели.</p></div>
        <Demo active={section==='project'}/>
        <p className="demo-caption">Оценку считает обученная модель. При недоступности сервера расчёт выполняется в браузере с теми же весами. Данные заявки не сохраняются.</p>
        <a className="plain-link" href="#about-project" onClick={e=>navigate(e,'about-project')}>Как устроена модель и какие у неё результаты <ArrowRight size={16}/></a>
      </section>

      <section id="about-project" className="view about-project-view" hidden={section!=='about-project'}>
        <div className="page-intro"><span className="eyebrow">О ПРОЕКТЕ</span><h1>От данных<br/>до работающего сервиса.</h1><p>Учебный скоринг, который оценивает принадлежность заявки к классу повышенного кредитного риска.</p></div>
        <div className="simple-metrics"><div><strong>{decimal(metric.roc_auc)}</strong><span>ROC-AUC</span><p>Ранжирование заявок</p></div><div><strong>{pct(metric.recall)}</strong><span>Recall</span><p>Найденные заявки класса bad</p></div><div><strong>{pct(metric.precision)}</strong><span>Precision</span><p>Доля bad среди отмеченных</p></div></div>
        <p className="metrics-note">Результаты воспроизведённой исходной модели на {report.protocol.test_rows} проверочных заявках. Порог — {pct(metric.threshold,0)}.</p>
        <div className="about-grid"><article><Database size={23}/><h2>На каких данных</h2><p><strong>German Credit:</strong> {number(report.data.rows)} исторических заявок, {report.data.features} исходных признаков. {report.data.good} заявок класса good и {report.data.bad} — bad.</p><a href={report.data.original_source} target="_blank" rel="noreferrer">Посмотреть датасет <ArrowUpRight size={14}/></a></article><article><TrendingUp size={23}/><h2>Зачем этот продукт</h2><p>Пройти путь от анализа банковских данных до модели, прогноз которой можно получить через интерфейс и API. Исследовать, какие ошибки она допускает.</p></article></div>
        <div className="simple-process"><h2>Как создан</h2><ol><li><span>01</span><div><strong>Изучение данных</strong><p>Распределения, баланс классов и особенности признаков.</p></div></li><li><span>02</span><div><strong>Подготовка и обучение</strong><p>Масштабирование чисел, кодирование категорий, производные признаки. Сравнение Logistic Regression и LightGBM.</p></div></li><li><span>03</span><div><strong>Сохранение и объяснение</strong><p>Эксперименты в MLflow. Отдельный SHAP-анализ для LightGBM.</p></div></li><li><span>04</span><div><strong>API и интерфейс</strong><p>В исходном проекте — FastAPI и Docker. Веб-демо использует экспорт тех же обученных весов. Основной расчёт — через API; при сбое связи та же модель работает в браузере.</p></div></li></ol></div>
        <details className="info-details"><summary>Подробные результаты <ChevronDown size={18}/></summary><div className="detail-columns"><div><h3>ROC-кривая</h3><RocChart large/><p>Кривая построена по фактическим прогнозам исходной модели.</p></div><div><h3>Ошибки на проверке</h3><div className="confusion-grid"><div className="matrix-cell correct"><strong>{tn}</strong><span>Верно определены good</span></div><div className="matrix-cell wrong"><strong>{fp}</strong><span>Ложные срабатывания</span></div><div className="matrix-cell wrong"><strong>{fn}</strong><span>Пропущены bad</span></div><div className="matrix-cell correct"><strong>{tp}</strong><span>Найдены bad</span></div></div><p>Average Precision: {decimal(metric.average_precision)} · Brier score: {decimal(metric.brier_score)}.</p></div></div></details>
        <details className="info-details"><summary>Проверка и ограничения <ChevronDown size={18}/></summary><p><strong>Совпадение расчётов проверено:</strong> веб-версия и исходная Python-модель дают одинаковые прогнозы на 200 заявках. Это проверка реализации, а не гарантия качества прогнозов.</p><p>Скор в демо — выход модели, умноженный на 100. Это не отдельная банковская скоринговая шкала и не подтверждённая вероятность дефолта.</p><p>На этой проверочной выборке {tp} из {tp+fp} заявок, отмеченных моделью как рискованные, действительно относятся к классу bad. Остальные {fp} — ложные срабатывания.</p><p>Разбиение: {report.protocol.train_rows} заявок для обучения, {report.protocol.test_rows} для проверки, seed {report.protocol.seed}. {report.protocol.note}</p><ul>{report.limitations.map(text=><li key={text}>{text}</li>)}</ul><p className="attribution">{report.data.attribution}. Лицензия: {report.data.license}.</p></details>
        <div className="source-links"><a href="https://github.com/rriizz2k/credit-risk-scoring" target="_blank" rel="noreferrer"><Github size={17}/>Исходный ML-проект <ArrowUpRight size={14}/></a><a href="/downloads/credit-risk-source.zip" download><Download size={17}/>Код демо</a><a href="/downloads/evaluation.json" download><Download size={17}/>Отчёт с метриками</a></div>
      </section>

      <section id="contact" className="view contact-view" hidden={section!=='contact'}>
        <span className="eyebrow">СВЯЗАТЬСЯ</span><h1>Давайте<br/>познакомимся.</h1><p className="contact-intro">Буду рад обсудить стажировку, junior-позицию<br className="desktop-break"/> или задачи вашей команды.</p><div className="contact-options"><a href="mailto:ruslanmirzoev60@gmail.com"><Mail size={25}/><div><span>Почта</span><strong>ruslanmirzoev60@gmail.com</strong></div><ArrowUpRight size={21}/></a><a href="https://t.me/mirzoevXO" target="_blank" rel="noreferrer"><Send size={25}/><div><span>Telegram</span><strong>@mirzoevXO</strong></div><ArrowUpRight size={21}/></a><a href="https://github.com/rriizz2k" target="_blank" rel="noreferrer"><Github size={25}/><div><span>GitHub</span><strong>rriizz2k</strong></div><ArrowUpRight size={21}/></a></div>
      </section>
    </main><footer className="site-footer"><span>© {new Date().getFullYear()} Руслан Мирзоев</span><span>Data Science · Учебные проекты</span></footer>
  </>;
}

