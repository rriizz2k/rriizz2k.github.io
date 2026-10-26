import modelData from './model.json';

type Input = Record<string, number | string>;
const model = modelData;
const numericBounds: Record<string, [number, number]> = {
  duration: [4, 72], credit_amount: [250, 18424], age: [19, 75],
  installment_commitment: [1, 4], residence_since: [1, 4],
  existing_credits: [1, 4], num_dependents: [1, 2],
};
const labels: Record<string, string> = {
  duration: 'Срок кредита', credit_amount: 'Сумма кредита', age: 'Возраст',
  checking_status: 'Текущий счёт', credit_history: 'Кредитная история',
  purpose: 'Цель кредита', savings_status: 'Сбережения', employment: 'Стаж работы',
  installment_commitment: 'Доля платежа в доходе', personal_status: 'Личный статус',
  other_parties: 'Поручительство', residence_since: 'Срок проживания',
  property_magnitude: 'Имущество', other_payment_plans: 'Другие платежи',
  housing: 'Жильё', existing_credits: 'Количество кредитов', job: 'Занятость',
  num_dependents: 'Иждивенцы', own_telephone: 'Телефон', foreign_worker: 'Иностранный работник',
};

export class InputError extends Error {
  issues: string[];
  constructor(issues: string[]) { super(issues.join('; ')); this.issues = issues; }
}

export function validateApplication(value: unknown): {application: Input; imputed: string[]} {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new InputError(['application должен быть объектом с характеристиками заявки']);
  }
  const input = value as Record<string, unknown>;
  const defaults = model.defaults as Input;
  const categories = model.schema.categorical as Record<string, string[]>;
  const result: Input = {...defaults};
  const issues: string[] = [];
  for (const [name, v] of Object.entries(input)) {
    if (!(name in defaults)) { issues.push(`Неизвестное поле: ${name}`); continue; }
    if (v === null || v === undefined) continue;
    if (name in numericBounds) {
      const [min, max] = numericBounds[name];
      if (typeof v !== 'number' || !Number.isFinite(v) ||
          (name !== 'credit_amount' && !Number.isInteger(v)) || v < min || v > max) {
        issues.push(`${labels[name]}: число от ${min} до ${max}`);
      } else result[name] = v;
    } else if (typeof v !== 'string' || !categories[name]?.includes(v)) {
      issues.push(`${labels[name]}: выберите одно из значений ${categories[name]?.join(', ')}`);
    } else result[name] = v;
  }
  if (issues.length) throw new InputError(issues);
  return {application: result, imputed: Object.keys(defaults).filter(k=>input[k] === null || input[k] === undefined)};
}

export function probability(application: Input): number {
  const values: Record<string, number|string> = {
    ...application,
    credit_amount_log: Math.log1p(Number(application.credit_amount)),
    credit_amount_per_month: Number(application.credit_amount) / Number(application.duration),
  };
  const vector = model.numeric_columns.map((name, i) =>
    (Number(values[name]) - model.numeric_means[i]) / model.numeric_scales[i]);
  model.categorical_columns.forEach((name, i) => {
    for (const category of model.one_hot_categories[i]) vector.push(application[name] === category ? 1 : 0);
  });
  const logit = vector.reduce((sum, x, i) => sum + x * model.coefficients[i], model.intercept);
  return logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));
}

export function scoreApplication(value: unknown) {
  const {application, imputed} = validateApplication(value);
  const p = probability(application);
  const defaults = model.defaults as Input;
  const factors = Object.keys(defaults)
    .filter(k=>application[k] !== defaults[k])
    .map(feature=>({feature, label: labels[feature],
      delta: p - probability({...application, [feature]: defaults[feature]})}))
    .sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta)).slice(0,4);
  return {
    probability: p, prediction: p >= model.threshold ? 1 : 0,
    threshold: model.threshold, model_version: model.model_version,
    imputed_fields: imputed, factors, purpose: 'educational_demo',
    factor_method: 'Изменение прогноза при замене одного признака типичным значением из train. Не причинный эффект и не SHAP.',
  };
}

export const modelVersion = model.model_version;
