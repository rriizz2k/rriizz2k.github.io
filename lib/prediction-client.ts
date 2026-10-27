import {scoreApplication, InputError} from './inference';

export type Application = Record<string, string | number>;
export type Score = {
  probability: number;
  prediction: number;
  threshold: number;
  model_version: string;
  imputed_fields: string[];
  factors?: {feature: string; label: string; delta: number}[];
};
export type PredictionPhase = 'server' | 'retry' | 'browser';
export type PredictionResult = {score: Score; source: 'server' | 'browser'};
type Options = {
  signal: AbortSignal;
  onPhase?: (phase: PredictionPhase) => void;
  fetcher?: typeof fetch;
  timeoutMs?: number;
  retryDelayMs?: number;
  scoreLocally?: (application: Application) => Promise<Score>;
};

export class PredictionInputError extends Error {}
class ServiceUnavailable extends Error {
  constructor(readonly retryable: boolean) { super('Сервис расчёта временно недоступен.'); }
}

function checkCancelled(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException('Расчёт отменён', 'AbortError');
}

function pause(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    checkCancelled(signal);
    const cancel = () => { clearTimeout(timer); reject(new DOMException('Расчёт отменён', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', cancel); resolve(); }, ms);
    signal.addEventListener('abort', cancel, {once: true});
  });
}

function isScore(value: unknown): value is Score {
  if (!value || typeof value !== 'object') return false;
  const data = value as Partial<Score>;
  return typeof data.probability === 'number' && Number.isFinite(data.probability) &&
    data.probability >= 0 && data.probability <= 1 &&
    (data.prediction === 0 || data.prediction === 1) &&
    typeof data.threshold === 'number' && Number.isFinite(data.threshold) && data.threshold >= 0 && data.threshold <= 1 &&
    typeof data.model_version === 'string' && data.model_version.length > 0 &&
    Array.isArray(data.imputed_fields) && data.imputed_fields.every(field => typeof field === 'string');
}

async function requestScore(application: Application, apiOrigin: string, options: Options): Promise<Score> {
  checkCancelled(options.signal);
  const controller = new AbortController();
  const cancel = () => controller.abort();
  options.signal.addEventListener('abort', cancel, {once: true});
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 3000);
  try {
    const response = await (options.fetcher ?? fetch)(`${apiOrigin.replace(/\/$/, '')}/api/predict`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({application}), signal: controller.signal, cache: 'no-store',
    });
    const isJson = response.headers.get('content-type')?.includes('application/json');
    let data: unknown;
    if (isJson) {
      try { data = await response.json(); }
      catch { throw new ServiceUnavailable(true); }
    }
    if ([400, 413, 422].includes(response.status) && isJson && data && typeof data === 'object') {
      const message = (data as {error?: unknown}).error;
      throw new PredictionInputError(typeof message === 'string' ? message : 'Проверьте значения заявки.');
    }
    if (!response.ok) throw new ServiceUnavailable(response.status === 408 || response.status === 429 || response.status >= 500);
    if (!isScore(data)) throw new ServiceUnavailable(true);
    checkCancelled(options.signal);
    return data;
  } catch (error) {
    checkCancelled(options.signal);
    if (error instanceof PredictionInputError || error instanceof ServiceUnavailable) throw error;
    throw new ServiceUnavailable(true);
  } finally {
    clearTimeout(timer);
    options.signal.removeEventListener('abort', cancel);
  }
}

// Uses the same exported weights and preprocessing as the API; no alternate model or canned answers.
async function scoreInBrowser(application: Application): Promise<Score> {
  try { return scoreApplication(application); }
  catch (error) {
    if (error instanceof InputError) throw new PredictionInputError(error.message);
    throw error;
  }
}

export async function predictApplication(application: Application, apiOrigin: string, options: Options): Promise<PredictionResult> {
  checkCancelled(options.signal);
  for (let attempt = 0; attempt < 2; attempt++) {
    options.onPhase?.(attempt === 0 ? 'server' : 'retry');
    try { return {score: await requestScore(application, apiOrigin, options), source: 'server'}; }
    catch (error) {
      checkCancelled(options.signal);
      if (error instanceof PredictionInputError) throw error;
      if (!(error instanceof ServiceUnavailable) || !error.retryable || attempt === 1) break;
      await pause(options.retryDelayMs ?? 300, options.signal);
    }
  }
  checkCancelled(options.signal);
  options.onPhase?.('browser');
  const score = await (options.scoreLocally ?? scoreInBrowser)(application);
  checkCancelled(options.signal);
  if (!isScore(score)) throw new Error('Не удалось выполнить расчёт. Обновите страницу и попробуйте ещё раз.');
  return {score, source: 'browser'};
}
