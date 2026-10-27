import {build} from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';

// Transport failures are injected: this check never contacts the deployed service.
const temporary = fs.mkdtempSync('/private/tmp/credit-prediction-client-');
const require = createRequire(import.meta.url);
const completed = [];
let maxError = 0;
try {
  for (const [entry, output] of [['lib/prediction-client.ts', 'client.cjs'], ['lib/inference.ts', 'inference.cjs']]) {
    await build({entryPoints: [entry], bundle: true, platform: 'node', format: 'cjs', outfile: path.join(temporary, output), logLevel: 'silent'});
  }
  const {predictApplication, PredictionInputError} = require(path.join(temporary, 'client.cjs'));
  const {scoreApplication} = require(path.join(temporary, 'inference.cjs'));
  const cases = JSON.parse(fs.readFileSync('scripts/parity-cases.json', 'utf8'));
  const application = cases[0].input;
  const expected = scoreApplication(application);
  const json = (body, status = 200) => new Response(JSON.stringify(body), {status, headers: {'Content-Type': 'application/json'}});
  const html = status => new Response('<html>Upstream unavailable</html>', {status, headers: {'Content-Type': 'text/html'}});
  const options = extra => ({signal: new AbortController().signal, timeoutMs: 100, retryDelayMs: 0, ...extra});
  const test = async (name, run) => { await run(); completed.push(name); };
  const cancelled = error => error instanceof DOMException && error.name === 'AbortError';

  await test('server success preserves request and avoids fallback', async () => {
    let requests = 0;
    const result = await predictApplication(application, 'https://example.invalid/', options({
      fetcher: async (url, init) => {
        requests++;
        assert.equal(url, 'https://example.invalid/api/predict');
        assert.equal(init.method, 'POST');
        assert.equal(init.cache, 'no-store');
        assert.deepEqual(JSON.parse(init.body), {application});
        assert.equal(init.headers['Content-Type'], 'application/json');
        assert(init.signal instanceof AbortSignal);
        return json(expected);
      },
      scoreLocally: async () => assert.fail('A healthy server must not use the fallback'),
    }));
    assert.equal(requests, 1);
    assert.equal(result.source, 'server');
    assert.deepEqual(result.score, expected);
  });

  await test('temporary network failure retries and can recover on server', async () => {
    let requests = 0;
    const phases = [];
    const result = await predictApplication(application, '', options({
      onPhase: phase => phases.push(phase),
      fetcher: async () => { if (++requests === 1) throw new TypeError('Failed to fetch'); return json(expected); },
      scoreLocally: async () => assert.fail('Recovered transport must not use the fallback'),
    }));
    assert.equal(requests, 2);
    assert.equal(result.source, 'server');
    assert.deepEqual(phases, ['server', 'retry']);
  });

  for (const [name, failure] of [
    ['network failure', async () => { throw new TypeError('Failed to fetch'); }],
    ['HTML 503', async () => html(503)],
    ['invalid JSON 200', async () => new Response('{broken', {headers: {'Content-Type': 'application/json'}})],
  ]) {
    await test(`${name} retries then computes real browser fallback`, async () => {
      let requests = 0;
      const phases = [];
      const result = await predictApplication(application, '', options({
        fetcher: async () => { requests++; return failure(); },
        onPhase: phase => phases.push(phase),
      }));
      assert.equal(requests, 2);
      assert.equal(result.source, 'browser');
      assert.deepEqual(result.score, expected);
      assert.deepEqual(phases, ['server', 'retry', 'browser']);
    });
  }

  await test('403 is not retried and uses real local model', async () => {
    let requests = 0;
    const result = await predictApplication(application, '', options({fetcher: async () => { requests++; return html(403); }}));
    assert.equal(requests, 1);
    assert.equal(result.source, 'browser');
    assert.deepEqual(result.score, expected);
  });

  for (const status of [400, 413, 422]) {
    await test(`${status} input error is neither retried nor masked by fallback`, async () => {
      let requests = 0;
      await assert.rejects(predictApplication(application, '', options({
        fetcher: async () => { requests++; return json({error: 'Проверьте возраст'}, status); },
        scoreLocally: async () => assert.fail('Invalid input must not be hidden by fallback'),
      })), error => error instanceof PredictionInputError && error.message === 'Проверьте возраст');
      assert.equal(requests, 1);
    });
  }

  await test('request timeout aborts both attempts and then uses real model', async () => {
    const signals = [];
    const result = await predictApplication(application, '', options({
      timeoutMs: 5,
      fetcher: async (_url, init) => new Promise((_resolve, reject) => {
        signals.push(init.signal);
        init.signal.addEventListener('abort', () => reject(new DOMException('Timed out', 'AbortError')), {once: true});
      }),
    }));
    assert.equal(signals.length, 2);
    assert(signals.every(signal => signal.aborted));
    assert.equal(result.source, 'browser');
    assert.deepEqual(result.score, expected);
  });

  await test('already cancelled operation does not request or calculate', async () => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(predictApplication(application, '', options({
      signal: controller.signal,
      fetcher: async () => assert.fail('Cancelled operation must not request'),
      scoreLocally: async () => assert.fail('Cancelled operation must not calculate'),
    })), cancelled);
  });

  await test('cancelling in-flight request prevents retry and local fallback', async () => {
    const controller = new AbortController();
    let requests = 0;
    await assert.rejects(predictApplication(application, '', options({
      signal: controller.signal,
      fetcher: async (_url, init) => new Promise((_resolve, reject) => {
        requests++;
        init.signal.addEventListener('abort', () => reject(new DOMException('Cancelled', 'AbortError')), {once: true});
        queueMicrotask(() => controller.abort());
      }),
      scoreLocally: async () => assert.fail('Cancelled operation must not calculate'),
    })), cancelled);
    assert.equal(requests, 1);
  });

  await test('cancelling retry pause prevents a second request', async () => {
    const controller = new AbortController();
    let requests = 0;
    await assert.rejects(predictApplication(application, '', options({
      signal: controller.signal,
      retryDelayMs: 100,
      fetcher: async () => { requests++; setTimeout(() => controller.abort(), 5); throw new TypeError('Offline'); },
      scoreLocally: async () => assert.fail('Cancelled retry must not calculate'),
    })), cancelled);
    assert.equal(requests, 1);
  });

  await test('cancelling local calculation prevents a returned result', async () => {
    const controller = new AbortController();
    await assert.rejects(predictApplication(application, '', options({
      signal: controller.signal,
      fetcher: async () => html(403),
      scoreLocally: async () => { controller.abort(); return expected; },
    })), cancelled);
  });

  for (const [field, value] of [
    ['probability', null], ['probability', 1.1], ['prediction', 2],
    ['threshold', -1], ['model_version', ''], ['imputed_fields', [12]],
  ]) {
    await test(`invalid server ${field} is not accepted as a server result`, async () => {
      let requests = 0;
      const result = await predictApplication(application, '', options({fetcher: async () => { requests++; return json({...expected, [field]: value}); }}));
      assert.equal(requests, 2);
      assert.equal(result.source, 'browser');
      assert.deepEqual(result.score, expected);
    });
  }

  await test('invalid local score is rejected instead of displayed', async () => {
    await assert.rejects(predictApplication(application, '', options({
      fetcher: async () => html(403),
      scoreLocally: async () => ({...expected, probability: NaN}),
    })), /Не удалось выполнить расчёт/);
  });

  await test('invalid input rejected by real browser model', async () => {
    await assert.rejects(predictApplication({...application, duration: 0}, '', options({fetcher: async () => html(403)})), PredictionInputError);
  });

  await test('all 200 original Python cases match through real browser fallback', async () => {
    let requests = 0;
    for (const item of cases) {
      const result = await predictApplication(item.input, '', options({fetcher: async () => { requests++; return html(403); }}));
      assert.equal(result.source, 'browser');
      const difference = Math.abs(result.score.probability - item.probability);
      maxError = Math.max(maxError, difference);
      assert(difference < 1e-12);
      assert.equal(result.score.imputed_fields.length, 0);
      assert.equal(result.score.prediction, Number(item.probability >= 0.5));
    }
    assert.equal(requests, cases.length);
    assert.equal(cases.length, 200);
  });

  console.log(JSON.stringify({transport_checks: completed.length, sklearn_parity_cases: cases.length, max_absolute_error: maxError, real_network_requests: 0, checks: completed}, null, 2));
} finally {
  fs.rmSync(temporary, {recursive: true, force: true});
}
