import {InputError, scoreApplication} from '../../../lib/inference';

function cors(request: Request) {
  const origin = request.headers.get('Origin');
  const headers: Record<string,string> = {'Content-Type':'application/json; charset=utf-8', 'Cache-Control':'no-store', 'Vary':'Origin'};
  if (origin === 'https://rriizz2k.github.io' || (origin && /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin))) {
    headers['Access-Control-Allow-Origin'] = origin;
    headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS';
    headers['Access-Control-Allow-Headers'] = 'Content-Type';
  }
  return headers;
}

export async function POST(request: Request) {
  const headers = cors(request);
  const body = await request.text();
  if (body.length > 16000) return Response.json({error:'Слишком большой запрос'}, {status:413,headers});
  let parsed: unknown;
  try { parsed=JSON.parse(body); } catch { return Response.json({error:'Тело запроса должно быть JSON'}, {status:400,headers}); }
  if (!parsed || typeof parsed!=='object' || Array.isArray(parsed) ||
      Object.keys(parsed).some(k=>k!=='application') || !('application' in parsed)) {
    return Response.json({error:'Ожидается объект { application: {...} }'}, {status:422,headers});
  }
  try { return Response.json(scoreApplication((parsed as {application:unknown}).application), {headers}); }
  catch(error) {
    if (error instanceof InputError) return Response.json({error:error.message,details:error.issues}, {status:422,headers});
    console.error('Credit inference failed', error instanceof Error ? error.message : 'unknown');
    return Response.json({error:'Не удалось выполнить расчёт. Попробуйте ещё раз.'}, {status:500,headers});
  }
}

export function OPTIONS(request: Request) { return new Response(null,{status:204,headers:cors(request)}); }
