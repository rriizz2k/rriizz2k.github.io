import {modelVersion} from '../../../lib/inference';
export function GET() {
  return Response.json({status:'ok',model_version:modelVersion,mode:'original_baseline'},
    {headers:{'Cache-Control':'no-store','Access-Control-Allow-Origin':'https://rriizz2k.github.io'}});
}
