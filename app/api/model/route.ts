import report from '../../../lib/report.json';
export function GET() { return Response.json(report,{headers:{'Access-Control-Allow-Origin':'https://rriizz2k.github.io'}}); }
