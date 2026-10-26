import {build} from 'esbuild';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
await build({entryPoints:['lib/inference.ts'],bundle:true,platform:'node',format:'cjs',outfile:'/private/tmp/credit-baseline-inference.cjs',logLevel:'silent'});
const require=createRequire(import.meta.url);
const {scoreApplication,InputError}=require('/private/tmp/credit-baseline-inference.cjs');
const cases=JSON.parse(fs.readFileSync('scripts/parity-cases.json','utf8'));
let maxError=0;
for(const c of cases){const p=scoreApplication(c.input);maxError=Math.max(maxError,Math.abs(p.probability-c.probability));assert(Math.abs(p.probability-c.probability)<1e-12);assert.equal(p.imputed_fields.length,0);assert.equal(p.prediction,Number(p.probability>=.5));}
for(const application of [{duration:0},{duration:true},{duration:'24'},{credit_amount:-1},{checking_status:'wrong'},{some_unknown_field:1},{age:NaN},{foreign_worker:'maybe'},null,[]]) assert.throws(()=>scoreApplication(application),InputError);
assert.equal(scoreApplication({duration:24}).imputed_fields.length,19);
console.log(JSON.stringify({sklearn_parity_cases:cases.length,max_absolute_error:maxError,invalid_inputs_rejected:true,model_version:scoreApplication({}).model_version}));
