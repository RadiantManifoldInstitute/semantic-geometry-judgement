#!/usr/bin/env python3
"""Offline saved-output entrypoints. Scientific commands are explicit, never import-time."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import json
from pathlib import Path
import sys

STUDIES = {'C':'01-scalar-and-norms','D':'02-component-measures','E':'03-two-stage-policy-checks','F':'04-routing-relation-disposition'}


def fingerprint(path):
    if path.is_symlink() or not path.is_file(): raise ValueError('regular nonsymlink input required')
    h=hashlib.sha256(); size=0
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block);size+=len(block)
    return size,h.hexdigest()


def verify(root):
    doc=json.loads((root/'B4-MANIFEST.json').read_bytes())
    for row in doc['files']:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('manifest path escape')
        path=root/rel
        if any(p.is_symlink() for p in (path,*path.parents)): raise ValueError('symlink input')
        if fingerprint(path)!=(row['bytes'],row['sha256']): raise ValueError('manifest bytes mismatch: '+row['path'])
    return {'verified_files':len(doc['files']),'science_executed':False}


def jsonl(path):
    with path.open(encoding='utf-8') as f: return [json.loads(line) for line in f]


def compare(actual, expected, path='', tolerance=Decimal('0.000000001')):
    """Report field mismatches, never silently accept a partial or rounded result."""
    errors=[]
    if isinstance(actual,dict) and isinstance(expected,dict):
        if set(actual)!=set(expected): errors.append({'path':path,'kind':'KEY_SET_MISMATCH'})
        for key in sorted(set(actual)&set(expected)): errors.extend(compare(actual[key],expected[key],path+'/'+key,tolerance))
    elif isinstance(actual,list) and isinstance(expected,list):
        if len(actual)!=len(expected):errors.append({'path':path,'kind':'LENGTH_MISMATCH'})
        for i,(a,b) in enumerate(zip(actual,expected)):errors.extend(compare(a,b,path+'/'+str(i),tolerance))
    elif type(actual) is bool or type(expected) is bool or actual is None or expected is None:
        if actual!=expected or type(actual)!=type(expected): errors.append({'path':path,'kind':'EXACT_MISMATCH'})
    else:
        try:
            a,b=Decimal(str(actual)),Decimal(str(expected))
            if not a.is_finite() or not b.is_finite() or abs(a-b)>tolerance:errors.append({'path':path,'kind':'NUMERIC_MISMATCH'})
        except InvalidOperation:
            if actual!=expected:errors.append({'path':path,'kind':'EXACT_MISMATCH'})
    return errors


def bind(root, study):
    # No model runtime, custody, executor, provider or fitting entrypoint is imported.
    sys.path.insert(0,str(root/'studies'/STUDIES[study]/'src'))
    return importlib.import_module('kf1'+study.lower()+'.analysis')


def require_gold(root, study, records):
    expected=json.loads((root/'config'/study/'construction.json').read_bytes())['record_hashes']
    needed=['sealed-gold.jsonl','sealed-join.jsonl']
    for name in needed:
        path=records/name
        if not path.exists():raise ValueError('MISSING_HASH_BOUND_GOLD_JOIN:'+study+'/'+name)
        if fingerprint(path)[1]!=expected[name]:raise ValueError('FROZEN_GOLD_JOIN_HASH_MISMATCH:'+name)


def primary_c(root, result):
    a=bind(root,'C'); import numpy as np
    recalculated={}; comparisons=[]
    for slot, enc in result['per_encoder'].items():
        rows=enc['row_outputs']; outputs={}
        for arm in a.ARMS:
            probs=np.asarray([[float(row['arms'][arm]['class_probabilities'][label]) for label in a.CLASS_ORDER] for row in rows],dtype=np.float64)
            predictions=[row['arms'][arm]['prediction'] for row in rows]
            outputs[arm]={'metrics':a._metric_bundle(rows,probs,predictions),
                'wording_stability':a._wording(rows,probs,predictions),'cross_level_conflict':a._cross_level(rows,predictions)}
            for field in outputs[arm]:
                comparisons.extend(compare(outputs[arm][field],enc['arms'][arm][field],f'/{slot}/{arm}/{field}'))
        recalculated[slot]=outputs
    return {'coverage':'C_PRIMARY_ARM_METRICS_WORDING_CROSS_LEVEL_ONLY', 'comparisons':comparisons,'recalculated':recalculated,
        'excluded':'control row predictions, fresh fits, norms provenance and latency; stored probabilities are 12-decimal projections'}


def components(root, result, study, records):
    require_gold(root,study,records)
    # Verify pair/order keys mechanically before importing a scientific kernel.
    blinded=jsonl(root/'data'/study/'prejoin/blinded-scoring-input.jsonl')
    tables={}
    for slot in ('E01','E02','E03'):
        features=jsonl(root/'data'/study/'prejoin/representations'/f'{slot}.jsonl')
        if len(features)!=len(blinded):raise ValueError('FEATURE_ROW_CENSUS_MISMATCH')
        fields=('blind_pair_id','blind_surface_id') if study=='D' else ('blind_pair_id',)
        for b,r in zip(blinded,features):
            if r['encoder_slot']!=slot or any(b[k]!=r[k] for k in fields):raise ValueError('FEATURE_ORDER_KEY_MISMATCH')
        tables[slot]=[{k:float(v) for k,v in r['representation'].items()} for r in features]
    frozen=json.loads((root/'config'/study/'thresholds.json').read_bytes())
    a=bind(root,study)
    join=importlib.import_module('kf1'+study.lower()+'.frozen_join')
    calculated={}; comparisons=[]
    for slot,features in tables.items():
        rows,reps=join._join_split(blinded,records,features,'sealed')
        calculated[slot]=a.analyze_encoder(rows,reps,frozen['per_encoder'][slot])
        comparisons.extend(compare(calculated[slot],result['per_encoder'][slot],'/per_encoder/'+slot))
    return {'coverage':study+'_SEALED_ANALYSIS_WITH_FROZEN_THRESHOLDS_NO_RETUNE', 'comparisons':comparisons,'recalculated':calculated,
        'limitation':'12-decimal feature serialization can change boundary ties; report mismatches, do not retune. Calibration curves are retained but not refitted here.'}


def summary_f(root,result):
    a=bind(root,'F'); calculated={};comparisons=[]
    for slot,enc in result['per_encoder'].items():
        stored=enc['experiment_B'];families=stored['pair_level_by_family_macro_F1']
        freeze=json.loads((root/'config/F/model-and-router-freeze.json').read_bytes())['per_encoder'][slot]['relation']
        full,best=freeze['full_feature_arm'],freeze['best_singleton']; keys=sorted(families[full])
        effects=[float(families[full][k])-float(families[best][k]) for k in keys]
        inference=a.exact_sign_flip(effects)
        inference['descriptive_cluster_bootstrap']=a.cluster_interval(dict(zip(keys,effects)))
        others={}
        for arm in a.FEATURE_ARMS:
            if arm!=full:others[arm]=a.exact_sign_flip([float(families[full][k])-float(families[arm][k]) for k in keys])
        running=0.0
        for rank,(arm,p) in enumerate(sorted(((arm,value['two_sided_p']) for arm,value in others.items()),key=lambda r:(r[1],r[0]))):
            running=max(running,min(1.0,p*(len(others)-rank)));others[arm]['Holm_adjusted_p']=running
        inference['Holm_adjusted_p']=others[best]['Holm_adjusted_p']
        calculated[slot]={'full_vs_best_singleton':inference,'full_vs_other_arms':others}
        for field,value in calculated[slot].items():comparisons.extend(compare(value,stored[field],'/per_encoder/'+slot+'/experiment_B/'+field))
    return {'coverage':'F_RELATION_SUMMARY_LEVEL_INFERENCE_ONLY', 'comparisons':comparisons,'recalculated':calculated,
        'excluded':'cannot certify sampled truth/family provenance from summaries, nor regenerate predictions or model fits'}


def sample_f(root,result,records,blinded_path):
    require_gold(root,'F',records)
    provenance=json.loads((root/'config/F/SAMPLER-PROVENANCE.json').read_bytes())
    if blinded_path is None or fingerprint(blinded_path)[1]!=provenance['required_blinded_sha256']:
        raise ValueError('FROZEN_ORDINAL_TRUTH_JOIN_REQUIRED: --blinded must match original construction hash')
    blinded=jsonl(blinded_path)
    a=bind(root,'F');import numpy as np
    join=importlib.import_module('kf1f.frozen_join')
    rows,_,indices=join._join_split(blinded,records,[None]*len(blinded),'sealed')
    sample=a.relation_sample_mask(rows)
    if int(sample.sum())!=3840:raise ValueError('F_RELATION_SAMPLE_CENSUS_MISMATCH')
    frozen=json.loads((root/'config/F/model-and-router-freeze.json').read_bytes())
    calculated={};comparisons=[]
    for slot in ('E01','E02'):
        predictions={}
        for arm in a.FEATURE_ARMS:
            array=np.load(root/'data/F/prediction-freeze/relation'/f'{slot}-{arm}.npy',allow_pickle=False,mmap_mode='r')
            if array.shape!=(322560,4) or array.dtype!=np.dtype('<f4'):raise ValueError('FROZEN_PROBABILITY_LAYOUT_MISMATCH')
            predictions[arm]=array[indices]
        calculated[slot]=a.analyze_relation(rows,predictions,frozen['per_encoder'][slot]['relation'])
        comparisons.extend(compare(calculated[slot],result['per_encoder'][slot]['experiment_B'],'/per_encoder/'+slot+'/experiment_B'))
    sample_rows=[{'row_id':rows[i]['row_id'],'family_id':rows[i]['family_id'],'truth':rows[i]['relation'],'blinded_ordinal':int(indices[i])} for i,use in enumerate(sample) if use]
    return {'coverage':'F_EXPERIMENT_B_ONLY_WITH_HASH_VERIFIED_PRIMITIVE_JOIN','comparisons':comparisons,'recalculated':calculated,
        'sample_row_mapping':sample_rows,'excluded':'full routing budget-match and accepted serial comparator need F feature rows; no fits or model inference'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify','C-primary','D-sealed','E-sealed','F-summary','F-sampled-relations'))
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--result',type=Path);p.add_argument('--records',type=Path);p.add_argument('--blinded',type=Path)
    args=p.parse_args();verify(args.root)
    if args.command=='verify': print(json.dumps(verify(args.root)));return 0
    if not args.result:p.error('--result requires a hash-bound scientific projection')
    study=args.command[0];expected=json.loads((args.root/'config'/study/'RESULT-BINDING.json').read_bytes())
    if fingerprint(args.result)!=(expected['bytes'],expected['sha256']):raise ValueError('RESULT_PROJECTION_HASH_MISMATCH')
    result=json.loads(args.result.read_bytes())
    if args.command=='C-primary': value=primary_c(args.root,result)
    elif study in 'DE':value=components(args.root,result,study,args.records or args.root/'gold'/study)
    elif args.command=='F-sampled-relations':value=sample_f(args.root,result,args.records or args.root/'gold/F',args.blinded)
    else:value=summary_f(args.root,result)
    value['fresh_model_execution']=False;value['full_replication']=False
    print(json.dumps(value,sort_keys=True,allow_nan=False))
    return 1 if value['comparisons'] else 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,FileNotFoundError) as e: print(str(e),file=sys.stderr);raise SystemExit(2)
