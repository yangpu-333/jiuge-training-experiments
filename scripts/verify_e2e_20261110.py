"""Independent whole-task statistics and integrity-receipt review after extraction."""
from pathlib import Path
import argparse,hashlib,json,math,random,statistics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('project_root',type=Path)
    args=parser.parse_args();b=args.project_root;r=b/'checks/20261109-e2e-coalesced/campaign-v2'
    def read(p):return json.loads(p.read_text())
    def close(a,z):assert math.isclose(a,z,rel_tol=1e-12,abs_tol=1e-10),(a,z)
    manifest=read(r/'evidence-file-hashes.json')
    for rel,entry in manifest.items():
        p=b/rel;assert p.resolve().is_relative_to(b.resolve())
        data=p.read_bytes();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],rel
    result=read(r/'complete.json');assert result['completed'] and result['native_restore_and_integrity_passed']
    assert read(r/'integrity-preflight-complete.json')['passed']
    for mode in ['baseline','current','coalesced']:
        control=r/('preflight-control-'+mode);resumed=r/('preflight-resume-'+mode)
        images=read(control/'persisted-state-fingerprints.json')
        for rank in range(4):
            assert read(control/f'saved-state-rank{rank}.json')==read(resumed/f'loaded-state-rank{rank}.json')
            assert read(control/f'final-state-rank{rank}.json')==read(resumed/f'final-state-rank{rank}.json')
            assert read(control/f'final-counters-rank{rank}.json')==read(resumed/f'final-counters-rank{rank}.json')
            assert images[str(rank)]['optimizer']==read(control/f'prewrite-optimizer-rank{rank}.json')==read(resumed/f'loaded-optimizer-cpu-rank{rank}.json')
            rows=[json.loads(s) for s in (control/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
            loaded=[json.loads(s) for s in (resumed/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
            assert len(rows)==11 and rows[-1:]==loaded
    status=read(r/'status.json');assert len(status)==27
    assert all(not row['running'] and row['returncode']==0 for row in status.values())
    cycles=result['cycles'];assert len(cycles)==6
    records=files=pair_checks=0
    for cycle in cycles:
        c=cycle['cycle'];arms={};images={}
        for mode in ['baseline','current','coalesced']:
            out=r/f'cycle{c}-{mode}';cfg=read(out/'config.json')
            assert cfg['task']['name']=='cpt' and cfg['task']['recompute'] is False and cfg['task']['seq_length']==512
            assert cfg['task']['micro_batch_size']==1 and cfg['task']['global_batch_size']==16 and cfg['task']['seed']==1234
            assert cfg['task']['train_iters']==50 and not cfg['task'].get('load')
            assert cfg['parallel']['tp']==cfg['parallel']['pp']==2 and cfg['parallel']['dp']==1
            close(cycle[mode],status[f'cycle{c}-{mode}']['seconds'])
            ds=[read(out/f'e2e-rank{rank}.json') for rank in range(4)]
            for rank,d in enumerate(ds):
                assert d['completed'] and d['automatic_gc_enabled'] and len(d['steps'])==50
                assert d['processed_tokens']==409600 and d['consumed_train_samples']==800
                assert all(s['skipped']==0 for s in d['steps']);records+=50
                rt=read(out/f'checkpoint-runtime-rank{rank}.json')
                assert rt['mode']==mode and len(rt['writes'])==2 and len(rt['calls'])==1
                assert all(w['fsync_and_rename_complete'] and w['crc32_enabled'] for w in rt['writes'])
                if rank==0:assert set(rt['metadata_synced'])=={'jiuge_contract.json','latest_checkpointed_iteration.txt'}
            hs=read(out/'checkpoint-file-hashes.json');assert len(hs)==8 and all(v['zip_crc_passed'] for v in hs.values());files+=8
            arms[mode]=ds;images[mode]=read(out/'persisted-state-fingerprints.json')
        assert images['current']==images['coalesced']
        for left,right in [('baseline','current'),('baseline','coalesced'),('current','coalesced')]:
            for rank in range(4):
                for x,y in zip(arms[left][rank]['steps'],arms[right][rank]['steps']):
                    assert x['iteration']==y['iteration'] and x['losses'].keys()==y['losses'].keys()
                    assert abs(x['grad_norm']-y['grad_norm'])<=1e-12+.01*abs(x['grad_norm'])
                    assert all(abs(x['losses'][k]-y['losses'][k])<=.001+.001*abs(x['losses'][k]) for k in x['losses']);pair_checks+=1
        close(cycle['incremental_ratio'],cycle['current']/cycle['coalesced'])
        close(cycle['total_ratio'],cycle['baseline']/cycle['coalesced'])
    for rank in range(4):
        for mode in ['baseline','current','coalesced']:
            assert read(r/f'cycle6-{mode}/persisted-state-fingerprints.json')[str(rank)]['optimizer']==read(r/f'final-native50to51-{mode}/loaded-optimizer-cpu-rank{rank}.json')
        for name in ['loaded-state','final-state','final-counters']:
            assert read(r/f'final-native50to51-current/{name}-rank{rank}.json')==read(r/f'final-native50to51-coalesced/{name}-rank{rank}.json')
        assert (r/f'final-native50to51-current/step-audit-rank{rank}.jsonl').read_text()==(r/f'final-native50to51-coalesced/step-audit-rank{rank}.jsonl').read_text()
    means={a:statistics.mean(c[a] for c in cycles) for a in ['baseline','current','coalesced']}
    for a,v in means.items():close(v,result['means_seconds'][a]);close(409600/v,result['processed_tokens_per_second'][a])
    def interval(left,right):
        rng=random.Random(1234);xs=[]
        for _ in range(20000):
            sample=rng.choices(cycles,k=6);xs.append(sum(c[left] for c in sample)/sum(c[right] for c in sample))
        xs.sort();return [xs[500],xs[19499]]
    inc=means['current']/means['coalesced'];total=means['baseline']/means['coalesced']
    close(inc,result['incremental_speed_ratio']);close(total,result['total_speed_ratio'])
    close(1-means['coalesced']/means['current'],result['incremental_time_reduction'])
    close(1-means['coalesced']/means['baseline'],result['total_time_reduction'])
    ci=interval('baseline','coalesced');ic=interval('current','coalesced')
    assert ci==result['total_bootstrap95'] and ic==result['incremental_bootstrap95']
    assert (all(c['incremental_ratio']>1 for c in cycles) and inc>=1.005)==result['candidate_incremental_selected']
    assert (total>=1.15 and ci[0]>=1.15 and means['coalesced']/means['baseline']<=.88)==result['cpt_final_target_met']
    output={'passed':True,'member_hashes_verified':len(manifest),'rank_step_records':records,'paired_scalar_checks':pair_checks,'checkpoint_file_crc_receipts':files,
            'means_seconds':means,'incremental_speed_ratio':inc,'total_speed_ratio':total,'incremental_bootstrap95':ic,'total_bootstrap95':ci,
            'scope':'local review of archived full-state/CRC/native-load receipts and independent whole-task statistics; no second execution of remote GPU state checks'}
    (r/'independent-local-review.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
