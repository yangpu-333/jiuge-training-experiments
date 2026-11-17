"""Combine six same-layout paired E2E results; no GPU or network."""
from pathlib import Path
import csv,json,math,sys
B=Path('experiments/jiuge_training');P=B/'results/eight-gpu-plan-20261113'
A=Path(sys.argv[1]) if len(sys.argv)>1 else P/'host-pool-final-20261116'
D=Path(sys.argv[2]) if len(sys.argv)>2 else P/'host-pool-stage2-final-20261116'
read=lambda p:json.loads(p.read_text())
assert read(A/'independent-result-review.json')['passed'] and read(D/'independent-result-review.json')['passed']
rows=[]
for case in ['pp4-tp2-cpt','pp4-tp2-lora','pp2-tp4-cpt','pp2-tp4-lora','tp8-cpt','tp8-lora']:
    first=case.startswith('pp4')
    folder=A if first else D
    file=folder/(case+'-e2e-summary.json')
    if not file.exists():file=folder/'evidence'/(case+'-e2e-summary.json')
    s=read(file)
    c=s['comparisons']['baseline_vs_host-only'] if first else s
    means=s['mean_seconds'];task='CPT' if case.endswith('cpt') else 'LoRA'
    target_tput=15 if task=='CPT' else 17;target_time=12 if task=='CPT' else 15
    ratio=means['baseline']/means['host-only'];threshold=max(1+target_tput/100,1/(1-target_time/100))
    ci=c['paired_bootstrap95_speed_ratio']
    r={'case':case,'parallel':'TP2/PP4' if case.startswith('pp4') else ('TP4/PP2' if case.startswith('pp2') else 'TP8/PP1'),'task':task,
       'baseline_seconds':means['baseline'],'optimized_seconds':means['host-only'],'speed_ratio':ratio,
       'throughput_gain_percent':100*(ratio-1),'time_reduction_percent':100*(1-1/ratio),'speed_ratio_ci95_low':ci[0],'speed_ratio_ci95_high':ci[1],
       'target_throughput_percent':target_tput,'target_time_reduction_percent':target_time,'mean_targets_passed':ratio>=threshold,'confidence_targets_passed':ci[0]>=threshold,
       'throughput_gap_percentage_points':max(0,target_tput-100*(ratio-1)),'time_gap_percentage_points':max(0,target_time-100*(1-1/ratio)),
       'additional_optimized_seconds_to_reduce_for_mean_targets':max(0,means['host-only']-means['baseline']/threshold),
       'additional_optimized_time_reduction_percent_needed':max(0,100*(1-ratio/threshold)),
       'faster_pairs_or_cycles':c['faster_cycles'] if first else c['faster_pairs']}
    rows.append(r)
result={'configuration':'8BI-V150/CoreX5/4B BF16 S512 MBS1 GBS16 DP1 seed1234 no recompute',
    'measurement':'50step complete process start/load/train/full native checkpoint/fsync/exit;container local storage;CRC/SHA outside timer;no profiler',
    'baseline':'same-layout same-round attention-only adapted local with reference RMSNorm;not unmodified official native model',
    'rows':rows,'mean_pass_cases':sum(r['mean_targets_passed'] for r in rows),'confidence_pass_cases':sum(r['confidence_targets_passed'] for r in rows),
    'valid_performance_jobs':84,'numeric_arms':14,'native_restore_pairs':12,'paired_rank_steps_original_scalar_gate':19200,
    'scope_limit':'first-stageTP2PP4 has3arms,second-stageTP4PP2/TP8 has2arms.Total gains refer to own same-round baseline;onlyTP2PP4 isolated compiler increment.No longtraining quality/new50-to51audit.'}
(D/'all-layouts-e2e-summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (D/'all-layouts-e2e-summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(result,indent=2))
