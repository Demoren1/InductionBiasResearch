"""Run bounded paired experiments only on previously approved GPUs 1 and 2."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from .repaired_run import VARIANTS,SEEDS

GPU_UUIDS=('GPU-6784dc4e-6ec9-2266-5d23-85bf1b1c2af3','GPU-f8501f2d-53bc-9087-6041-64ee69876325')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=('meta','tune','test'),default='meta');p.add_argument('--phase',default='')
    a=p.parse_args();root=a.root.resolve();stage=root/'orchestration'/(a.stage+('_'+a.phase if a.phase else ''));stage.mkdir(parents=True,exist_ok=True)
    snap=stage/'source';snap.mkdir(exist_ok=True)
    hashes={}
    for file in Path(__file__).parent.glob('*.py'):
        raw=file.read_bytes();target=snap/file.name
        if target.exists():assert target.read_bytes()==raw,'Source changed after launch'
        else:target.write_bytes(raw)
        hashes[file.name]=hashlib.sha256(raw).hexdigest()
    jobs=[]
    if a.stage=='meta':
        for seed in SEEDS:
            for idx,variant in enumerate(VARIANTS):
                jobs.append({'name':f'{seed}_{variant}','gpu':GPU_UUIDS[idx],
                    'cmd':[sys.executable,'-m','pattern.task_quality.repaired_run','--root',str(root),'--seed',str(seed),'--variant',variant,'--device','cuda']})
    else:
        for idx,seed in enumerate(SEEDS):
            jobs.append({'name':str(seed),'gpu':GPU_UUIDS[idx],
                'cmd':[sys.executable,'-m','pattern.task_quality.repaired_eval',a.stage,'--root',str(root),'--seed',str(seed),'--device','cuda']})
    procs=[]
    def stop(sig,frame):
        for proc,_,_ in procs:
            if proc.poll() is None:proc.terminate()
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for job in jobs:
            log=(stage/(job['name']+'.log')).open('a')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=job['gpu'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            proc=subprocess.Popen(job['cmd'],env=env,stdout=log,stderr=subprocess.STDOUT)
            procs.append((proc,log,job));job['pid']=proc.pid
            print('Started',job['name'],'pid',proc.pid,flush=True)
        (stage/'allocation.json').write_text(json.dumps({'approved_gpu_indices':[1,2],'jobs':jobs,'source_sha256':hashes},indent=2)+'\n')
        while any(proc.poll() is None for proc,_,_ in procs):
            statuses=[{'name':job['name'],'pid':proc.pid,'returncode':proc.poll()} for proc,_,job in procs]
            (stage/'progress.json').write_text(json.dumps(statuses,indent=2)+'\n')
            if any(s['returncode'] not in (None,0) for s in statuses):raise RuntimeError('Worker failed; inspect logs')
            time.sleep(5)
        assert all(proc.returncode==0 for proc,_,_ in procs)
        (stage/'completed.json').write_text(json.dumps({'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'jobs':jobs},indent=2)+'\n')
        print('Completed',a.stage,flush=True)
    finally:
        for proc,log,_ in procs:
            if proc.poll() is None:proc.terminate()
            proc.wait();log.close()

if __name__=='__main__':main()
