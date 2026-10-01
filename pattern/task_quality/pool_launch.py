"""Launch the two pooling runs, with owned process tracking and snapshots."""
from pathlib import Path
import json,os,subprocess,sys,time,hashlib,signal,argparse
from .repaired_launch import GPU_UUIDS
parser=argparse.ArgumentParser();parser.add_argument('--bounded',action='store_true');args=parser.parse_args()
variant='functional_set_pool_bounded' if args.bounded else 'functional_set_pool'
root=Path('pattern/outputs/task_quality_debug_20261001').resolve()
folder=root/('orchestration/pool_meta_bounded' if args.bounded else 'orchestration/pool_meta');folder.mkdir(parents=True,exist_ok=True)
snap=folder/'source';snap.mkdir(exist_ok=True)
hashes={}
for f in Path(__file__).parent.glob('*.py'):
    data=f.read_bytes();(snap/f.name).write_bytes(data);hashes[f.name]=hashlib.sha256(data).hexdigest()
procs=[]
def stop(sig,frame):
    for p,_,_ in procs:
        if p.poll() is None:p.terminate()
    raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
    jobs=[]
    for seed,gpu in zip((8100,8102),GPU_UUIDS):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        cmd=[sys.executable,'-m','pattern.task_quality.pool_run','--root',str(root),'--seed',str(seed),'--variant',variant,'--device','cuda']
        log=(folder/f'{seed}.log').open('a');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        procs.append((p,log,seed));jobs.append({'seed':seed,'gpu_uuid':gpu,'pid':p.pid,'command':cmd})
    (folder/'allocation.json').write_text(json.dumps({'approved_gpu_indices':[1,2],'jobs':jobs,'source_sha256':hashes},indent=2)+'\n')
    while any(p.poll() is None for p,_,_ in procs):
        states=[{'seed':seed,'pid':p.pid,'returncode':p.poll()} for p,_,seed in procs]
        (folder/'progress.json').write_text(json.dumps(states,indent=2)+'\n')
        if any(s['returncode'] not in (None,0) for s in states):raise RuntimeError('Pooling worker failed')
        time.sleep(5)
    assert all(p.returncode==0 for p,_,_ in procs)
    (folder/'completed.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
    print('Pooling meta completed',flush=True)
finally:
    for p,log,_ in procs:
        if p.poll() is None:p.terminate()
        p.wait();log.close()
