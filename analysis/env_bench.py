"""Exercise 64 real sandboxes' setup/cleanup with the local runtime patches."""
import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
import tomllib
import uuid
from pathlib import Path

DA = Path(__file__).resolve().parents[1]
SMOL = DA.parent / "SmolDataEnvs"
sys.path[:0] = [str(DA), str(SMOL)]

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", action="store_true")
    p.add_argument("--direct-exec", action="store_true")
    p.add_argument("--ops-limit", type=int, default=0)
    p.add_argument("--network", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tmp = SMOL / "runs/tmp"
    tmp.mkdir(exist_ok=True, parents=True)
    os.environ['TMPDIR'] = str(tmp)
    tempfile.tempdir = str(tmp)
    from serve_harbor import install_fast_setup, install_fast_teardown, install_fast_metadata, install_direct_exec, install_docker_limit, install_shared_network
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    install_fast_setup()
    install_fast_teardown()
    if a.metadata:
        install_fast_metadata()
    if a.direct_exec:
        install_direct_exec()
    if a.ops_limit:
        install_docker_limit(a.ops_limit)
    if a.network:
        install_shared_network(a.network)
    names = json.loads((DA / 'manifests/train1000.json').read_text())[:8]
    job = uuid.uuid4().hex[:8]
    async def one(k):
        task = SMOL / 'prepared/datasets/train/tasks' / names[k//8]
        cfg = EnvironmentConfig(**tomllib.loads((task/'task.toml').read_text())['environment'])
        # Cached healthchecks never download; avoid expanding any external credential.
        cfg.env['HF_TOKEN'] = ''
        trial = SMOL / 'runs/codex_env_bench' / f'{job}-{k}'
        mounts = []
        for name in ['agent', 'verifier', 'artifacts']:
            (trial/name).mkdir(exist_ok=True, parents=True)
            mounts.append({'type':'bind','source':str(trial/name),'target':f'/logs/{name}'})
        env = DockerEnvironment(environment_dir=task/'environment', environment_name=names[k//8],
            session_id=f'codex-bench-{job}-{k}__env', trial_paths=TrialPaths(trial_dir=trial),
            task_env_config=cfg, mounts=mounts)
        started = time.monotonic()
        try:
            await env.start(False)
            check = await env.exec('python3 /opt/pull_bucket.py && test -f /home/user/input/.complete')
            assert check.return_code == 0, 'cached healthcheck failed'
            setup = time.monotonic()-started
            stopped = time.monotonic()
        finally:
            await env.stop(True)
        return {'setup_s':setup,'teardown_s':time.monotonic()-stopped}
    t0 = time.monotonic()
    rows = await asyncio.gather(*(one(k) for k in range(64)))
    def summary(key):
        v = sorted(r[key] for r in rows)
        return {'mean':round(statistics.mean(v),2),'median':round(statistics.median(v),2),
                'p95':round(v[int(.95*len(v))],2),'max':round(v[-1],2)}
    result = {'sandboxes':64,'fast_metadata':a.metadata,'direct_exec':a.direct_exec,'ops_limit':a.ops_limit,'network':a.network,'gomaxprocs':os.environ.get('GOMAXPROCS'),'wall_s':round(time.monotonic()-t0,2),
              'setup_s':summary('setup_s'),'teardown_s':summary('teardown_s')}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    asyncio.run(main())
