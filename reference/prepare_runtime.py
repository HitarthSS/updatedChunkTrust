"""Create isolated evaluation environments; never modify the shared base runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main(root):
    base=Path('/home/hitarth/rpent-reproduction/.venv')
    env=os.environ.copy()
    for key in ('OPENAI_API_KEY','OPENAI_API_TOKEN_FILE','OPENAI_RESPONSES_URL','ANTHROPIC_API_KEY','HF_TOKEN'):
        env.pop(key,None)
    env.update(GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
        WANDB_MODE='disabled',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    shared=next(base.glob('lib/python*/site-packages'))
    policy=['jax==0.5.0','jaxlib==0.5.0','jax-cuda12-plugin==0.5.0','jax-cuda12-pjrt==0.5.0',
        'flax==0.10.2','orbax-checkpoint==0.11.1','equinox==0.11.10','optax==0.2.4','chex==0.1.88',
        'tensorstore==0.1.71','transformers==4.48.1','tokenizers==0.21.0','websockets==14.1',
        'jaxtyping==0.2.36','ml-collections==1.0.0',
        'lerobot @ git+https://github.com/huggingface/lerobot.git@a445d9c9da6bea99a8972daa4fe1fdd053d711d2']
    sim=['robosuite==1.5.1','mujoco==3.2.6','gymnasium==0.29.1','numba==0.61.2','llvmlite==0.44.0',
        'numpy==1.26.4','pygame==2.6.1','tianshou==0.5.1']
    def status(state,error=None):
        (root/'runtime-status.json').write_text(json.dumps(dict(state=state,time=time.time(),error=error),indent=2)+'\n')
    try:
        for name,requirements in (('policy',policy),('sim',sim)):
            status('installing_'+name)
            folder=root/f'.venv-{name}'
            if not folder.exists():subprocess.run([str(base/'bin/python'),'-m','venv',str(folder)],check=True,env=env)
            site=next(folder.glob('lib/python*/site-packages'))
            (site/'shared_readonly_dependencies.pth').write_text(str(shared)+'\n')
            subprocess.run([str(folder/'bin/python'),'-m','pip','install','--no-deps',*requirements],check=True,env=env)
            subprocess.run([str(folder/'bin/python'),'-m','pip','install','--no-deps','-e',str(root/'ChunkTrust')],check=True,env=env)
            if name=='sim':
                subprocess.run([str(folder/'bin/python'),'-m','pip','install','--no-deps','-e',str(root/'workspaces/robocasa-gr1-tabletop-tasks')],check=True,env=env)
        status('complete')
    except BaseException as exc:status('failed',repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root.resolve())
