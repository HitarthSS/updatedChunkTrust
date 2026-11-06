"""Anonymous, checksum-verified download of the released GR1 inference package."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import time
import urllib.parse
import urllib.request


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(root):
    source=json.loads((root/'modelscope-files.json').read_text())
    assert source['Code']==200 and source['Success'] is True
    files=sorted((x for x in source['Data']['Files'] if x['Type']=='blob'),key=lambda x:x['Size'])
    package=root/'models/pi05_robocasa_gr1_tabletop24_step30000';package.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock();done={};total=sum(x['Size'] for x in files)
    def status(state,error=None):
        payload=dict(state=state,time=time.time(),completed_files=len(done),total_files=len(files),
            completed_bytes=sum(done.values()),total_bytes=total,error=error)
        p=root/'download-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(p)
    status('downloading')
    def fetch(item):
        rel=PurePosixPath(item['Path']);assert not rel.is_absolute() and '..' not in rel.parts
        assert len(item['Sha256'])==64
        path=package/rel;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==item['Size'] and digest(path)==item['Sha256']:
            with lock:done[item['Path']]=item['Size'];status('downloading')
            return
        query=urllib.parse.urlencode(dict(Revision='master',FilePath=item['Path']))
        url='https://modelscope.cn/api/v1/models/NewGain/pi05_robocasa_gr1_tabletop24/repo?'+query
        tmp=path.with_suffix(path.suffix+'.partial')
        for attempt in range(4):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'Manclab-reproduction'})
                with urllib.request.urlopen(request,timeout=90) as response,tmp.open('wb') as out:
                    for block in iter(lambda:response.read(4*1024*1024),b''):out.write(block)
                assert tmp.stat().st_size==item['Size'],('size',item['Path'],tmp.stat().st_size,item['Size'])
                assert digest(tmp)==item['Sha256'],('checksum',item['Path'])
                tmp.replace(path)
                with lock:done[item['Path']]=item['Size'];status('downloading')
                print('VERIFIED',item['Path'],item['Size'],flush=True);return
            except Exception:
                if attempt==3:raise
                time.sleep(2*(attempt+1))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(fetch,item) for item in files]):future.result()
        assert len(done)==len(files)
        status('complete')
    except BaseException as exc:status('failed',repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root.resolve())
