#!/usr/bin/env python3
"""Create the fixed 11,257/1,407/1,407 TransVAE release split."""
from __future__ import annotations
import argparse, csv, hashlib, json, random
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    with a.input.open(newline='', encoding='utf-8-sig') as h: rows=list(csv.DictReader(h))
    if len(rows)!=14071: raise ValueError(f'expected 14071 rows, found {len(rows)}')
    for i,row in enumerate(rows,1):
        seq=''.join(row['realB'].upper().split())
        if len(seq)!=165 or set(seq)-set('ACGT'): raise ValueError(f'invalid sequence at row {i}')
        row['record_id']=hashlib.sha256(seq.encode()).hexdigest()
    rng=random.Random(a.seed); order=list(range(len(rows))); rng.shuffle(order)
    sizes=(11257,1407,1407); start=0; manifest={'source':str(a.input),'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'seed':a.seed,'sizes':{}}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,size in zip(('train','validation','test'),sizes):
        selected=[rows[i] for i in order[start:start+size]]; start+=size
        out=a.output_dir/f'{name}.csv'
        fields=['record_id']+[k for k in rows[0] if k!='record_id']
        with out.open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(selected)
        manifest['sizes'][name]=len(selected); manifest[f'{name}_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
