"""Train a two-view embedding from aligned processed arrays."""
import argparse,json,os,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=['misar','tonsil'],default='misar')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--full-graph',action='store_true',help='Small smoke tests only; changes training protocol')
    a=parser.parse_args()
    if a.epochs is not None and a.epochs <= 0:
        parser.error('--epochs must be positive')
    if a.output.exists():parser.error('Use a new output directory; existing results are never overwritten.')
    if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    if a.profile == 'tonsil':
        from .tonsil import RunConfig, train_model
    else:
        from .common_spatial_stable import RunConfig, train_model
    d=np.load(a.input,allow_pickle=False)
    keys=sorted((k for k in d.files if k.startswith('view_')),key=lambda x:int(x[5:]))
    if keys != ['view_0', 'view_1']:parser.error('Expected view_0 and view_1, plus edge_index shaped [2,E].')
    x=[np.asarray(d[k],dtype=np.float32) for k in keys];edge=np.asarray(d['edge_index'])
    if any(z.ndim!=2 or len(z)!=len(x[0]) or not np.isfinite(z).all() for z in x):parser.error('Invalid aligned feature matrices.')
    if edge.ndim!=2 or edge.shape[0]!=2 or not np.issubdtype(edge.dtype,np.integer) or edge.size==0 or edge.min()<0 or edge.max()>=len(x[0]):parser.error('Invalid edge_index.')
    if a.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA not available; use --device cpu for a small test.')
    a.output.mkdir(parents=True)
    tonsil=a.profile=='tonsil'
    config=RunConfig(variant='portable_processed',encoder='SAGE',sampling='full' if a.full_graph else 'cluster',
        weights=(.5,.5),selected_views=(0,1),graph_weight=0,spatial_weight=1 if tonsil else .25,geometry_weight=.05 if tonsil else 1)
    z,meta=train_model([torch.from_numpy(v) for v in x],torch.from_numpy(edge.astype(np.int64)),config,a.seed,
        epochs=a.epochs or (100 if tonsil else 50),train_batch_size=1000,infer_batch_size=20000,
        device=torch.device(a.device),emb_dim=64 if tonsil else 32,lr=1e-4 if tonsil else 2e-4,weight_decay=1e-6,partition_dir=a.output/'partitions')
    if not np.isfinite(z).all():raise ValueError('Nonfinite embedding')
    np.save(a.output/'embedding.npy',z)
    meta['scope']='Processed two-view embedding; preprocessing and clustering are separate steps.'
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'shape':list(z.shape),'finite':True,'updates':meta['updates']}))

if __name__=='__main__':main()
