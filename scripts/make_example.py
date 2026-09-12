"""Small artificial smoke input, not a paper benchmark or real tissue."""
from pathlib import Path
import argparse
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('Output exists')
rng=np.random.RandomState(11);n=120
edge=np.array([(i,j) for i in range(n) for j in [(i+1)%n,(i-1)%n]],dtype=np.int64).T
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,view_0=rng.normal(size=(n,30)).astype('float32'),view_1=rng.normal(size=(n,20)).astype('float32'),edge_index=edge)
print(a.output)
