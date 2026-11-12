from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from experiments.omniglot_signature_v1.signatures import pair_cloud,zero_forest,complete_partition
from experiments.omniglot_signature_v1.gpu_neighbors import GpuNeighbors
from experiments.sharing_structure_research_v1.blind_contract import save_json,digest
root=Path('artifacts/local/omniglot_signature_2026-11-12/study')
output=Path('artifacts/omniglot_signature_2026-11-12')
source=root/'source/signature_spaces/a60_votes_3.npz'
arrays=np.load(source);mask=arrays['mask'];basis=arrays['basis'][:,:7]
rows,columns=np.where(mask);first,second,ranges,signatures=pair_cloud(basis,columns)
norms=np.linalg.norm(signatures,axis=1);threshold=.02517273005937685
adjacency=np.eye(len(rows),dtype=bool);zero=norms<=threshold
adjacency[first[zero],second[zero]]=True
nonzero=np.flatnonzero(~zero)
started=time.perf_counter()
matches=cKDTree(signatures[nonzero]).query_pairs(2*threshold,output_type='ndarray')
left,right=nonzero[matches[:,0]],nonzero[matches[:,1]]
adjacency[first[left],first[right]]=True;adjacency[second[left],second[right]]=True
reference_count,reference=connected_components(csr_matrix(adjacency),directed=False)
reference_seconds=time.perf_counter()-started
maxima=[]
for neuron in np.unique(columns):
 positions=np.flatnonzero(columns==neuron);a,b=ranges[positions[0],0],ranges[positions[-1],1]
 maxima.append(norms[a:b].max(initial=0))
cloud=first,second,ranges,signatures,norms,GpuNeighbors(signatures,'cuda:7',norms),np.asarray(maxima)
started=time.perf_counter();recovered,detail=complete_partition(basis,columns,threshold,[8,16],cloud,zero_forest(basis,columns))
def canon(labels):
 rename={};return np.asarray([rename.setdefault(int(x),len(rename)) for x in labels])
result={'source_sha256':digest(source),'rank':7,'threshold':threshold,'active_positions':len(rows),'ordered_pairs':len(first),'literal_equal_pair_matches':len(matches),'reference_groups':int(reference_count),'streamed_groups':None if recovered is None else int(recovered.max()+1),'same_partition':recovered is not None and np.array_equal(canon(reference),canon(recovered)),'reference_seconds':reference_seconds,'streamed_seconds':time.perf_counter()-started,'streamed_diagnostic':detail,'target_access':False}
save_json(output/'merge_rule_equivalence.json',result)
print(result,flush=True)
