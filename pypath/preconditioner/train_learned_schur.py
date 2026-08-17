import argparse, json, os, sys
from typing import Any
import numpy as np, torch
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from pypath.preconditioner.block_schwarz import BlockPlanConfig, build_block_schwarz_plan
from pypath.preconditioner.schur_interface import ExplicitSchurInterfacePreconditioner, SCHUR_ROW_FEATURE_DIM, SchurDiagonalScaleNet, build_schur_row_features
from pypath.utils.external_gmres_prototype import _load_trajectory_linear_system_steps,_make_system_matrix,_parse_circuit_ids

def jd(v:Any):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(np.floating,)): return float(v)
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    return str(v)

def probes(rhs,res,n,seed):
    out=[]
    if rhs.size: out.append(rhs.astype(np.float64))
    if res.size and np.linalg.norm(res)>0: out.append(res.astype(np.float64))
    rng=np.random.default_rng(seed)
    for _ in range(n): out.append(rng.standard_normal(rhs.shape[0]).astype(np.float64))
    return out

def load_samples(a):
    ss=[]
    for cid in _parse_circuit_ids(a.circuit_ids):
        net=os.path.join(a.netlist_dir,f'{cid}.sp')
        c=_load_trajectory_linear_system_steps(trajectory_dir=a.trajectory_dir,circuit_id=int(cid),netlist_path=net)
        steps=c.get('steps',[])[a.step_offset:]
        if a.max_steps_per_circuit>0: steps=steps[:a.max_steps_per_circuit]
        for st in steps:
            A=_make_system_matrix(st,apply_gmin_diagonal=not a.disable_gmin_diagonal)
            kw=dict(matrix=A,node_map=st.get('node_map',{}),netlist_path=net)
            cfg=dict(max_block_size=a.max_block_size,min_block_size=a.min_block_size,max_blocks=a.max_blocks,max_total_block_nnz=a.max_total_block_nnz,uncovered_row_policy=a.uncovered_row_policy)
            cp=build_block_schwarz_plan(**kw,config=BlockPlanConfig(block_mode='cell_core',**cfg))
            bp=build_block_schwarz_plan(**kw,config=BlockPlanConfig(block_mode='cell_core_plus_onehop_boundary',**cfg))
            ex=ExplicitSchurInterfacePreconditioner(matrix=A,core_plan=cp,boundary_plan=bp,uncovered_row_policy=a.uncovered_row_policy)
            if ex.interface_rows.shape[0]==0: continue
            d=np.diag(ex.schur_matrix)
            ss.append(dict(cid=int(cid),ex=ex,feat=build_schur_row_features(ex),inv=np.sign(d)/np.maximum(np.abs(d),a.eps),rhs=np.asarray(st.get('rhs',[]),float),res=np.asarray(st.get('raw_residual',[]),float)))
    return ss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trajectory-dir',required=True); ap.add_argument('--netlist-dir',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--circuit-ids',default='0-7'); ap.add_argument('--step-offset',type=int,default=0); ap.add_argument('--max-steps-per-circuit',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--hidden-dim',type=int,default=32); ap.add_argument('--log-scale-clip',type=float,default=4.0)
    ap.add_argument('--gaussian-probes',type=int,default=4); ap.add_argument('--seed',type=int,default=1); ap.add_argument('--max-block-size',type=int,default=32); ap.add_argument('--min-block-size',type=int,default=2)
    ap.add_argument('--max-blocks',type=int,default=0); ap.add_argument('--max-total-block-nnz',type=int,default=0); ap.add_argument('--uncovered-row-policy',default='row_sum'); ap.add_argument('--disable-gmin-diagonal',action='store_true'); ap.add_argument('--eps',type=float,default=1e-30)
    a=ap.parse_args(); os.makedirs(a.output_dir,exist_ok=True); ss=load_samples(a)
    if not ss: raise RuntimeError('no samples')
    allf=np.concatenate([s['feat'] for s in ss],0); mean=allf.mean(0); std=np.maximum(allf.std(0),1e-12)
    m=SchurDiagonalScaleNet(SCHUR_ROW_FEATURE_DIM,a.hidden_dim,a.log_scale_clip); m.set_feature_stats(mean,std); opt=torch.optim.Adam(m.parameters(),lr=a.lr); log=[]
    for ep in range(a.epochs):
        tot=0.; cnt=0
        for si,s in enumerate(ss):
            ex=s['ex']; feat=torch.as_tensor(s['feat'],dtype=torch.float64); inv=torch.as_tensor(s['inv'],dtype=torch.float64)
            opt.zero_grad(set_to_none=True); scale=m(feat); loss=torch.zeros((),dtype=torch.float64); pc=0
            for v in probes(s['rhs'],s['res'],a.gaussian_probes,a.seed+1009*ep+si):
                core=ex._apply_core_only(v); rb=v[ex.interface_rows]-ex.matrix[np.ix_(ex.interface_rows,ex.core_rows)].dot(core[ex.core_rows]); tgt=ex.schur_factor.dot(rb)
                rb=torch.as_tensor(rb,dtype=torch.float64); tgt=torch.as_tensor(tgt,dtype=torch.float64); pred=scale*inv*rb
                loss=loss+torch.sum((pred-tgt)**2)/torch.sum(tgt**2).clamp_min(a.eps); pc+=1
            loss=loss/max(pc,1); loss.backward(); opt.step(); tot+=float(loss.detach()); cnt+=1
        row={'epoch':ep+1,'mean_teacher_loss':tot/max(cnt,1)}; log.append(row); print(json.dumps(row),flush=True)
    ck=os.path.join(a.output_dir,'learned_schur_diagonal.pt')
    torch.save({'model_kind':'learned_schur_diagonal','model_config':{'feature_dim':SCHUR_ROW_FEATURE_DIM,'hidden_dim':a.hidden_dim,'log_scale_clip':a.log_scale_clip},'model_state_dict':m.state_dict(),'feature_mean':mean,'feature_std':std,'training_args':vars(a),'training_log':log,'sample_count':len(ss)},ck)
    json.dump({'checkpoint':ck,'sample_count':len(ss),'final_mean_teacher_loss':log[-1]['mean_teacher_loss'],'training_log':log},open(os.path.join(a.output_dir,'training_summary.json'),'w'),indent=2,default=jd); print('checkpoint='+ck)
if __name__=='__main__': main()
