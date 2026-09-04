#!/usr/bin/env python3
"""Ablate tail corrections from equal-budget proportional ranking on CPU."""

from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
from typing import Any
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/geometria_proporcional")]
import run_proportional_graph_equal_budget_ranking_diagnostic as ranking  # noqa: E402
import run_proportional_graph_fresh_mixed_gate as mixed  # noqa: E402
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_selected_action_transport_power_audit as transport  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_mean_only_ranking_ablation_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_ONLY_RANKING_ABLATION_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_mean_only_ranking_ablation_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_mean_only_ranking_ablation.py",
    "experiments/geometria_proporcional/run_proportional_graph_equal_budget_ranking_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_calibration_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
CONTRASTS = ("topology_tail_minus_mean_only", "constant_tail_minus_mean_only", "topology_tail_minus_constant_tail", "mean_only_minus_identity", "mean_only_minus_public_base_tail", "mean_only_minus_permuted_tail")
CELLS = ranking.CELLS

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def load_config(path: Path) -> dict[str, Any]:
    cfg=json.loads(path.read_text()); expected={"schema_version","source_equal_budget_ranking","source_manifest_sha256","arms","alphas","budget_fractions","control_replicates","numerical_zero_tolerance","execution"}
    if set(cfg)!=expected or cfg["schema_version"]!="proportional-graph-mean-only-ranking-ablation-v1": raise ValueError("invalid mean-only ablation schema")
    if cfg["arms"]!=["raw_generic","raw_typed","closure_generic","closure_typed"] or cfg["alphas"]!=[0.0,0.25,0.5,0.75,1.0]: raise ValueError("factorial changed")
    if cfg["budget_fractions"]!=[0.01,0.02,0.05,0.10,0.20,0.40] or cfg["control_replicates"]!=16: raise ValueError("ranking contract changed")
    if cfg["numerical_zero_tolerance"]!=1e-12 or cfg["execution"]!={"max_seconds":300,"max_rss_gib":4.0}: raise ValueError("execution contract changed")
    return cfg

def source_root(cfg): return ROOT / cfg["source_equal_budget_ranking"]
def verify_source(cfg):
    root=source_root(cfg)
    if sha(root/"manifest.json")!=cfg["source_manifest_sha256"]: raise AssertionError("R366 manifest mismatch")
    manifest=json.loads((root/"manifest.json").read_text())
    for rel,h in manifest["deterministic_files"].items():
        if sha(root/rel)!=h: raise AssertionError(f"R366 source mismatch: {rel}")
    r366=ranking.load_config(root/"resolved_config.json"); ranking.verify_source(r366); return r366

def require_clean(development):
    if not development and subprocess.run(["git","status","--porcelain"],cwd=ROOT,capture_output=True,text=True,check=True).stdout: raise RuntimeError("official ablation requires clean worktree")
def git_head(): return subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()

def ablation_scores(reconstructed, arm_index):
    base=ranking.common_proposal_scores(reconstructed,arm_index); proposal=base["proposal"]; state=reconstructed["state"]; mu=state["mu"][arm_index]
    base["mean_only"]=ranking.selected_component(mu,proposal)
    base["constant_tail"]=ranking.selected_component(mu+state["predicted"]["constant_selected_action"][arm_index],proposal)
    return base

def summarize_cohort(cfg,name,reconstructed):
    report={"cohort":name,"n_masters":int(reconstructed["bootstrap"].shape[1]),"arms":{}}; effects={"bootstrap_indices":reconstructed["bootstrap"]}; packed={}
    for ai,arm in enumerate(cfg["arms"]):
        scores=ablation_scores(reconstructed,ai); proposal=scores["proposal"]; quotient=reconstructed["data"]["quotient_rmse"][ai]
        for key,val in scores.items(): packed[f"{arm}|score|{key}"]=val
        ar={"budgets":{},"budget_average":{}}; aggregate={cell:{c:[] for c in CONTRASTS} for cell in CELLS}
        for frac in cfg["budget_fractions"]:
            label=f"{frac:.2f}"; actions={p:ranking.equal_budget_action(scores[p],proposal,frac) for p in ("mean_only","constant_tail","public_base","topology")}
            controls=np.stack([ranking.equal_budget_action(scores["topology_permuted"][i],proposal,frac) for i in range(cfg["control_replicates"])])
            for p,a in actions.items(): packed[f"{arm}|budget={label}|action|{p}"]=a
            packed[f"{arm}|budget={label}|action|permuted_tail"]=controls
            values={p:mixed.selected_values(quotient,a) for p,a in actions.items()}; values["permuted_tail"]=np.mean([mixed.selected_values(quotient,a) for a in controls],axis=0); values["identity"]=quotient[:,0]
            br={"total_action_budget":int(np.count_nonzero(actions["mean_only"])),"overlap":{p:ranking.jaccard(actions["mean_only"],actions[p]) for p in ("constant_tail","public_base","topology")},"cells":{}}
            for cell in CELLS:
                v={p:ranking.policy_cell(x,cell) for p,x in values.items()}; contrasts={
                    "topology_tail_minus_mean_only":v["topology"]-v["mean_only"], "constant_tail_minus_mean_only":v["constant_tail"]-v["mean_only"],
                    "topology_tail_minus_constant_tail":v["topology"]-v["constant_tail"], "mean_only_minus_identity":v["mean_only"]-v["identity"],
                    "mean_only_minus_public_base_tail":v["mean_only"]-v["public_base"], "mean_only_minus_permuted_tail":v["mean_only"]-v["permuted_tail"]}
                br["cells"][cell]={c:ranking.interval(x,reconstructed["bootstrap"]) for c,x in contrasts.items()}
                for c,x in contrasts.items(): effects[f"{name}|{arm}|budget={label}|{cell}|{c}"]=x; aggregate[cell][c].append(x)
            ar["budgets"][label]=br
        for cell in CELLS:
            ar["budget_average"][cell]={}
            for c in CONTRASTS:
                x=np.mean(aggregate[cell][c],axis=0); ar["budget_average"][cell][c]=ranking.interval(x,reconstructed["bootstrap"]); effects[f"{name}|{arm}|budget_average|{cell}|{c}"]=x
        report["arms"][arm]=ar
    return report,effects,packed

def build_analysis(cfg,reports):
    out={"status":"OPENED_MEAN_ONLY_RANKING_ABLATION","primary_estimand":"topology_tail minus mean_only; positive favors mean_only","arms":{}}
    counts={"MEAN_ONLY_BETTER_BOTH":0,"TAIL_BETTER_BOTH":0,"SIGN_UNSTABLE":0,"IDENTITY_OR_NUMERICAL_ZERO":0}
    for arm in cfg["arms"]:
        ar={"budgets":{},"budget_average":{}}
        for frac in cfg["budget_fractions"]:
            label=f"{frac:.2f}"; ar["budgets"][label]={"cells":{}}
            for cell in CELLS:
                ar["budgets"][label]["cells"][cell]={}
                for c in CONTRASTS:
                    rows={co:reports[co]["arms"][arm]["budgets"][label]["cells"][cell][c] for co in ("A","B")}; generic=transport.classify_transport(rows["A"]["mean"],rows["B"]["mean"],cfg["numerical_zero_tolerance"])
                    label_class=generic
                    if c=="topology_tail_minus_mean_only":
                        label_class={"ADVERSE_BOTH":"MEAN_ONLY_BETTER_BOTH","FAVORABLE_BOTH":"TAIL_BETTER_BOTH"}.get(generic,generic); counts[label_class]+=1
                    ar["budgets"][label]["cells"][cell][c]={"classification":label_class,"cohorts":rows}
        for cell in CELLS:
            ar["budget_average"][cell]={}
            for c in CONTRASTS:
                rows={co:reports[co]["arms"][arm]["budget_average"][cell][c] for co in ("A","B")}; generic=transport.classify_transport(rows["A"]["mean"],rows["B"]["mean"],cfg["numerical_zero_tolerance"])
                if c=="topology_tail_minus_mean_only": generic={"ADVERSE_BOTH":"MEAN_ONLY_BETTER_BOTH","FAVORABLE_BOTH":"TAIL_BETTER_BOTH"}.get(generic,generic)
                ar["budget_average"][cell][c]={"classification":generic,"cohorts":rows}
        out["arms"][arm]=ar
    out["primary_transport_counts_over_72_arm_budget_cells"]=counts; return out

def write_replay(output,cfg,head):
    checks="\n".join(f"printf '%s  %s\\n' '{sha(ROOT/r)}' \"$repo/{r}\" | sha256sum -c -" for r in SOURCE_FILES); checks+=f"\nprintf '%s  %s\\n' '{sha(source_root(cfg)/'manifest.json')}' \"$repo/{cfg['source_equal_budget_ranking']}/manifest.json\" | sha256sum -c -"
    text=f'''#!/bin/sh\nset -eu\nrepo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)\n[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2\n[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2\n{checks}\n: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"\nexec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_mean_only_ranking_ablation.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_mean_only_ranking_ablation_v1.json" --output "$OUTPUT_DIR"\n'''; (output/"replay.sh").write_text(text); (output/"replay.sh").chmod(0o755)

def run(cfg,output,development):
    require_clean(development)
    if output.exists(): raise FileExistsError(output)
    if not development and (ROOT/"data/geometria_proporcional").resolve() not in output.resolve().parents: raise ValueError("official output path invalid")
    start=time.monotonic(); r366=verify_source(cfg); source_cfg=transport.load_config(ranking.source_root(r366)/"resolved_config.json"); output.mkdir(parents=True)
    rec={"A":transport.reconstruct_a(source_cfg),"B":transport.reconstruct_b(source_cfg)}; reports={}; all_effects={}
    for name in ("A","B"):
        reports[name],eff,pack=summarize_cohort(cfg,name,rec[name]); all_effects.update(eff); write_json(output/f"cohort_{name.lower()}_summary.json",reports[name]); frozen.save_npz(output/f"cohort_{name.lower()}_ablation.npz",pack)
    frozen.save_npz(output/"effects_by_master.npz",all_effects); analysis=build_analysis(cfg,reports); analysis["cohort_b_exact_r364_reproduction"]=bool(rec["B"]["exact_reproduction"]); write_json(output/"analysis.json",analysis)
    write_json(output/"environment.json",{"python":platform.python_version(),"numpy":importlib.metadata.version("numpy"),"scipy":importlib.metadata.version("scipy"),"scikit_learn":importlib.metadata.version("scikit-learn"),"threads":1,"cuda_visible_devices":"","refit":False,"new_views":0,"new_solves":0,"gpu_queried":False}); write_json(output/"resolved_config.json",cfg); write_replay(output,cfg,git_head())
    runtime={"elapsed_seconds":time.monotonic()-start,"max_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    if runtime["elapsed_seconds"]>cfg["execution"]["max_seconds"] or runtime["max_rss_gib"]>cfg["execution"]["max_rss_gib"]: raise RuntimeError("resource budget exceeded")
    write_json(output/"runtime_observation.json",runtime); files=sorted(p for p in output.rglob("*") if p.is_file() and p.name not in {"manifest.json","runtime_observation.json"}); write_json(output/"manifest.json",{"schema_version":cfg["schema_version"],"git_head":git_head(),"source_manifest_sha256":cfg["source_manifest_sha256"],"source_hashes":{r:sha(ROOT/r) for r in SOURCE_FILES},"deterministic_files":{str(p.relative_to(output)):sha(p) for p in files},"runtime_exclusions":["runtime_observation.json"]}); print(json.dumps(runtime,sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,default=DEFAULT_CONFIG); p.add_argument("--output",type=Path,required=True); p.add_argument("--development",action="store_true"); a=p.parse_args(); run(load_config(a.config.resolve()),a.output.resolve(),a.development)
if __name__=="__main__": main()
