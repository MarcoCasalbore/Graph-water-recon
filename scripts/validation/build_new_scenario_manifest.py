"""Content-aware classification of Section V source and final artifacts."""
from pathlib import Path
import json
import re

import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/"results_section_v"
FINAL=PROJECT/"results_section_v_final"

RUN_RE=re.compile(r"g(?P<g>\d+)_s(?P<s>\d+)_t(?P<t>\d+)_l(?P<l>\d+)_h(?P<h>\d+).*?(?:a(?P<a>\d+p\d+))?.*?(?:kd(?P<k>\d+))?.*?e(?P<e>\d+)")


def artifact_type(path):
    if path.suffix.lower() in [".png",".pdf",".svg"]: return "figure"
    if path.suffix.lower() in [".csv",".md",".tex"] and "table" in path.name.lower(): return "table"
    if path.name in ["config.json","meta.json"]: return "configuration"
    if path.suffix.lower() in [".pt",".pth"]: return "checkpoint_or_graph_tensor"
    if path.suffix.lower() in [".npy",".npz"]: return "numeric_data"
    if path.suffix.lower()==".log": return "log"
    if path.suffix.lower()==".py": return "reusable_code"
    if path.suffix.lower()==".json": return "metrics_or_metadata"
    return "other"


def nearest_json(path,name):
    for parent in [path.parent,*path.parents]:
        candidate=parent/name
        if candidate.exists():
            try: return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception: return {}
        if parent in [SOURCE.parent,FINAL.parent]: break
    return {}


def classify(path,root):
    rel=path.relative_to(PROJECT).as_posix(); config=nearest_json(path,"config.json"); meta=nearest_json(path,"meta.json")
    text=""
    if path.suffix.lower() in [".json",".md",".csv"] and path.stat().st_size<2_000_000:
        try: text=path.read_text(encoding="utf-8",errors="ignore")[:200000]
        except Exception: pass
    match=RUN_RE.search(rel)
    graph_seed=config.get("graph_seed",meta.get("graph_seed",int(match.group("g")) if match else None))
    signal_seed=config.get("signal_seed",meta.get("signal_seed",int(match.group("s")) if match else None))
    training_seed=config.get("training_seed",int(match.group("t")) if match else None)
    dataset_id=meta.get("name") or config.get("generator")
    if not dataset_id:
        dataset_id="synthetic_homogeneous_relational" if "relational" in rel else None
    is_code=path.suffix.lower()==".py" or path.name=="plot_style.py"
    new=(root==FINAL or "campaign_v2/relational" in rel or "homogeneous_relational" in rel)
    pilot=("campaign_v2" in rel and "/relational/" not in rel and "homogeneous_v2" in rel) or "self-autoregressive" in text
    legacy=(not new and not is_code and ("campaign_v2" not in rel or "district" in text.lower() or "community" in text.lower()))
    category="generic reusable code" if is_code else "new homogeneous scenario" if new else "legacy district/community scenario" if legacy else "ambiguous"
    included=bool(root==FINAL and not ("runs/" in rel or "datasets/" in rel))
    if root==SOURCE and new and any(token in rel for token in ["/tables/","/figures/","README.md","validation_checks"]): included=True
    if pilot: included=False
    reason=""
    if legacy: reason="legacy district/community scenario"
    elif pilot: reason="homogeneous self-autoregressive pilot; retained as ablation only"
    elif is_code: reason="reusable code, not a quantitative result"
    elif not included: reason="raw/intermediate artifact retained but not selected for final paper-facing analysis"
    exp=config.get("group") or ("simple_topological_baselines" if "simple_topological_baselines" in rel else
        "multiple_graph_realizations" if "multiple_graph_realizations" in rel else
        "error_distribution" if "error_analysis" in rel else path.parent.name)
    alpha=config.get("alpha")
    if alpha is None and match and match.group("a"): alpha=float(match.group("a").replace("p","."))
    return {"source_path":rel,"artifact_type":artifact_type(path),"experiment_type":exp,
            "dataset_id":dataset_id,"graph_seed":graph_seed,"signal_seed":signal_seed,
            "training_seed":training_seed,"L":config.get("lags",int(match.group("l")) if match else None),
            "H":config.get("horizon",int(match.group("h")) if match else None),"alpha":alpha,
            "k_dgm":config.get("k_dgm",int(match.group("k")) if match and match.group("k") else None),
            "decoder":config.get("decoder"),"consensus":config.get("consensus"),
            "epochs":config.get("epochs",int(match.group("e")) if match else None),
            "classification":category,"new_scenario_yes_no":"yes" if new else "no",
            "legacy_yes_no":"yes" if legacy else "no","included_in_final_analysis_yes_no":"yes" if included else "no",
            "exclusion_reason":reason,"size_bytes":path.stat().st_size,"modified_time":path.stat().st_mtime}


def main():
    rows=[]
    for root in [SOURCE,FINAL]:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    rows.append(classify(path,root))
    FINAL.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows).sort_values("source_path")
    frame.to_csv(FINAL/"new_scenario_manifest.csv",index=False)
    summary=frame.groupby(["classification","included_in_final_analysis_yes_no"]).size().reset_index(name="artifacts")
    summary.to_csv(FINAL/"scenario_classification_summary.csv",index=False)
    print(summary.to_string(index=False)); print(f"[OK] manifest rows={len(frame)}")


if __name__=="__main__": main()
