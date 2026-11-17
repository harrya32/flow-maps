#!/usr/bin/env python3
"""Compare branch-ID and direct broad-tissue classification from PCA100.

The two targets use identical cells, splits, training-fitted HVGs/PCA and PC
scalers. MLP checkpoints follow the repo's PyTorch/NumPy export schema. A
linear-softmax baseline is fitted for the random-cell protocol. The second
protocol holds out every assigned cell in the inferred 8-10 h window.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/calderon2022/matplotlib"))
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,recall_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from matplotlib.ticker import PercentFormatter

from scripts import train_cite_multi_celltype_classifiers as shared


def log(message):print(message,flush=True)


class TensorBatches:
    """Vectorised batches avoid Python per-cell collation for 300k cells."""
    def __init__(self,x,y,batch_size,shuffle=False,seed=0):
        self.x=torch.from_numpy(x);self.y=torch.from_numpy(y)
        self.batch_size=batch_size;self.shuffle=shuffle
        self.generator=torch.Generator().manual_seed(seed)

    def __iter__(self):
        order=torch.randperm(len(self.y),generator=self.generator) if self.shuffle else None
        for start in range(0,len(self.y),self.batch_size):
            stop=min(start+self.batch_size,len(self.y))
            if self.shuffle and stop-start==1:continue
            selection=order[start:stop] if self.shuffle else slice(start,stop)
            yield self.x[selection],self.y[selection]


def load_data(path,expected_seed=None,expected_variant=None):
    backed=ad.read_h5ad(path,backed="r")
    try:
        x=np.asarray(backed.obsm["X_pca"],dtype=np.float32).copy()
        obs=backed.obs.copy();features=backed.var_names.astype(str).tolist()
        preprocessing=backed.uns["preprocessing"]
        if expected_seed is not None and preprocessing["seed"]!=expected_seed:
            raise ValueError("Cached PCA seed differs; use a separate --data-dir for a different fit")
        if expected_variant is not None and preprocessing["variant"]!=expected_variant:
            raise ValueError("The PCA representation does not match the requested evaluation protocol")
    finally:backed.file.close()
    if x.shape[1]!=100 or not np.isfinite(x).all():raise ValueError("Expected finite PCA100 coordinates")
    split=obs.classifier_split.astype(str).to_numpy()
    if set(split)!={"train","validation","test"}:raise ValueError("Invalid fixed data split")
    return x,obs,features,{name:np.flatnonzero(split==name) for name in ("train","validation","test")}


def evaluate_probabilities(model,x,batch_size=8192):
    model.eval();outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            outputs.append(model(torch.from_numpy(x[start:start+batch_size])).numpy())
    logits=np.concatenate(outputs)
    if not np.isfinite(logits).all():raise ValueError("Nonfinite classifier logits")
    return softmax(logits,axis=1).astype(np.float32)


def scores(y,probabilities,train_counts):
    predicted=probabilities.argmax(1);present=np.unique(y)
    # Evaluate negative log-likelihood in float64. Clip and renormalise
    # exactly as sklearn does, without its warning after rounding/clipping.
    loss_probabilities=probabilities.astype(np.float64)
    loss_probabilities/=loss_probabilities.sum(axis=1,keepdims=True)
    epsilon=np.finfo(np.float64).eps
    np.clip(loss_probabilities,epsilon,1-epsilon,out=loss_probabilities)
    loss_probabilities/=loss_probabilities.sum(axis=1,keepdims=True)
    negative_log_likelihood=-np.log(loss_probabilities[np.arange(len(y)),y]).mean()
    confidence=probabilities.max(1);correct=predicted==y
    high={}
    for threshold in (0.8,0.9):
        selected=confidence>=threshold
        high[str(threshold)]={"coverage":float(selected.mean()),
                              "accuracy":float(correct[selected].mean()) if selected.any() else None,
                              "n_cells":int(selected.sum())}
    ece=0.0
    for lower,upper in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        selected=(confidence>lower)&(confidence<=upper)
        if selected.any():ece+=selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return {"n_cells":len(y),"n_present_classes":len(present),"n_output_classes":probabilities.shape[1],
            "accuracy":float(accuracy_score(y,predicted)),
            "balanced_accuracy":float(recall_score(y,predicted,labels=present,average="macro",zero_division=0)),
            "macro_f1":float(f1_score(y,predicted,labels=present,average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(y,predicted,average="weighted",zero_division=0)),
            "top3_accuracy":float(np.mean(np.any(np.argsort(probabilities,axis=1)[:,-3:]==y[:,None],axis=1))),
            "log_loss":float(negative_log_likelihood),
            "majority_training_class_accuracy":float(np.mean(y==np.argmax(train_counts))),
            "confidence_subsets":high,"expected_calibration_error_15bins":float(ece),
            "macro_average_scope":"Classes present in this evaluated split"}


def save_figures(prefix,history,confusion,per_class,target,variant):
    label="Branch IDs" if target=="branch_id" else "Broad tissue labels"
    protocol="random cell split" if variant=="random" else "held-out 8-10 h window"
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name in ("train","validation"):
        axes[0].plot(history.epoch,history[f"{name}_loss"],label=name)
        axes[1].plot(history.epoch,history[f"{name}_accuracy"],label=name)
    axes[0].set(xlabel="Epoch",ylabel="Weighted smoothed cross-entropy",title="Loss")
    axes[1].set(xlabel="Epoch",ylabel="Accuracy",title="Accuracy",ylim=(0,1.01))
    for axis in axes:axis.legend();axis.grid(alpha=0.2)
    fig.suptitle(f"{label}: {protocol}")
    fig.savefig(prefix.with_name(prefix.name+"_training.png"),dpi=180);plt.close(fig)

    names=per_class.index.tolist();denominator=np.maximum(confusion.sum(1,keepdims=True),1)
    normalised=confusion/denominator
    absent=confusion.sum(1)==0
    shown=np.ma.masked_array(normalised,mask=np.broadcast_to(absent[:,None],normalised.shape))
    cmap=plt.get_cmap("Blues").copy();cmap.set_bad("#ECEEEF")
    size=max(8,min(15,len(names)*0.23+3))
    fig,ax=plt.subplots(figsize=(size,size),constrained_layout=True)
    image=ax.imshow(shown,cmap=cmap,vmin=0,vmax=1,interpolation="nearest")
    ax.set_xticks(np.arange(len(names)));ax.set_xticklabels(names,rotation=90,fontsize=7 if len(names)>30 else 9)
    ax.set_yticks(np.arange(len(names)));ax.set_yticklabels(names,fontsize=7 if len(names)>30 else 9)
    for row,column in zip(*np.where(normalised>=0.15)):
        ax.text(column,row,f"{normalised[row,column]:.0%}",ha="center",va="center",fontsize=6.5,
                color="white" if normalised[row,column]>.5 else "#1F3444")
    title=f"{label}: test confusion ({protocol})"
    if absent.any():title+="\nGrey rows: labels absent from this test split"
    ax.set(xlabel="Predicted",ylabel="Assigned label",title=title)
    fig.colorbar(image,ax=ax,shrink=.7,label="Fraction within true class")
    fig.savefig(prefix.with_name(prefix.name+"_test_confusion.png"),dpi=180)
    fig.savefig(prefix.with_name(prefix.name+"_test_confusion.svg"));plt.close(fig)

    shown=per_class[per_class.support>0].sort_values("recall")
    fig,ax=plt.subplots(figsize=(11,max(6,len(shown)*.23+1.5)),constrained_layout=True)
    ax.barh(np.arange(len(shown)),shown.recall,color="#557F9B")
    ax.set_yticks(np.arange(len(shown)));ax.set_yticklabels(shown.index,fontsize=8.5)
    for row,(_,values) in enumerate(shown.iterrows()):
        ax.text(1.01,row,f"n={int(values.support)}",va="center",fontsize=8,color="#526574")
    ax.set(xlim=(0,1.20),xlabel="Recall on held-out cells",title=f"{label}: per-class recall ({protocol})")
    ax.set_xticks(np.linspace(0,1,6));ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x",alpha=.2)
    fig.savefig(prefix.with_name(prefix.name+"_test_recall.png"),dpi=180)
    fig.savefig(prefix.with_name(prefix.name+"_test_recall.svg"));plt.close(fig)


def train(args,path,target,variant,kind):
    stem=f"calderon_{target}_hvg{args.n_hvgs}_pca100_{variant}_{kind}"
    prefix=args.output_dir/"reports"/stem
    metrics_path=prefix.with_name(prefix.name+"_metrics.json")
    if metrics_path.exists() and not args.overwrite:
        completed=json.loads(metrics_path.read_text())
        settings={"seed":args.seed,"n_features":args.n_hvgs,"batch_size":args.batch_size,
                  "dropout":args.dropout if kind=="mlp" else 0,
                  "label_smoothing":args.label_smoothing,"class_weight_power":args.class_weight_power,
                  "learning_rate":args.learning_rate if kind=="mlp" else 3e-3,"weight_decay":args.weight_decay}
        if any(completed[key]!=value for key,value in settings.items()):
            raise ValueError(f"Completed model settings differ for {stem}; pass --overwrite to refit")
        if not (args.output_dir/f"{stem}.pt").exists() or not (args.output_dir/f"{stem}.npz").exists():
            raise ValueError(f"Checkpoint missing for {stem}; pass --overwrite to refit")
        log(f"Reusing completed model: {stem}");return completed
    prefix.parent.mkdir(parents=True,exist_ok=True)
    x,obs,features,indices=load_data(path,expected_seed=args.seed,expected_variant=variant)
    classes=sorted(obs[target].astype(str).unique().tolist())
    label_ids={name:index for index,name in enumerate(classes)}
    y=obs[target].astype(str).map(label_ids).to_numpy(dtype=np.int64)
    train_counts=np.bincount(y[indices["train"]],minlength=len(classes))
    if np.any(train_counts==0):raise ValueError("Training data miss a target class")
    scaler=StandardScaler().fit(x[indices["train"]])
    train_x=scaler.transform(x[indices["train"]]).astype(np.float32)
    val_x=scaler.transform(x[indices["validation"]]).astype(np.float32)
    train_y=y[indices["train"]];val_y=y[indices["validation"]]
    train_batches=TensorBatches(train_x,train_y,args.batch_size,shuffle=True,seed=args.seed)
    val_batches=TensorBatches(val_x,val_y,8192)
    shared.seed_everything(np,torch,args.seed)
    if kind=="mlp":
        model=shared.make_model(torch.nn,100,len(classes),args.dropout,(128,64))
    else:model=torch.nn.Linear(100,len(classes))
    weights=(len(train_y)/(len(classes)*train_counts))**args.class_weight_power
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=args.label_smoothing)
    learning_rate=args.learning_rate if kind=="mlp" else 3e-3
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.3,patience=5,min_lr=1e-6)
    max_epochs=args.max_epochs if kind=="mlp" else min(args.max_epochs,50)
    best_loss=float("inf");stopping_loss=float("inf");bad_epochs=0;best_state=None;best_epoch=0;rows=[]
    started=time.monotonic();device=torch.device("cpu")
    log(f"{stem}: {len(classes)} classes; " + ", ".join(f"{key}={len(value):,}" for key,value in indices.items()))
    for epoch in range(1,max_epochs+1):
        train_loss,train_accuracy=shared.run_epoch(torch,model,train_batches,criterion,device,optimizer)
        val_loss,val_accuracy,true,predicted=shared.evaluate(np,torch,model,val_batches,criterion,device)
        present=np.unique(true)
        val_macro=float(f1_score(true,predicted,labels=present,average="macro",zero_division=0))
        if not np.isfinite(val_loss):raise ValueError("Nonfinite validation loss")
        scheduler.step(val_loss)
        if val_loss<best_loss:
            best_loss=val_loss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
        if val_loss<stopping_loss-1e-4:stopping_loss=val_loss;bad_epochs=0
        else:bad_epochs+=1
        rows.append({"epoch":epoch,"train_loss":train_loss,"train_accuracy":train_accuracy,
                     "validation_loss":val_loss,"validation_accuracy":val_accuracy,"validation_macro_f1":val_macro,
                     "learning_rate":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.monotonic()-started})
        pd.DataFrame(rows).to_csv(prefix.with_name(prefix.name+"_history.csv"),index=False)
        if epoch==1 or epoch%5==0:
            log(f"{stem}: epoch {epoch}, train acc={train_accuracy:.3f}, val acc={val_accuracy:.3f}, "
                f"val macro-F1={val_macro:.3f}, val loss={val_loss:.4f}, elapsed={time.monotonic()-started:.0f}s")
        if bad_epochs>=args.patience:
            log(f"{stem}: early stopping at epoch {epoch}");break
    model.load_state_dict(best_state);model.eval()
    arrays=shared.state_dict_to_numpy(np,model.state_dict())
    verification=val_x[:256]
    with torch.no_grad():reference=model(torch.from_numpy(verification)).numpy()
    exported=(shared.numpy_logits(np,arrays,verification) if kind=="mlp"
              else verification@arrays["weight"].T+arrays["bias"])
    np.testing.assert_allclose(reference,exported,rtol=1e-5,atol=1e-5)

    results={}
    test_probabilities=None
    for split in ("validation","test"):
        evaluated_x=scaler.transform(x[indices[split]]).astype(np.float32)
        probabilities=evaluate_probabilities(model,evaluated_x)
        results[split]=scores(y[indices[split]],probabilities,train_counts)
        if split=="test":test_probabilities=probabilities
    test_y=y[indices["test"]];test_pred=test_probabilities.argmax(1)
    report=classification_report(test_y,test_pred,labels=np.arange(len(classes)),target_names=classes,output_dict=True,zero_division=0)
    per_class=pd.DataFrame({name:report[name] for name in classes}).T
    per_class["train_cells"]=train_counts
    per_class["validation_cells"]=np.bincount(val_y,minlength=len(classes))
    per_class.to_csv(prefix.with_name(prefix.name+"_test_per_class.csv"),index_label="class")
    confusion=confusion_matrix(test_y,test_pred,labels=np.arange(len(classes)))
    pd.DataFrame(confusion,index=classes,columns=classes).to_csv(prefix.with_name(prefix.name+"_test_confusion.csv"))
    test_obs=obs.iloc[indices["test"]].copy()
    test_obs["assigned_label"]=np.asarray(classes)[test_y]
    test_obs["predicted_label"]=np.asarray(classes)[test_pred]
    test_obs["prediction_confidence"]=test_probabilities.max(1)
    predictions_dir=args.data_dir/"predictions";predictions_dir.mkdir(exist_ok=True)
    test_obs.to_csv(predictions_dir/f"{stem}_test_predictions.csv.gz")
    window_rows=[]
    for window in sorted(test_obs.day.astype(str).unique(),key=lambda value:int(value.split("-")[0])):
        selected=test_obs.day.astype(str).to_numpy()==window
        window_metrics=scores(test_y[selected],test_probabilities[selected],train_counts)
        window_rows.append({"window":window,**{key:window_metrics[key] for key in
                            ("n_cells","n_present_classes","accuracy","balanced_accuracy","macro_f1")}})
    pd.DataFrame(window_rows).to_csv(prefix.with_name(prefix.name+"_test_by_window.csv"),index=False)

    derived_broad=None
    if target=="branch_id":
        mapping=obs.groupby("branch_id",observed=True).broad_tissue.first().astype(str).to_dict()
        tissue_classes=sorted(set(mapping.values()));tissue_ids={name:i for i,name in enumerate(tissue_classes)}
        broad_prob=np.zeros((len(test_y),len(tissue_classes)),dtype=np.float32)
        for index,branch in enumerate(classes):broad_prob[:,tissue_ids[mapping[branch]]]+=test_probabilities[:,index]
        true_broad=test_obs.broad_tissue.astype(str).map(tissue_ids).to_numpy(dtype=np.int64)
        broad_counts=np.bincount(obs.iloc[indices["train"]].broad_tissue.astype(str).map(tissue_ids),minlength=len(tissue_classes))
        derived_broad=scores(true_broad,broad_prob,broad_counts)

    split_digest=hashlib.sha256()
    split_digest.update(obs.index.to_numpy(dtype=str).tobytes())
    split_digest.update(obs.classifier_split.astype(str).to_numpy(dtype=str).tobytes())
    metadata={"dataset":"calderon2022","variant":variant,"target":target,"label_key":target,"model_kind":kind,
              "source_h5ad":str(path),"selected_genes":features,"n_pcs":100,"class_names":classes,
              "architecture":[100,128,64,len(classes)] if kind=="mlp" else [100,len(classes)],
              "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),
              "seed":args.seed,"best_epoch":best_epoch,"epochs_run":len(rows),
              "selection_metric":"validation_weighted_smoothed_cross_entropy","best_validation_loss":best_loss,
              "split_sizes":{key:len(value) for key,value in indices.items()},"cell_and_split_sha256":split_digest.hexdigest(),
              "n_cells":len(obs),"n_features":len(features),"dropout":args.dropout if kind=="mlp" else 0,
              "class_weight_power":args.class_weight_power,"label_smoothing":args.label_smoothing,
              "batch_size":args.batch_size,"learning_rate":learning_rate,"weight_decay":args.weight_decay,
              "elapsed_seconds":time.monotonic()-started,"numpy_export_max_abs_error":float(np.max(abs(reference-exported))),
              "preprocessing_fit":"HVGs, gene scaling, PCA and PC scaler fit on training cells only",
              "classifier_inputs":"PCA100 only; no time, cluster, manual annotation or tissue metadata inputs",
              "metrics":results,"derived_tissue_metrics_from_branch_probabilities":derived_broad}
    torch.save({"model_state_dict":model.state_dict(),"metadata":metadata},args.output_dir/f"{stem}.pt")
    np.savez_compressed(args.output_dir/f"{stem}.npz",class_names=np.asarray(classes),
                        scaler_mean=scaler.mean_.astype(np.float32),scaler_scale=scaler.scale_.astype(np.float32),
                        n_pcs=np.asarray(100),dataset=np.asarray("calderon2022"),variant=np.asarray(variant),
                        target=np.asarray(target),model_kind=np.asarray(kind),**arrays)
    metrics_path.write_text(json.dumps(metadata,indent=2)+"\n")
    save_figures(prefix,pd.DataFrame(rows),confusion,per_class,target,variant)
    log(f"COMPLETE {stem}: test acc={results['test']['accuracy']:.4f}, "
        f"balanced acc={results['test']['balanced_accuracy']:.4f}, macro-F1={results['test']['macro_f1']:.4f}")
    return metadata


def write_comparison(args):
    pattern=f"calderon_*_hvg{args.n_hvgs}_pca100_*_metrics.json"
    records=[json.loads(path.read_text()) for path in (args.output_dir/"reports").glob(pattern)]
    records.sort(key=lambda record:(record["target"],record["variant"],record["model_kind"]))
    table=[]
    for record in records:
        values=record["metrics"]["test"]
        table.append({"target":record["target"],"variant":record["variant"],"model":record["model_kind"],
                      "classes":values["n_output_classes"],"test_classes":values["n_present_classes"],
                      "test_cells":values["n_cells"],"accuracy":values["accuracy"],
                      "balanced_accuracy":values["balanced_accuracy"],"macro_f1":values["macro_f1"],
                      "best_epoch":record["best_epoch"]})
    frame=pd.DataFrame(table);frame.to_csv(args.output_dir/"reports/comparison.csv",index=False)
    mlp=[record for variant in ("random","heldout_8_10") for target in ("branch_id","broad_tissue")
         for record in records if record["variant"]==variant and record["target"]==target and record["model_kind"]=="mlp"]
    if mlp:
        fig,ax=plt.subplots(figsize=(11,5.5),constrained_layout=True)
        locations=np.arange(len(mlp));width=.24
        for offset,(metric,label,color) in enumerate([("accuracy","Accuracy","#4D7592"),
                                                    ("balanced_accuracy","Mean class recall","#88A8BE"),
                                                    ("macro_f1","Macro-F1","#80A990")]):
            values=[record["metrics"]["test"][metric] for record in mlp]
            bars=ax.bar(locations+(offset-1)*width,values,width,label=label,color=color)
            for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.015,f"{value:.1%}",ha="center",fontsize=9)
        names=[("Branch IDs" if record["target"]=="branch_id" else "Broad tissues")+"\n"+
               ("Random cell split" if record["variant"]=="random" else "Held-out 8-10 h") for record in mlp]
        ax.set_xticks(locations);ax.set_xticklabels(names)
        ax.set(ylim=(0,1.13),ylabel="Held-out test score",title=f"Calderon RNA: direct classification from training-fitted HVG{args.n_hvgs} / PCA100")
        ax.set_yticks(np.linspace(0,1,6));ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper center",ncol=3);ax.grid(axis="y",alpha=.15)
        fig.savefig(args.output_dir/"reports/classifier_comparison.png",dpi=190)
        fig.savefig(args.output_dir/"reports/classifier_comparison.svg");plt.close(fig)
    pruning=json.loads((args.data_dir/"pruning.json").read_text())
    lines=["# Calderon RNA: PCA100 classifier feasibility", "",
           f"The downloaded RNA count matrix was pruned to **{pruning['shape'][0]:,} lineage-assigned cells**, "
           f"{pruning['shape'][1]:,} original features, **{pruning['branches']} branch IDs** and **{pruning['broad_tissues']} broad labels**. "
           f"The retained matrix has {pruning['retained_entries']:,} entries and {pruning['retained_umis']:,} UMIs.","",
           "Both targets are trained directly using exactly the same cells, train/validation/test partitions and PCA100 features. "
           "Only PCA features enter the classifiers; inferred time, cluster IDs and annotations are not predictor inputs.","",
           "## Held-out results","",
           "| Target | Protocol | Model | Output / test-present classes | Test cells | Accuracy | Mean class recall | Macro-F1 |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for record in records:
        metric=record["metrics"]["test"]
        label="Branch IDs" if record["target"]=="branch_id" else "Broad tissues"
        protocol="Random cells" if record["variant"]=="random" else "Entire 8-10 h bin"
        lines.append(f"| {label} | {protocol} | {record['model_kind']} | {metric['n_output_classes']} / {metric['n_present_classes']} | "
                     f"{metric['n_cells']:,} | {metric['accuracy']:.2%} | {metric['balanced_accuracy']:.2%} | {metric['macro_f1']:.2%} |")
    if mlp:lines.extend(["","![Direct classifier comparison](../../calderon-classifiers/reports/classifier_comparison.png)"])
    complete={(record["target"],record["variant"]):record for record in mlp}
    if all((target,variant) in complete for target in ("branch_id","broad_tissue")
           for variant in ("random","heldout_8_10")):
        branch_random=complete["branch_id","random"]["metrics"]["test"]
        tissue_random=complete["broad_tissue","random"]["metrics"]["test"]
        branch_time=complete["branch_id","heldout_8_10"]["metrics"]["test"]
        tissue_time=complete["broad_tissue","heldout_8_10"]["metrics"]["test"]
        random_gain=100*(tissue_random["accuracy"]-branch_random["accuracy"])
        time_gain=100*(tissue_time["accuracy"]-branch_time["accuracy"])
        lines.extend(["","## Feasibility interpretation","",
            f"Direct broad-tissue prediction gains **{random_gain:.2f} percentage points** of accuracy over branch-ID prediction "
            f"on the random split and **{time_gain:.2f} points** on the held-out developmental window. "
            f"The latter achieves {tissue_time['accuracy']:.2%} tissue accuracy versus {branch_time['accuracy']:.2%} branch accuracy. "
            "At this baseline, the coarser labels are the more practical starting point for an additional flow experiment. "
            "The substantial errors, particularly across developmental windows, argue for reporting classifier uncertainty and using soft label probabilities. "
            "An experiment that treats every predicted branch as a correct hard label is not supported by these results.","",
            "These are single-seed baseline results with the repo's existing MLP architecture, not an exhaustive search for the best possible classifier. "
            "Each MLP is allowed up to 100 epochs with early stopping; the best validation-loss checkpoint is selected. "
            "The random split shares atlas collection groups across training and testing, and the temporal split holds out an expression-inferred age window. "
            "Neither is an independent-embryo validation.","","## Confidence and coverage","",
            "The following subsets keep test cells whose predicted probability is at least 0.9. They are descriptive subsets of the same fitted models; "
            "the full-population results above remain the primary comparison.","",
            "| Target | Protocol | Coverage | Accuracy within subset | Cells |",
            "| --- | --- | ---: | ---: | ---: |"])
        for record in mlp:
            subset=record["metrics"]["test"]["confidence_subsets"]["0.9"]
            label="Branch IDs" if record["target"]=="branch_id" else "Broad tissues"
            protocol="Random cells" if record["variant"]=="random" else "Entire 8-10 h bin"
            subset_accuracy=f"{subset['accuracy']:.2%}" if subset["accuracy"] is not None else "n/a"
            lines.append(f"| {label} | {protocol} | {subset['coverage']:.2%} | {subset_accuracy} | {subset['n_cells']:,} |")
        lines.extend(["","High confidence improves reliability but leaves most cells outside the retained subset. "
                      "Its coverage must be reported if it is used in a downstream experiment.","",
                      "## Weakest classes","",
                      "The five lowest-recall classes present in each MLP test split are shown below. The complete CSV files also contain precision, F1 and training/validation support.","",
                      "| Target | Protocol | Assigned label | Test cells | Recall |",
                      "| --- | --- | --- | ---: | ---: |"])
        for record in mlp:
            stem=f"calderon_{record['target']}_hvg{args.n_hvgs}_pca100_{record['variant']}_mlp"
            per_class=pd.read_csv(args.output_dir/"reports"/f"{stem}_test_per_class.csv")
            shown=per_class[per_class.support>0].sort_values("recall").head(5)
            label="Branch IDs" if record["target"]=="branch_id" else "Broad tissues"
            protocol="Random cells" if record["variant"]=="random" else "Entire 8-10 h bin"
            for _,row in shown.iterrows():
                lines.append(f"| {label} | {protocol} | `{row['class']}` | {int(row.support):,} | {row.recall:.2%} |")
        lines.extend(["","## Coarsening the branch classifier's probabilities","",
            "As a separate diagnostic, sum each branch classifier's probabilities over branches mapped to the same broad tissue. "
            "This uses the already trained branch model; it is distinct from the directly trained broad-tissue classifiers above.","",
            "| Protocol | Aggregated branch accuracy | Aggregated branch macro-F1 | Direct tissue accuracy | Direct tissue macro-F1 |",
            "| --- | ---: | ---: | ---: | ---: |"])
        for variant in ("random","heldout_8_10"):
            derived=complete["branch_id",variant]["derived_tissue_metrics_from_branch_probabilities"]
            direct=complete["broad_tissue",variant]["metrics"]["test"]
            protocol="Random cells" if variant=="random" else "Entire 8-10 h bin"
            lines.append(f"| {protocol} | {derived['accuracy']:.2%} | {derived['macro_f1']:.2%} | "
                         f"{direct['accuracy']:.2%} | {direct['macro_f1']:.2%} |")
    lines.extend(["","## Evaluation and preprocessing","",
        "The random-cell protocol uses a branch-stratified 80/10/10 split. The temporal protocol places every assigned 8-10 h cell in the test set, "
        "then splits the other cells 90/10 into training/validation. Every branch and broad tissue remains represented in training. "
        "Macro-F1 and mean recall average only classes present in the evaluated split; absent classes are retained in the model head and per-class CSV reports.","",
        "HVG selection uses Seurat v3's raw-count variance-stabilisation formula (LOESS span 0.3), restricted to genes detected in at least three training cells. "
        "The 3,000 highest-scoring features are normalised using each cell's complete original library size (target 10,000), then log1p transformed. "
        "Gene means, unit-variance scaling and exact covariance PCA100 are fitted on training cells only. The PC scaler is also fitted only on training cells. "
        "Different evaluation protocols have different training-fitted PCA spaces; checkpoints must be paired with their corresponding representation.","",
        "The MLP matches the repo's existing 100 -> 128 -> 64 -> output architecture, with BatchNorm, ReLU and dropout 0.2. "
        "AdamW uses learning rate 0.001, weight decay 0.0001, inverse-frequency class-weight power 0.25 and label smoothing 0.05. "
        "A validation-loss scheduler and early stopping select the checkpoint without inspecting test performance. A linear-softmax baseline is fitted for the random-cell protocol.","",
        "## Data, checkpoints and detailed reports","",
        "- Raw pruned counts: `datasets/calderon2022/lineage_assigned_counts.h5ad` (integer cell-by-feature counts).",
        "- Fitted representations: `datasets/calderon2022/lineage_assigned_hvg3000_pca100_{random,heldout_8_10}.h5ad`; "
        "`X` contains log-normalised HVGs, `obsm['X_pca']` contains PCA100, and `obs` retains branch, broad tissue, manual annotation and both clocks.",
        "- Both `.pt` and `.npz` checkpoints are in `calderon-classifiers/`; MLP exports use the existing repo schema.",
        "- Metrics, per-class precision/recall/F1, per-window performance, training curves, row-normalised confusion matrices and recall plots are in `calderon-classifiers/reports/`.",
        "- Per-cell test predictions are in the ignored `datasets/calderon2022/predictions/` directory.","",
        "The broad labels are the authors' **branch-level tissue decoder**, including `Unknown`; they are not independently measured per-cell tissue identities. "
        "`Unknown` accounts for 50,501 retained cells (16.06%). "
        "The branch labels and inferred time coordinate come from the atlas's computational reconstruction. This experiment measures recovery of those assigned labels on held-out expression profiles, not lineage-tracing accuracy.","",
        "## Reproduce","","```sh",
        "python scripts/prepare_calderon_lineage_data.py --stage prune",
        "python scripts/train_calderon_lineage_classifiers.py",
        "# Regenerate the report from completed results:",
        "python scripts/train_calderon_lineage_classifiers.py --report-only",
        "```","",
        "Dependencies are NumPy, SciPy, pandas, AnnData, openpyxl, scikit-misc, scikit-learn, PyTorch and matplotlib; "
        "`rdata` is needed to parse the downloaded annotation RDS when a previously parsed cache is unavailable. "
        "The download/pruning script verifies source hashes, all retained per-cell detected-feature/UMI totals, the full source entry/UMI totals and gzip CRC. "
        "Numerical tests compare the bounded-memory VST calculation with Scanpy and implicit PCA with dense scikit-learn PCA."])
    lines.extend(["","The complete download hashes are recorded in [download_manifest.json](download_manifest.json). "
        "The [saved-model validation audit](classifier_validation.json) verifies raw-to-PCA projection on 256 sampled cells, "
        "training-index/feature alignment, identical branch/tissue splits, and reproduction of all six test metrics from reloaded PyTorch checkpoints. "
        "It also verifies the MLP exports through the repository's existing NumPy classifier loader."])
    report=ROOT/"docs/calderon2022/classification.md";report.write_text("\n".join(lines)+"\n")
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"datasets/calderon2022")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"calderon-classifiers")
    parser.add_argument("--targets",nargs="+",choices=["branch_id","broad_tissue"],default=["branch_id","broad_tissue"])
    parser.add_argument("--variants",nargs="+",choices=["random","heldout_8_10"],default=["random","heldout_8_10"])
    parser.add_argument("--models",nargs="+",choices=["linear","mlp"],default=["linear","mlp"])
    parser.add_argument("--n-hvgs",type=int,default=3000)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--batch-size",type=int,default=2048)
    parser.add_argument("--max-epochs",type=int,default=100)
    parser.add_argument("--patience",type=int,default=15)
    parser.add_argument("--learning-rate",type=float,default=1e-3)
    parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--class-weight-power",type=float,default=.25)
    parser.add_argument("--label-smoothing",type=float,default=.05)
    parser.add_argument("--dropout",type=float,default=.2)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--report-only",action="store_true",help="Regenerate the comparison report and figure from completed metrics")
    args=parser.parse_args();args.data_dir=args.data_dir.resolve();args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    plt.rcParams.update({"svg.fonttype":"none","font.family":"DejaVu Sans"})
    if args.report_only:
        write_comparison(args);log("Comparison report regenerated.");return
    with threadpool_limits(limits=args.threads):
        for target in args.targets:
            for variant in args.variants:
                path=args.data_dir/f"lineage_assigned_hvg{args.n_hvgs}_pca100_{variant}.h5ad"
                if not path.exists():
                    subprocess.run([sys.executable,str(ROOT/"scripts/prepare_calderon_lineage_data.py"),
                                    "--stage","pca","--variant",variant,"--data-dir",str(args.data_dir),
                                    "--n-hvgs",str(args.n_hvgs),"--seed",str(args.seed),"--threads",str(args.threads)],check=True)
                for kind in args.models:
                    if kind=="linear" and variant!="random":continue
                    train(args,path,target,variant,kind)
                    write_comparison(args)
    log("All requested classifier comparisons completed.")


if __name__=="__main__":main()
