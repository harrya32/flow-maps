#!/usr/bin/env python3
"""Download, prune and prepare train-fitted HVG/PCA data for Calderon RNA.

Raw and derived data default to the git-ignored datasets/calderon2022 folder.
The Matrix Market file is streamed in its released column-major order, so the
full 547,805-cell COO matrix is never materialised. HVGs and PCA are fitted on
training cells only. PCA uses a small gene covariance matrix with implicit
centering/scaling rather than a dense cell-by-gene matrix.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets/calderon2022"
DOCS = ROOT / "docs/calderon2022"
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/calderon2022/numba"))

import anndata as ad
import numpy as np
import openpyxl
import pandas as pd
from scipy import sparse, linalg
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


def log(message):
    print(message, flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_sources(folder, cache_dir=None):
    folder.mkdir(parents=True, exist_ok=True)
    inspection = json.loads((DOCS / "inspection.json").read_text())
    sources = {row["id"]:row for row in inspection["sources"]}
    specifications = {
        "matrix": ("GSE190147_scirnaseq_gene_matrix.mtx.gz", "rna_expression_matrix_prefix", None),
        "columns": ("GSE190147_scirnaseq_gene_matrix.columns.txt.gz", "matrix_columns", "calderon2022-cells.columns.txt.gz"),
        "genes": ("GSE190147_scirnaseq_gene_matrix.rows.txt.gz", "matrix_rows", "calderon2022-genes.rows.txt.gz"),
        "annotations": ("rna_meta.rds", "rna_cell_annotations", "calderon2022-rna-meta.rds"),
        "lineages": ("table_s1_withLineage.xlsx", "rna_lineage_table", "calderon2022-table-s1-lineage.xlsx"),
        "decoder": ("celltype_list_rna.txt", "rna_lineage_decoder", "calderon2022-celltype-list.txt"),
    }
    paths, manifest = {}, {}
    for key,(filename,source_id,cached_name) in specifications.items():
        spec = sources[source_id]
        if key=="matrix":
            expected_bytes = inspection["expression_matrix"]["compressed_bytes"]
            completed_manifest = DOCS/"download_manifest.json"
            expected_hash = (json.loads(completed_manifest.read_text())["matrix"]["sha256"]
                             if completed_manifest.exists() else None)
        else:
            expected_bytes,expected_hash = spec["bytes"],spec["sha256"]
        target = folder/filename
        part = target.with_name(target.name+".part")
        if not target.exists() and part.exists() and part.stat().st_size==expected_bytes:
            part.replace(target)
        candidates = []
        if key in ("lineages","decoder"):candidates.append(DOCS/filename)
        if cache_dir and cached_name:candidates.append(cache_dir/cached_name)
        if not target.exists():
            cached = next((path for path in candidates if path.exists()
                           and path.stat().st_size==expected_bytes
                           and (expected_hash is None or file_hash(path)==expected_hash)),None)
            if cached:
                log(f"Reusing verified source: {cached.name}")
                shutil.copyfile(cached,target)
            else:
                log(f"Downloading {filename} ({expected_bytes:,} bytes)")
                offset = part.stat().st_size if part.exists() else 0
                headers = {"Range":f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(spec["url"],headers=headers)
                with urllib.request.urlopen(request,timeout=90) as response:
                    append = offset>0 and response.status==206
                    with part.open("ab" if append else "wb") as output:
                        shutil.copyfileobj(response,output,length=1<<20)
                if part.stat().st_size!=expected_bytes:
                    raise ValueError(f"Incomplete download: {part} ({part.stat().st_size:,}/{expected_bytes:,} bytes)")
                part.replace(target)
        if target.stat().st_size!=expected_bytes:
            raise ValueError(f"Unexpected source size: {target}")
        digest = file_hash(target)
        if expected_hash and digest!=expected_hash:
            raise ValueError(f"Source hash mismatch: {target}")
        paths[key] = target
        manifest[key] = {"url":spec["url"],"bytes":target.stat().st_size,"sha256":digest}
    (folder/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return paths


def annotation_frame(path, parsed_cache=None):
    if parsed_cache:
        # This optional cache was generated from the hash-verified RDS in the
        # preceding atlas inspection; downloaded third-party pickles are not used.
        return pd.read_pickle(parsed_cache)
    try:
        import rdata
    except ImportError:
        local = ROOT/".cache/calderon2022/python"
        if local.exists():sys.path.insert(0,str(local))
        import rdata
    log("Parsing the authors' per-cell RDS annotations")
    frame = rdata.conversion.convert(rdata.parser.parse_file(path))
    if not isinstance(frame,pd.DataFrame):raise TypeError("RNA annotations are not a data frame")
    return frame


def joined_metadata(paths, parsed_cache=None):
    columns = pd.read_csv(paths["columns"],sep="\t")
    metadata = annotation_frame(paths["annotations"],parsed_cache)
    if not columns.cell.is_unique or not metadata.cell.is_unique:
        raise ValueError("Cell identifiers must be unique")
    if set(columns.cell)!=set(metadata.cell):raise ValueError("Matrix and annotation cell ID sets differ")
    metadata = metadata.set_index("cell").reindex(columns.cell)
    for name in ("nCount_RNA","nFeature_RNA","time"):
        if not np.array_equal(columns[name].to_numpy(),metadata[name].to_numpy()):
            raise ValueError(f"Annotation/column mismatch for {name}")
    rows = list(openpyxl.load_workbook(paths["lineages"],data_only=True).active.values)[1:]
    table = pd.DataFrame({"inferred_time_window":[row[0] for row in rows],
                          "seurat_clusters":[int(row[1]) for row in rows],
                          "table_annotation":[str(row[6]).strip() for row in rows],
                          "branch_id":[str(row[11]).strip() for row in rows]})
    if table.duplicated(["inferred_time_window","seurat_clusters"]).any():
        raise ValueError("Duplicate time-window/cluster lineage keys")
    metadata = metadata.reset_index().merge(table,on=["inferred_time_window","seurat_clusters"],
                                            how="left",sort=False,validate="many_to_one")
    if metadata.branch_id.isna().any():raise ValueError("Cells lack a workbook state annotation")
    if not np.array_equal(metadata.cell.to_numpy(),columns.cell.to_numpy()):raise ValueError("Join changed matrix-column order")
    if not (metadata.manual_annot.astype(str).str.strip()==metadata.table_annotation).all():
        raise ValueError("Manual annotations disagree with the lineage table")
    keep = metadata.branch_id.ne("NA").to_numpy()
    # The 18-20 h rows all have NA lineages in the authors' released table.
    selected = metadata.loc[keep].copy()
    if selected.inferred_time_window.eq("18-20").any():raise ValueError("Unexpected assigned late state")
    selected["matrix_column"] = np.flatnonzero(keep).astype(np.int32)
    selected["manual_annot"] = selected.manual_annot.astype(str).str.strip()
    decoder = {}
    for line in paths["decoder"].read_text().splitlines():
        branch,germ,tissue = line.split("\t")
        if branch in decoder and decoder[branch]!=tissue:raise ValueError("Inconsistent decoder")
        decoder[branch]=tissue
    selected["broad_tissue"] = selected.branch_id.map(decoder)
    if selected.broad_tissue.isna().any():raise ValueError("Assigned branches lack decoder entries")
    selected["day"] = selected.inferred_time_window.astype(str)
    selected["time_hours_midpoint"] = selected.day.map(lambda s:int(s.split("-")[0])+1).astype(np.float32)
    selected.index = pd.Index(selected.cell,name="cell_id")
    selected = selected.drop(columns=["cell","table_annotation"])
    return selected,keep


def stream_retained_counts(path,keep,features_per_cell,output_dir,n_genes,progress=True):
    """Return cell-by-feature CSR using the source's verified sorted columns."""
    output_dir.mkdir(parents=True,exist_ok=True)
    n_cells,nnz = int(keep.sum()),int(np.asarray(features_per_cell,dtype=np.int64).sum())
    old_to_new = np.full(len(keep),-1,dtype=np.int32)
    old_to_new[keep] = np.arange(n_cells,dtype=np.int32)
    indptr = np.r_[0,np.cumsum(features_per_cell,dtype=np.int64)]
    if indptr[-1]>np.iinfo(np.int32).max:raise ValueError("This implementation expects <2^31 retained entries")
    indptr = indptr.astype(np.int32)
    data = np.lib.format.open_memmap(output_dir/"data.npy",mode="w+",dtype=np.int32,shape=(nnz,))
    indices = np.lib.format.open_memmap(output_dir/"indices.npy",mode="w+",dtype=np.int32,shape=(nnz,))
    total,written,total_umis = 0,0,0
    previous_column,previous_gene = -1,-1
    next_log = 20_000_000
    with gzip.open(path,"rb") as handle:
        if handle.readline().strip()!=b"%%MatrixMarket matrix coordinate integer general":
            raise ValueError("Unexpected Matrix Market format")
        line = handle.readline()
        while line.startswith(b"%"):line=handle.readline()
        features,cells,expected_nnz = map(int,line.split())
        if (features,cells)!=(n_genes,len(keep)):raise ValueError("Matrix dimensions disagree with metadata")
        carry = b""
        while True:
            block = handle.read(8<<20)
            if not block:
                payload,carry=carry,b""
                if not payload:break
            else:
                buffer=carry+block
                end=buffer.rfind(b"\n")
                if end<0:carry=buffer;continue
                payload,carry=buffer[:end+1],buffer[end+1:]
            values=np.fromstring(payload,sep=" ",dtype=np.int32)
            if values.size%3:raise ValueError("Malformed coordinate triples")
            triples=values.reshape(-1,3)
            genes,columns,counts=triples[:,0]-1,triples[:,1]-1,triples[:,2]
            if len(columns):
                if columns[0]<previous_column or np.any(np.diff(columns)<0):
                    raise ValueError("Matrix is not column-major; cannot build CSR without sorting")
                if (columns[0]==previous_column and genes[0]<=previous_gene
                    or np.any((columns[1:]==columns[:-1]) & (genes[1:]<=genes[:-1]))):
                    raise ValueError("Features are unsorted or duplicated within a cell")
                if columns.min()<0 or columns.max()>=cells or genes.min()<0 or genes.max()>=features or counts.min()<=0:
                    raise ValueError("Out-of-range or nonpositive count entry")
                previous_column,previous_gene=int(columns[-1]),int(genes[-1])
            selected=old_to_new[columns]>=0
            count_selected=int(selected.sum())
            if written+count_selected>nnz:raise ValueError("More retained entries than the metadata reports")
            data[written:written+count_selected]=counts[selected]
            indices[written:written+count_selected]=genes[selected]
            total+=len(triples);written+=count_selected
            total_umis+=int(counts.sum(dtype=np.int64))
            if progress and total>=next_log:
                log(f"Streamed {total:,}/{expected_nnz:,} entries; retained {written:,}/{nnz:,}")
                next_log+=20_000_000
    if total!=expected_nnz or written!=nnz:raise ValueError("Coordinate entry counts differ from metadata")
    data.flush();indices.flush();np.save(output_dir/"indptr.npy",indptr)
    matrix=sparse.csr_matrix((data,indices,indptr),shape=(n_cells,n_genes),copy=False)
    matrix.has_sorted_indices=True
    return matrix,{"source_entries":total,"source_umis":total_umis,"retained_entries":written}


def load_counts(folder):
    shape=json.loads((folder/"pruning.json").read_text())["shape"]
    data=np.load(folder/"csr/data.npy",mmap_mode="r")
    indices=np.load(folder/"csr/indices.npy",mmap_mode="r")
    indptr=np.load(folder/"csr/indptr.npy",mmap_mode="r")
    return sparse.csr_matrix((data,indices,indptr),shape=tuple(shape),copy=False)


def h5ad_safe_obs(frame):
    """Keep compatibility with AnnData 0.8's object-string HDF5 encoding."""
    frame=frame.copy()
    for column in frame:
        if isinstance(frame[column].dtype,pd.StringDtype) or pd.api.types.is_categorical_dtype(frame[column].dtype):
            frame[column]=frame[column].astype(str).astype(object)
    frame.index=pd.Index(frame.index.to_numpy(dtype=str),dtype=object,name=frame.index.name)
    return frame


def prune(folder,cache_dir=None,parsed_cache=None):
    if (folder/"pruning.json").exists():
        log("Reusing validated lineage-assigned count dataset")
        return
    paths=ensure_sources(folder/"raw",cache_dir)
    obs,keep=joined_metadata(paths,parsed_cache)
    with gzip.open(paths["genes"],"rt") as handle:genes=[line.rstrip("\r\n") for line in handle]
    if len(set(genes))!=len(genes):raise ValueError("Duplicate feature names")
    log(f"Retaining {len(obs):,}/{len(keep):,} cells, {obs.branch_id.nunique()} branches")
    audit_path=folder/"stream_audit.json"
    if audit_path.exists() and (folder/"csr/indptr.npy").exists():
        audit=json.loads(audit_path.read_text())
        data=np.load(folder/"csr/data.npy",mmap_mode="r")
        indices=np.load(folder/"csr/indices.npy",mmap_mode="r")
        indptr=np.load(folder/"csr/indptr.npy",mmap_mode="r")
        counts=sparse.csr_matrix((data,indices,indptr),shape=(len(obs),len(genes)),copy=False)
        if not np.array_equal(np.diff(counts.indptr),obs.nFeature_RNA.to_numpy()):
            raise ValueError("Cached CSR detected-feature counts differ from metadata")
        log("Reusing completed source stream; revalidating every retained cell's counts")
    else:
        counts,audit=stream_retained_counts(paths["matrix"],keep,obs.nFeature_RNA.to_numpy(),folder/"csr",len(genes))
        audit_path.write_text(json.dumps(audit,indent=2)+"\n")
    sums=np.zeros(len(obs),dtype=np.int64)
    for start in range(0,len(obs),10000):
        stop=min(start+10000,len(obs));left,right=int(counts.indptr[start]),int(counts.indptr[stop])
        if np.any(np.diff(counts.indptr[start:stop+1])==0):raise ValueError("Unexpected empty QC-passing cell")
        sums[start:stop]=np.add.reduceat(counts.data[left:right],counts.indptr[start:stop]-left,dtype=np.int64)
    if not np.array_equal(sums,obs.nCount_RNA.to_numpy(dtype=np.int64)):
        raise ValueError("Per-cell UMI totals disagree with independently released metadata")
    inspection=json.loads((DOCS/"inspection.json").read_text())
    if audit["source_umis"]!=inspection["expression_matrix"]["total_umis_from_column_metadata"]:
        raise ValueError("Full matrix UMI total differs from the column metadata")
    if len(obs)!=inspection["annotations"]["lineage_assigned_cells_0_to_18"]:
        raise ValueError("Assigned-cell count changed from the inspected release")
    obs.to_csv(folder/"lineage_assigned_metadata.csv.gz")
    (folder/"feature_names.txt").write_text("\n".join(genes)+"\n")
    obs=h5ad_safe_obs(obs)
    raw=ad.AnnData(X=counts,obs=obs,var=pd.DataFrame(index=pd.Index(genes,name="feature")),dtype=counts.dtype)
    raw.uns["source_dataset"]="GSE190147"
    raw.uns["lineage_assignment"]="Authors' inferred-time-window/cluster workbook lookup"
    log("Writing pruned raw-count H5AD")
    raw.write_h5ad(folder/"lineage_assigned_counts.h5ad",compression="gzip",compression_opts=1)
    manifest={"shape":[len(obs),len(genes)],"branches":int(obs.branch_id.nunique()),
              "broad_tissues":int(obs.broad_tissue.nunique()),"retained_umis":int(sums.sum()),
              "source_cells":len(keep),"pruned_cells":int((~keep).sum()),**audit,
              "cells_per_branch":{str(k):int(v) for k,v in obs.groupby("branch_id").size().items()},
              "cells_per_window":{str(k):int(v) for k,v in obs.groupby("day").size().items()},
              "count_validation":"All retained per-cell nFeature and UMI totals, full source totals and gzip CRC verified"}
    (folder/"pruning.json").write_text(json.dumps(manifest,indent=2)+"\n")
    log(f"Pruned dataset validated: {len(obs):,} cells, {counts.nnz:,} entries, {sums.sum():,} UMIs")


def training_blocks(matrix,training_mask,block_rows=10000):
    for start in range(0,matrix.shape[0],block_rows):
        stop=min(start+block_rows,matrix.shape[0])
        local=training_mask[start:stop]
        if local.any():yield matrix[start:stop][local]


def gene_moments(matrix,training_mask,clip=None):
    total=np.zeros(matrix.shape[1]);squares=np.zeros_like(total);detected=np.zeros_like(total,dtype=np.int64)
    for block in training_blocks(matrix,training_mask):
        values=block.data.astype(np.float64)
        if clip is not None:values=np.minimum(values,clip[block.indices])
        total+=np.bincount(block.indices,weights=values,minlength=matrix.shape[1])
        squares+=np.bincount(block.indices,weights=values*values,minlength=matrix.shape[1])
        detected+=np.bincount(block.indices,minlength=matrix.shape[1])
    n=int(training_mask.sum());mean=total/n
    variance=np.maximum((squares-total*mean)/(n-1),0)
    return mean,variance,total,squares,detected


def seurat_v3_hvgs(matrix,training_mask,n_hvgs,min_cells=3):
    """Scanpy/Seurat v3's VST formula, evaluated in bounded-memory blocks."""
    from skmisc.loess import loess
    mean,variance,_,_,detected=gene_moments(matrix,training_mask)
    nonconstant=variance>0
    fitted=np.zeros(matrix.shape[1])
    model=loess(np.log10(mean[nonconstant]),np.log10(variance[nonconstant]),span=0.3,degree=2)
    model.fit();fitted[nonconstant]=model.outputs.fitted_values
    std=np.sqrt(10**fitted);n=int(training_mask.sum())
    clip=mean+std*np.sqrt(n)
    _,_,clipped_sum,clipped_squares,_=gene_moments(matrix,training_mask,clip)
    score=(n*mean**2+clipped_squares-2*clipped_sum*mean)/((n-1)*std**2)
    eligible=nonconstant & (detected>=min_cells)
    if eligible.sum()<n_hvgs:raise ValueError("Too few eligible expressed genes for HVG selection")
    ranking=np.argsort(-np.where(eligible,score,-np.inf),kind="stable")
    selected=np.sort(ranking[:n_hvgs])
    stats=pd.DataFrame({"training_mean_counts":mean,"training_variance_counts":variance,
                        "training_detected_cells":detected,"vst_normalized_variance":score,
                        "eligible":eligible,"highly_variable":np.isin(np.arange(matrix.shape[1]),selected)})
    return selected,stats


def split_cells(obs,variant,seed):
    labels=obs.branch_id.to_numpy(dtype=str);indices=np.arange(len(obs))
    if variant=="random":
        remainder,test=train_test_split(indices,test_size=0.1,stratify=labels,random_state=seed)
        train,val=train_test_split(remainder,test_size=1/9,stratify=labels[remainder],random_state=seed+1)
    else:
        test=indices[obs.day.eq("8-10").to_numpy()]
        remainder=np.setdiff1d(indices,test)
        train,val=train_test_split(remainder,test_size=0.1,stratify=labels[remainder],random_state=seed)
    if set(labels[train])!=set(labels):raise ValueError("The training split does not cover all branches")
    split=np.full(len(obs),"train",dtype=object);split[val]="validation";split[test]="test"
    return split


def fit_scaled_pca(expression,training_mask,n_pcs):
    train=expression[training_mask].astype(np.float64)
    n=train.shape[0]
    mean=np.asarray(train.mean(axis=0)).ravel()
    log("Computing the training-only gene covariance matrix")
    covariance=(train.T@train).toarray()
    covariance-=n*np.outer(mean,mean);covariance/=(n-1)
    scale=np.sqrt(np.maximum(np.diag(covariance),1e-16))
    covariance/=scale[:,None];covariance/=scale[None,:]
    del train;gc.collect()
    log(f"Solving for the leading {n_pcs} PCs without densifying cell-by-gene data")
    eigenvalues,eigenvectors=linalg.eigh(covariance,subset_by_index=[expression.shape[1]-n_pcs,expression.shape[1]-1],
                                       overwrite_a=True,check_finite=False,driver="evr")
    eigenvalues=eigenvalues[::-1].copy();components=eigenvectors[:,::-1].T.copy()
    # Deterministic component signs, with the largest loading positive.
    pivots=np.abs(components).argmax(axis=1)
    components*=np.where(components[np.arange(n_pcs),pivots]<0,-1,1)[:,None]
    weights=(components/scale).T.astype(np.float32)
    offset=(mean/scale)@components.T
    scores=np.empty((expression.shape[0],n_pcs),dtype=np.float32)
    for start in range(0,len(scores),10000):
        stop=min(start+10000,len(scores))
        scores[start:stop]=expression[start:stop]@weights-offset.astype(np.float32)
    if not np.isfinite(scores).all() or np.any(eigenvalues<=0):raise ValueError("Invalid PCA coordinates or eigenvalues")
    return scores,mean,scale,components,eigenvalues


def prepare_pca(folder,variant,n_hvgs=3000,n_pcs=100,seed=0):
    destination=folder/f"lineage_assigned_hvg{n_hvgs}_pca{n_pcs}_{variant}.h5ad"
    if destination.exists():
        manifest_path=folder/f"preprocessing_{variant}.json"
        if not manifest_path.exists():
            raise ValueError(f"Incomplete PCA cache: {destination}; remove it and regenerate")
        manifest=json.loads(manifest_path.read_text())
        if (manifest["seed"]!=seed or manifest["n_pcs"]!=n_pcs or manifest["shape"][1]!=n_hvgs):
            raise ValueError("Cached PCA settings differ; use a separate --data-dir for a different preprocessing fit")
        log(f"Reusing PCA dataset: {destination}");return
    counts=load_counts(folder)
    obs=pd.read_csv(folder/"lineage_assigned_metadata.csv.gz",index_col=0)
    genes=(folder/"feature_names.txt").read_text().splitlines()
    split=split_cells(obs,variant,seed);obs["classifier_split"]=split
    training_mask=split=="train"
    log(f"{variant}: split sizes {dict(Counter(split))}; fitting {n_hvgs} HVGs on {training_mask.sum():,} training cells")
    selected,hvg_stats=seurat_v3_hvgs(counts,training_mask,n_hvgs)
    log("Normalising HVG counts to 10,000 total-library UMIs and log1p transforming")
    expression=counts[:,selected].astype(np.float32)
    del counts;gc.collect()
    factors=(10000/obs.nCount_RNA.to_numpy(dtype=np.float64)).astype(np.float32)
    from sklearn.utils.sparsefuncs import inplace_row_scale
    inplace_row_scale(expression,factors)
    np.log1p(expression.data,out=expression.data)
    scores,mean,scale,components,eigenvalues=fit_scaled_pca(expression,training_mask,n_pcs)
    hvg_stats.index=pd.Index(genes,name="feature")
    var=hvg_stats.iloc[selected].copy();var["pca_training_mean_log1p"]=mean;var["pca_training_scale_log1p"]=scale
    processed=ad.AnnData(X=expression,obs=h5ad_safe_obs(obs),var=var,dtype=expression.dtype)
    processed.obsm["X_pca"]=scores
    processed.varm["PCs"]=components.T.astype(np.float32)
    processed.uns["pca"]={"variance":eigenvalues,"variance_ratio":eigenvalues/n_hvgs,
                          "fit_split":"train","zero_center":True,"scale_genes_to_unit_variance":True}
    processed.uns["preprocessing"]={"variant":variant,"seed":seed,"n_hvgs":n_hvgs,"n_pcs":n_pcs,
        "hvg_method":"Seurat v3 VST: span 0.3; raw training counts; min 3 training cells per feature",
        "normalization_target":10000,"normalization_library":"All 23,932 original feature counts per cell",
        "pca_solver":"Exact training-only covariance eigh; implicit centering and unit variance scaling",
        "heldout_window":"8-10" if variant=="heldout_8_10" else "none"}
    hvg_stats.to_csv(folder/f"hvg_statistics_{variant}.csv.gz")
    np.savez_compressed(folder/f"preprocessing_hvg{n_hvgs}_pca{n_pcs}_{variant}.npz",
                        selected_feature_indices=selected,feature_names=np.asarray(genes)[selected],
                        gene_mean=mean,gene_scale=scale,components=components,variance=eigenvalues,
                        train_indices=np.flatnonzero(training_mask),seed=np.asarray(seed))
    log(f"Writing {destination.name}; PCA100 explains {eigenvalues.sum()/n_hvgs:.2%} of scaled HVG variance")
    processed.write_h5ad(destination,compression="gzip",compression_opts=1)
    manifest={"variant":variant,"shape":list(processed.shape),"n_pcs":n_pcs,"seed":seed,
              "split_sizes":{str(k):int(v) for k,v in Counter(split).items()},
              "n_train_branches":int(obs.loc[training_mask,"branch_id"].nunique()),
              "n_test_branches":int(obs.loc[split=="test","branch_id"].nunique()),
              "explained_variance_ratio":float(eigenvalues.sum()/n_hvgs),
              "normalization_target":10000,"hvg_fit":"train only","pca_fit":"train only",
              "source_pruning_sha256":file_hash(folder/"pruning.json")}
    (folder/f"preprocessing_{variant}.json").write_text(json.dumps(manifest,indent=2)+"\n")
    log(f"PCA dataset complete: {destination}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--stage",choices=["prune","pca","all"],default="all")
    parser.add_argument("--variant",choices=["random","heldout_8_10"],default="random")
    parser.add_argument("--reuse-source-cache",type=Path)
    parser.add_argument("--parsed-metadata-cache",type=Path)
    parser.add_argument("--n-hvgs",type=int,default=3000)
    parser.add_argument("--n-pcs",type=int,default=100)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if args.n_hvgs<=args.n_pcs or args.n_pcs<1:parser.error("HVG count must exceed the positive PC count")
    folder=args.data_dir.resolve();folder.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=args.threads):
        if args.stage in ("prune","all"):prune(folder,args.reuse_source_cache,args.parsed_metadata_cache)
        if args.stage in ("pca","all"):prepare_pca(folder,args.variant,args.n_hvgs,args.n_pcs,args.seed)


if __name__=="__main__":main()
