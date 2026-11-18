# Calderon et al. (2022): feasibility of a lineage-constrained flow experiment

Assessed 16 November 2026 against the paper supplied by the reviewer, its public data releases, and the existing Maizels and CITE/Multi implementations in this repository. Numerical inspection results, source URLs, and file hashes are in [inspection.json](inspection.json). The authors' RNA lineage workbook is preserved unchanged as [table_s1_withLineage.xlsx](table_s1_withLineage.xlsx).

The **current Calderon flow defaults** are the [three-time experiment](three_time_flow_map.md): training at **4–6 / 8–10 / 12–14 h**, intermediate W1 at **6–8 / 10–12 h**, the selected-times classifier for filtering and constraint loss, and the all-times reference classifier for final violations. Flow settings follow Maizels: independent 90/10 training/validation splits, one 1,024-pair validation batch with the training objective, instantaneous parameters, 50 steps per observed/query interval, and raw violations from 512 held-out sources at 50 full-window checks. W1 now uses **2,048-cell source/reference subsets**; MMD, direct and linear evaluation are disabled, while composed and Euler evaluation remain enabled. Logging cadence and OT fallbacks match Maizels. Plain OT, filtered OT, constrained OT and the weight-2000 config all use this setup. The [defaults comparison](maizels_defaults_comparison.md) documents remaining differences; the earlier endpoint pilot results below retain their original protocol.

The [stochastic counterparts](stochastic_flow_map.md) provide plain minibatch OT, endpoint-filtered OT, and endpoint-filtered OT with a differentiable direct-map lineage constraint. They inherit the stochastic Maizels training defaults, including EMA evaluation, three noise draws, and constraint weight 10, with **4,096-cell source/reference W1 subsets** and **composed-only evaluation; MMD and direct evaluation are disabled**. The same time windows and tissue classifiers are used. Preparation, training, and saved-checkpoint evaluation are supported; CPU tests and two-step GPU smoke runs passed.

The [graph-comparison figures and construction notes](figures.md) now compare the 53-branch hierarchy, a graph over 49 finer cell-state annotations, and a further coarsening to the authors' 24 broad tissue labels. The figures are documented reconstructions; they do not replace the missing complete weighted RNA state-edge export. Merging the finer annotations barely reduces node count and produces cycles, while the broader decoder produces a smaller graph with paths that can combine different branches.

The full RNA matrix has now been downloaded and pruned to **314,387 lineage-assigned cells**. The [PCA100 classifier experiment](classification.md) compares direct prediction of the 53 branch IDs and 24 broad tissue labels using 3,000 training-selected HVGs, a random-cell split, and an entirely held-out 8–10 h inferred-age window. Data and model outputs are described there.

The [OT flow-map example](flow_map_example.md) implements **4–6 → 12–14 h within `Ubi`**, with 8–10 h held out for interpolation evaluation and separate endpoint validation/test cells. Its full seed-0 run completed 7,000 steps, with endpoint validation selecting step 6,000. [Results and controls](ot_flow_map_results.md) show improved endpoint W1 but no improvement over unchanged sources at the held-out middle window. Raw classifier-detected path violations were 54.66%; confidence filtering gave 22.02% among assessed paths, with 43.16% assessed-path coverage and only 2.61% all-four-checkpoint coverage.

The subsequent [50-checkpoint evaluation](ot_flow_map_50_checkpoints.md) preserves the checkpoint, test cells and 100-step trajectories. It gives **63.74% raw path violations**, **16.37% filtered violations among assessed paths**, **60.14% assessed-path coverage**, and **1.85% coverage at every checkpoint**. Marginal W1 remains unchanged; the filtered rate falls as its assessed denominator grows.

The [filtered-interpolant variant](filtered_flow_map.md) follows Maizels ID 5: it masks forbidden endpoint pairs, checks 50 interior points with the frozen tissue classifier, and resolves the masked OT assignment after rejecting invalid interpolants. It uses the same endpoint splits, network, optimizer and plain-OT validation objective as the baseline. Partial assignments retain only supported transport mass; the run records retained mass and resulting tissue sampling weights.

Its full seed-0 run completed **4,700 updates**, selecting **step 3,700**. [The matched comparison and figure](filtered_ot_flow_map_results.md) show raw violations falling **63.74% → 58.31%**, confidence-filtered violations **16.37% → 11.61%**, and assessed-path coverage **60.14% → 58.28%**. Middle W1 changes **8.4545 → 8.4856** and endpoint W1 **10.7758 → 11.1041**: the violation reduction comes with slightly worse marginal agreement. The filter retains **82.54%** of minibatch mass and changes training tissue weights. The report includes named-tissue results and classifier limitations.

The [constrained and filtered variant](constrained_flow_map.md) adds the native Maizels ID 6 `loss_points_nll` penalty with weight 1000. Its differentiable frozen tissue ensemble was verified against the original decoder and with finite-difference gradients. It retains the same 50-point training filter and selects checkpoints using a separate copy of the shared plain-OT validation loss, with the lineage penalty disabled.

The constrained run completed **3,500 updates**, selecting **step 2,500**. [Its three-variant results and figure](constrained_ot_flow_map_results.md) show raw violations falling **58.31% → 36.39%** relative to filtering alone, while middle/end W1 worsen **8.4856 → 8.7176 / 11.1041 → 13.2912**. Aggregate confidence-filtered violations change **11.61% → 13.48%**, with much higher assessed-path coverage (**58.28% → 86.81%**). Among named-tissue starts, filtered violations fall **36.97% → 29.92%**, while coverage rises **38.17% → 81.18%**. The report explains the changing assessed population and the marginal-fit tradeoff.

The [expressive-network follow-up](expressive_classification.md) trains 18 wider, residual, regularized and joint branch/tissue networks, then selects calibrated ensembles using validation only. It improves random-cell accuracy to **65.56% for branch IDs and 72.45% for tissues**, but improves the held-out window only to **47.17% and 53.25%**. Extra capacity alone can worsen stage transfer. The larger models give a modest improvement, while leaving the main generalization and classifier-uncertainty limitations unresolved.

The [all-time atlas classifier experiment](all_times_classification.md), completed 17 November 2026, refits **HVG3000/PCA100 on all 314,387 lineage-assigned cells across 0–18 h**. No time window is reserved. Its shared branch-stratified 80/10/10 split has **251,509 training, 31,439 validation and 31,439 test cells**, with branch and tissue proportions matching the atlas to within 0.0026 percentage points. Nine new networks produce validation-selected ensembles with **65.88% branch-ID accuracy and 72.88% tissue accuracy** on the test cells. These are label holdouts: the requested HVG/PCA fit includes their expression profiles. The new classifiers require the new PCA space when used in a subsequent flow experiment.

The [PC1/PC2 figures](pca2_labels.md) show every lineage-assigned cell in that representation, separately coloured by branch ID and broad tissue. Both views use identical coordinates, axis limits and draw order. The first two PCs explain 1.37% of scaled HVG variance.

The [three-timepoint design audit](three_timepoint_design.md) checks all 84 three-window choices for tissue coverage and recomputes graph/marginal support. It compares the original 4–6 / 10–12 / 16–18 h proposal with shorter alternatives, including the selected symmetric **4–6 / 8–10 / 12–14 h** schedule. Each covers all 23 in-range tissues. The report also documents the substantial compatibility floor under the current tissue prior. Constraint and all-times evaluation classifiers have separate training roles.

The [selected symmetric-window classifier experiment](three_time_classifier.md) trains five neural recipes using only **4–6, 8–10 and 12–14 h**, retaining the all-times PCA100 representation. Its validation-selected tissue ensemble achieves **75.48% accuracy on 16,221 held-out cells from those windows**, **58.78% on the entire 6–8 h marginal** and **67.84% on the entire 10–12 h marginal** (**62.50% pooled**). All 23 in-range labels are represented in classifier training, validation and test, but stage transfer remains substantially weaker than cell holdout performance. The report includes per-window and per-tissue figures, saved checkpoints, and an audit verifying excluded temporal labels, frozen selection and an unchanged all-times reference classifier.

## Recommendation

**This is a worthwhile additional dataset. Its RNA data have the required counts, cell identities, annotations, and temporal coverage, and the paper explicitly reconstructs substantially more complex developmental relationships. It is not yet a drop-in hard-lineage experiment.** The main preparation is deciding which annotated states and inferred relationships constitute a defensible prior, rather than obtaining the expression data.

In particular, distinguish the paper's inferred, time-resolved cell-state graph from a validated genealogy or a static tree over cell-type names. Only about 59% of the 0–18 h RNA cells have an explicit lineage assignment in the released lineage table. A simple hierarchy reconstructed from those assignments also produces prematurely ending branches and incompatible snapshot proportions under strict balanced transport. The classifier pilot now finds moderate recovery of those assignments: **63.5% branch / 71.0% tissue accuracy** on held-out random cells, falling to **46.7% / 52.6%** on an entirely held-out 8–10 h window. A curated coarser prior with classifier uncertainty is a more defensible next experiment than treating all predicted branches as correct hard labels.

## 1. What biological process is measured?

The study profiles **whole Drosophila melanogaster embryogenesis at 25°C**, from 0 to 20 hours. It covers early maternal transcription and rapid nuclear divisions, cellularization at approximately 2 h 20 min, zygotic genome activation, gastrulation and germ-layer formation, and diversification into neuronal, epidermal, muscle, gut, tracheal, germ-cell, and other tissues. A major expansion of cellular diversity occurs around 6–10 h. This is a whole-organism developmental process, with many concurrent tissue trajectories.

Embryos were pooled from **11 overlapping collection windows**. Each collection was split for separate sci-RNA-seq3 and sci-ATAC-seq3 assays. The measurements are from **nuclei**, although the paper often calls them cells. RNA and ATAC are not paired measurements from the same nucleus.

The paper reports:

| Assay | Profiles after QC | Per-nucleus depth | Other relevant size |
| --- | ---: | --- | --- |
| RNA | 547,805 | Median 399 UMIs and 274 detected genes | Released count matrix: 23,932 features × 547,805 nuclei |
| ATAC | 976,460 | Median 5,206 nonduplicate reads | 110,185 accessible regions identified across embryogenesis |

The RNA assay alone is sufficient for an experiment comparable to this repository's expression-space experiments. Neither paired multimodal observations nor RNA velocity are prerequisites for the current flow models. The paper actually reports unstable velocity estimates from individual sparse RNA profiles, which is relevant if considering a velocity-based extension.

Sources: main paper, Results and Discussion, supplied PDF pp. 4–8 and 15; [published article](https://doi.org/10.1126/science.abn5800).

## 2. Is its structure comparable to the existing datasets?

**Yes at the expression/annotation level; its clock is different.** The counts matrix and metadata can be converted into an AnnData/PCA dataset with the same conceptual fields used here. This is an hours-long embryo atlas, with collection intervals and inferred nuclear ages, rather than a series of precisely observed culture days.

The following sizes are for the actual local benchmark exports, not necessarily the full original studies:

| Dataset | Cells/nuclei available | Expression representation used or available | Temporal information | Labels/prior |
| --- | ---: | --- | --- | --- |
| Maizels flow export | 44,713 | 50 PCs; original study also supplies count and new/old RNA layers | Ten observed times: D3, D3.2, D3.4, D3.6, D3.8, D4, D5, D6, D7, D8 | Nine cell types; active graph has eight differentiation edges and maximum depth four edges |
| CITE export | 31,240 | 22,050 expression features plus 100 PCs; exported expression is transformed, not integer counts | Observed days 2, 3, 4, 7 | Seven types; HSC branches directly into the six other types |
| Multi export | 33,701 | 23,418 expression features plus 100 PCs; exported expression is transformed, not integer counts | Observed days 2, 3, 4, 7 | Same seven-type prior as CITE |
| Calderon RNA | 547,805 total; 532,059 at inferred ages 0–18 h | Sparse integer count matrix; PCA must be prepared | Eleven overlapping measured collection windows; continuous NN-predicted age; nine 2 h bins for the main lineage analysis | Paper reports 54 annotation labels and 268 time-specific RNA states; released metadata has 51 distinct normalized manual labels |

The Maizels CSV contains cell IDs, timepoints, cell annotations, and PC1–PC50. The CITE/Multi loaders require `obs['day']`, `obs['cell_type']`, and `obsm['X_pca']`. Calderon supplies corresponding information across separate files, so a data adapter and preprocessing step are needed, rather than a different generative architecture.

The 54-versus-51 label count is a formatting issue in the released annotation workbook: `brain`, `epidermis`, and `ventral nerve cord` also occur with trailing nonbreaking spaces. Stripping whitespace reduces its 54 raw labels to 51, exactly matching the cell metadata. All per-cell manual annotations match the workbook after this normalization. Do not create extra classifier classes for those whitespace variants.

Repository sources: [Maizels utilities](../../py/common/maizels.py), [CITE/Multi utilities](../../py/common/cite_multi.py), and [CITE/Multi preparation](../../scripts/download_cite_multi.py). Local source-file dimensions and cell counts were checked read-only.

## 3. Where is the dataset, and what was verified?

The GEO superseries is [GSE190149](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190149). Its RNA subseries is [GSE190147](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190147), and its ATAC subseries is [GSE190130](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190130). The authors maintain a [DEAP download page](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/) with additional processed data and annotations.

For a Python RNA experiment, the useful files are:

| File | Contents | Download size verified |
| --- | --- | ---: |
| [GEO expression matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190147/suppl/GSE190147_scirnaseq_gene_matrix.mtx.gz) | Gene/feature-by-cell MatrixMarket integer counts | 697,773,156 bytes, about 665 MiB |
| [GEO matrix columns](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190147/suppl/GSE190147_scirnaseq_gene_matrix.columns.txt.gz) | Cell ID, experiment, RNA depth, collection window, mitochondrial/ribosomal fractions, NNv1 age; one row per matrix column | 19,958,916 bytes |
| [GEO matrix rows](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190147/suppl/GSE190147_scirnaseq_gene_matrix.rows.txt.gz) | 23,932 unique feature names in matrix row order | 88,224 bytes |
| [RNA annotation metadata](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/cell_to_annotations/rna_meta.rds) | 547,805 rows with cell ID, inferred window, cluster, manual annotation, germ layer, QC, collection window, and age predictions | 24,110,965 bytes |
| [RNA lineage workbook](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/RNA/update/rna_lineage_info/table_s1_withLineage.xlsx) | Time-window/cluster annotation table with an added `lineage` column; described by the authors as information used for Fig. 3C | 29,286 bytes |
| [Lineage decoder](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/RNA/update/rna_lineage_info/celltype_list_rna.txt) | Lineage-ID-to-germ-layer/tissue descriptions | Small text file |

There is also a [processed Seurat main object](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/RNA/update/seurat_objects/), listed at approximately 1.4 GB, plus objects split by collection or predicted-age window. The GEO sparse matrix route avoids needing to convert the full Seurat object. Reprocessing FASTQs is unnecessary for this experiment.

The inspection established the following:

- The matrix header is `%%MatrixMarket matrix coordinate integer general`, with **23,932 rows, 547,805 columns, and 212,703,167 stored entries**. The initial availability inspection used its compressed prefix; the subsequent classifier preparation downloaded and streamed the complete matrix, verifying all entries and gzip CRC without constructing the full matrix in memory.
- Independently, the sum of detected-feature counts in the column metadata is exactly **212,703,167**, matching the matrix's reported entry count.
- The full source has **347,601,285 UMIs**. The retained integer-count matrix has **110,680,144 entries and 184,972,858 UMIs**; detected-feature counts and UMI totals match the independent metadata for every retained cell.
- Both cell metadata files contain **547,805 unique IDs**, and their ID sets are identical. Their row orders differ. Join/reindex by `cell`, never concatenate them positionally.
- Collection windows, RNA count totals, and detected-feature counts match exactly by cell ID. The largest NNv1 age discrepancy is approximately 5.3 × 10^-14, consistent with text serialization.
- Joining the annotation metadata to the lineage workbook by **`(inferred_time_window, seurat_clusters)`** matches all 547,805 cells. Cluster numbers alone are not globally unique.

The column file has no cell-type labels; the RDS annotation file provides them. The annotation RDS has no lineage-ID column; the workbook supplies that additional mapping. The experiment therefore needs both joins.

## 4. How should time be interpreted?

The measured collection windows are 0–2, 1–3, 2–4, 3–7, 4–8, 6–10, 8–12, 10–14, 12–16, 14–18, and 16–20 h. They overlap and contain asynchronous embryos. A cell's exact developmental age was not measured.

The authors trained neural networks to predict the collection-window midpoint from RNA or ATAC, then used the resulting predictions as a more finely resolved developmental-age coordinate. `NNv1_age` is the MSE-based model selected for the downstream analysis. The release also includes lasso and NNv2 estimates and prediction-error fields.

For Fig. 3, they grouped cells into nonoverlapping **inferred-age** bins and excluded ages above 18 h because of edge effects and reduced late-bin support. These are useful pseudo-snapshots, but a held-out inferred-age bin is not a held-out experimentally timed culture day.

| Inferred age (h) | All RNA cells | Explicit lineage assignment | No lineage assignment |
| --- | ---: | ---: | ---: |
| 0–2 | 122,331 | 24,844 | 97,487 |
| 2–4 | 29,508 | 10,025 | 19,483 |
| 4–6 | 84,626 | 52,746 | 31,880 |
| 6–8 | 33,412 | 30,876 | 2,536 |
| 8–10 | 77,035 | 58,004 | 19,031 |
| 10–12 | 24,073 | 21,448 | 2,625 |
| 12–14 | 58,550 | 51,456 | 7,094 |
| 14–16 | 45,411 | 40,919 | 4,492 |
| 16–18 | 57,113 | 24,069 | 33,044 |
| 18–20 | 15,746 | Excluded from the paper's main lineage analysis | Excluded |
| **0–18 total** | **532,059** | **314,387** | **217,672** |

Preserve both the measured collection window and continuous NN age. A first implementation can use 2 h bin midpoints as its model clock while retaining the continuous age for sensitivity analyses. Minute-scale binning would exaggerate the certainty of individual ages and reduce cell/branch support. The paper identifies germ cells, early collections, and low-depth RNA cells as cases with larger prediction error.

Because ages are predicted from expression and the published annotations/tree use the full atlas, this would be a **retrospective interpolation benchmark with an atlas-derived prior**. Splitting flow-training and evaluation cells still matters, but it does not turn the age coordinate or lineage prior into independently measured ground truth.

## 5. Does the paper propose a lineage tree?

**Yes. Fig. 3C is the RNA developmental graph, and Fig. 3D is the independently inferred ATAC graph.** The paper links clusters from adjacent inferred-age bins using nearest neighbors in a joint embedding, retains strong precursor relationships, and obtains directed acyclic developmental graphs. It explicitly cautions that these are state-based inferences rather than lineage tracing and that some trajectories terminate prematurely.

The RNA graph has **268 time-specific states across nine bins**. The released workbook additionally contains 19 states from 18–20 h, for 287 rows in total. Of the 268 states at 0–18 h, **184 have a non-`NA` lineage ID and 84 do not**. Their corresponding cell counts are the 314,387 and 217,672 above.

### Published graph versus a candidate prior for this repository

The lineage IDs are structured strings, for example `Ubi_A`, `Ubi_AA`, and `Ubi_AAB`. A candidate branch hierarchy can be reconstructed by treating longer letter codes as descendants of the code obtained by removing their final letter. `Ubi_0` connects to `Ubi_A/B/C/D`; `Blastoderm_A` and `Yolk_A` are separate roots.

This candidate has:

- **53 assigned branch IDs**, **50 parent links**, three roots, **14 branching nodes**, and **31 terminal branch IDs**.
- Maximum ancestry depth **five edges**, for example `Ubi_0 → Ubi_A → Ubi_AA → Ubi_AAB → Ubi_AABA → Ubi_AABAA`.
- A large `Ubi` component containing **253,182 cells and 46 branch IDs** at 0–18 h. The other components contain 30,419 Blastoderm and 30,786 Yolk cells.

These are our reconstruction statistics, not a claim that the authors publish a validated 53-cell-type tree. The linked workbook does not provide the complete weighted RNA state-edge list. The public [RNA pipeline](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/scripts/RNA/data_processing/pipeline_inferred_ages.R) writes such precursor-link information during processing, but an exact Fig. 3C edge/weight export was not identified among the inspected linked files. There is an explicit public [ATAC edge file](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/scripts/ATAC/flydev_edge_atac.txt); applying it to RNA would require a separate cross-modality annotation mapping.

This is clearly broader than the CITE/Multi seven-type star or the Maizels nine-type graph. Its reconstructed branch ancestry is also somewhat deeper than the active Maizels hierarchy. However, nine temporal bins alone are not evidence of nine differentiation decisions: repeated states and maturation steps must not be counted as new biological branches.

### Why a tree over the 51 cell-type names needs care

Branch IDs and cell-type names are not interchangeable. Several branch IDs share a manual annotation, and a single branch ID can contain different annotations at successive stages. Some states are mixed or unknown.

A concrete example is `Ubi_BDAB`: its 8–10 h state is annotated **visceral muscle primordium**, and its 10–12 h state **brain**. The authors' comments explicitly describe both as mixed muscle/neuronal or midline clusters. A literal muscle-to-brain biological constraint would misinterpret this record. Similarly, collapsing all `brain` states into one node loses distinctions between the branches represented in Fig. 3C.

Consequently, either retain curated branch/state identities, with a classifier capable of recognizing them, or construct a biological annotation DAG that handles mixed states and repeated tissue labels explicitly. Do not silently equate the prefix hierarchy with a validated cell-type genealogy.

The paper also explores mesoderm (51,338 RNA profiles in its selected analysis) and fine neuronal diversity (6,703 mature-neuron profiles reclustered into 11 subtypes). Those are possible targeted alternatives, and [fine neuronal annotation files](https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/RNA/update/neuronal_fine_scale_annotations/) are released. The 11 subtypes should not simply be appended to the main tree without establishing how those labels and precursor states connect.

## 6. Practical checks before a full experiment

### Sparse counts and classifier quality

The RNA data are shallow: 399 median UMIs is substantially less information per cell than in many expression benchmarks. The [completed HVG3000/PCA100 classifier pilot](classification.md) tests all 53 branch IDs and the authors' 24 broad tissue labels on matched splits. The coarser target performs better, but neither classifier gives reliable labels across the whole held-out developmental window. Some branch recalls are below 4%; broad tracheal and somatic-muscle recalls are 10.9% and 22.2%. Tissue predictions above 0.9 confidence are 98.0% accurate but cover only 17.6% of the held-out-window cells. Classifier uncertainty and coverage therefore need to be part of a downstream experiment. A time-conditioned state classifier is a possible further improvement; it was not used in this PCA-only comparison.

### Held-out-bin class support

Removing a time bin can remove a transient manual label from classifier training. For example, holding out 8–10 h removes six manual annotation classes from the other 0–18 h bins, including procrystal cells and several primordia. Conversely, **every assigned branch ID occurring at 8–10 h is also represented outside that bin**. This favors a curated branch-based pilot, while still requiring classifier validation. Recheck support after any annotation merging or graph pruning.

### Compatibility of a hard mask with observed proportions

We audited normalized branch-count histograms under the candidate prefix hierarchy. The audit allows only self/descendant transitions and maximizes matched transport mass subject to source and target histogram capacities. A result below 100% means exact balanced transport cannot match both raw snapshot proportions under that particular mask.

| Interval | Dataset subset | Maximum supported transport mass |
| --- | --- | ---: |
| 6–8 → 8–10 h | All explicitly assigned lineages | 69.3% |
| 10–12 → 12–14 h | All explicitly assigned lineages | 62.5% |
| 14–16 → 16–18 h | All explicitly assigned lineages | 58.6% |
| 6–8 → 10–12 h, skipping an 8–10 h evaluation bin | `Ubi` component only | 69.8% |

For example, at 6–8 h the `Ubi_AB`, `Ubi_D`, and `Ubi_BB` branches contain cells but have no observed permitted destination at 8–10 h under the prefix mask. Changing tissue proportions and disconnected components introduce additional mass restrictions. The paper's comments on sampling and prematurely ending trajectories make this an expected issue to inspect.

These are **compatibility diagnostics for our naive mask**, not performance results, nor proof that a fully recovered/curated version of the authors' graph is incompatible. Nonetheless, they show why simply supplying the released IDs to a strict balanced masked-OT routine would be unsuitable. Curating missing/ambiguous transitions, using declared soft constraints, changing sampling weights, or explicitly modeling growth/loss can change the problem. Altering weights also changes the benchmark population and must be stated.

The repository already supports unconstrained classes in its reachability utilities and differentiable constraint variants. Those are useful for genuinely ambiguous states. They should not be used to obscure the fact that some parts of the prior are incomplete.

## 7. Suggested experiment sequence

1. **Prepare the RNA data.** This is now complete for the assigned-cell subset: the three GEO matrix files and RNA annotations were joined by cell ID, followed by `(window, cluster)` lineage lookup. The sparse integer-count export retains collection time, NN age and QC. The earlier classifier and flow experiments use 3,000 HVGs and 100 PCs fitted on their declared training population. The [all-time atlas representation](all_times_classification.md) refits both steps on every lineage-assigned cell and provides a reference classifier without reserving a temporal window.

2. **Curate the prior before fitting a flow.** Review mixed annotations and prematurely ending paths against Fig. 3C and the workbook comments. Decide whether the model states are branch IDs or biological cell-type/state labels. Freeze the resulting graph, missing-label policy, and uncertain-edge policy before examining model results. Keep time-resolved information; a flat mature-tissue star would lose the reason to use this dataset.

3. **Resolve the classifier/support limitations.** The pilot is complete for all 53 assigned branch IDs, with 8–10 h reserved in its temporal protocol. The newer all-time reference classifiers use validation/test cells from every window and achieve 65.88% branch and 72.88% tissue accuracy. The 24-label tissue decoder improves classification, but cross-stage errors in the temporal protocol remain substantial. Any additional classifier tuning or label merging should preserve a declared evaluation split and report per-class performance and confidence/coverage. The `Ubi` component, which spans 46 branch IDs, remains a possible targeted flow pilot.

4. **Use a common-root endpoint pilot to test the implementation.** The 0–2 h `Ubi_0` population has **5,477 cells** and the 14–16 h `Ubi` population **36,098 cells**. Every target branch is reachable from that root in the candidate hierarchy, so this endpoint pair has 100% supported label-level transport mass. Evaluate interpolation against the **48,630 `Ubi` cells at 8–10 h**. This avoids an immediate endpoint-support failure while testing a broad, deep branch structure. It does not establish that all intermediate raw proportions are consistent with a hard irreversible prior. For a baseline with informative endpoint restrictions, the subsequent [time-point audit](timepoint_selection.md) instead recommends **4–6 → 12–14 h**, primarily using the coarser tissue relation; the root pilot's endpoint-only score is automatically valid.

5. **Make the information regime explicit.** An intermediate-state classifier trained from other atlas bins provides reference information beyond the two flow endpoint marginals. State this when presenting an endpoint pilot. The published ages/tree themselves are also atlas-derived. A stricter benchmark would need a separate reference atlas or a redesigned collection-level split and refitting strategy.

6. **Proceed to the reviewer experiment after the pilot.** Compare the existing unconstrained and prior-based flow/flow-map variants with matched flow-training data, representations, sampling rules, and budgets. Use a curated multi-time schedule or an explicitly declared endpoint/reference-atlas protocol. Report held-out PCA-space Wasserstein/EMD, cell/state composition, and path violations relative to the fixed prior, with classifier uncertainty and label coverage. A graph-derived violation rate demonstrates agreement with the prior; it does not measure true clonal fate accuracy. There are no lineage-tracing clone labels in this release.

For a multi-time experiment with strict balanced matching, rerun the compatibility audit after curation. If substantial incompatibility remains, a soft-prior demonstration is plausible within the existing method family; exact marginal matching plus a completely hard graph would require further modeling assumptions.

## 8. Integration and resource implications

The core reachability code accepts arbitrary edge and class-name lists, and the flow architecture operates on the PCA representation. Sparse counts, PCA100 representations and classifier checkpoints are prepared; the MLP exports were verified through the existing NumPy classifier loader. A dedicated Calderon endpoint loader, native OT flow-map configs, classifier-based trajectory evaluation and post-training controls are implemented. Full plain-OT, filtered-interpolant and constrained/filtered runs have completed a matched comparison. The native lineage penalty reduces raw violations but worsens marginal fit at weight 1000; prior curation and constraint-strength selection remain relevant for the reviewer experiment.

The raw matrix is only about **1.62% nonzero**, but has 212.7 million entries. Densifying all 23,932 features would require approximately **48.8 GiB at float32** before intermediates. Sparse reading/conversion can still consume several GiB, so preprocessing needs adequate CPU memory and careful handling. In contrast, all 547,805 cells in 50 float32 PCs occupy only **104.5 MiB**, and the 314,387 assigned cells occupy about 60 MiB. The training representation is manageable; full-population pairwise OT is the expensive part. Use the existing minibatch-OT approach and fixed-size evaluation samples.

**Decision:** pursue an RNA pilot with a curated prior and explicit classifier uncertainty. The release provides a concrete route to an additional experiment with much more branching diversity. The completed classifier comparison favors the coarser labels, while showing that cross-stage generalization needs improvement. Biological/state-label curation, coupling support and flow performance remain to be resolved before choosing the strongest hard-constraint variants.

## Evidence boundaries

Completed: reading the supplied paper and Fig. 3; inspecting the public release and GEO records; parsing both cell metadata files and the full lineage workbook; verifying all joins, dimensions, label counts, age agreement, and class support; constructing and auditing a candidate prefix hierarchy; comparing the local benchmark exports and graph implementations; downloading and validating the complete RNA matrix; exporting the lineage-assigned sparse counts and training-fitted HVG3000/PCA100 representations. Classifier results are recorded separately in [classification.md](classification.md).

The original inspection results remain an audit of data availability and structural feasibility; the classifier reports and [OT flow-map results](ot_flow_map_results.md) record subsequent experiments. The flow pilot establishes execution and measured marginal/trajectory diagnostics for one seed. Biological validity of every inferred edge, true clone-fate accuracy and the exact weighted Fig. 3C RNA edge export remain unestablished.
