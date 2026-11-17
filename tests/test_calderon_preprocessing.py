"""Numerical and indexing checks for the large sparse Calderon pipeline."""
import gzip
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from scripts import prepare_calderon_lineage_data as preparation


class CalderonPreprocessingTests(unittest.TestCase):
    def test_streaming_preserves_selected_cells_and_integer_counts(self):
        text=("%%MatrixMarket matrix coordinate integer general\n3 4 7\n"
              "1 1 1\n3 1 2\n2 2 4\n3 2 1\n1 3 5\n2 4 2\n3 4 3\n")
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/"counts.mtx.gz"
            with gzip.open(source,"wt") as handle:handle.write(text)
            actual,audit=preparation.stream_retained_counts(
                source,np.array([False,True,False,True]),np.array([2,2]),folder/"csr",3,progress=False)
            np.testing.assert_array_equal(actual.toarray(),[[0,4,1],[0,2,3]])
            self.assertEqual(audit,{"source_entries":7,"source_umis":18,"retained_entries":4})

    def test_streaming_rejects_unsorted_cells(self):
        text="%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 2 3\n1 1 2\n"
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/"counts.mtx.gz"
            with gzip.open(source,"wt") as handle:handle.write(text)
            with self.assertRaisesRegex(ValueError,"column-major"):
                preparation.stream_retained_counts(source,np.array([True,True]),np.array([1,1]),folder/"csr",2,progress=False)

    def test_blockwise_vst_matches_scanpy_seurat_v3(self):
        import scanpy as sc
        rng=np.random.default_rng(12)
        rates=np.exp(rng.uniform(-2.5,2.5,size=400))
        counts=rng.poisson(rates,size=(2000,400)).astype(np.int32);counts[:,0]=0
        matrix=sparse.csr_matrix(counts);train=np.arange(2000)<1600
        selected,stats=preparation.seurat_v3_hvgs(matrix,train,50)
        reference=ad.AnnData(X=matrix[train])
        sc.pp.highly_variable_genes(reference,flavor="seurat_v3",n_top_genes=50)
        np.testing.assert_allclose(stats.vst_normalized_variance,reference.var.variances_norm,rtol=1e-9,atol=1e-10)
        np.testing.assert_array_equal(selected,np.flatnonzero(reference.var.highly_variable.to_numpy()))

    def test_implicit_scaled_pca_matches_dense_train_only_pca(self):
        rng=np.random.default_rng(4)
        expression=np.log1p(rng.poisson(np.exp(rng.uniform(-1,2,20)),size=(300,20))).astype(np.float32)
        train=np.arange(300)<210
        scores,mean,scale,components,variance=preparation.fit_scaled_pca(sparse.csr_matrix(expression),train,5)
        training=expression[train].astype(np.float64)
        reference=PCA(n_components=5,svd_solver="full").fit((training-training.mean(0))/training.std(0,ddof=1))
        np.testing.assert_allclose(variance,reference.explained_variance_,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(components.T@components,reference.components_.T@reference.components_,rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(scores,(expression-mean)/scale@components.T,rtol=1e-4,atol=2e-6)
        changed=expression.copy();changed[~train]+=10000
        _,changed_mean,changed_scale,changed_components,changed_variance=preparation.fit_scaled_pca(sparse.csr_matrix(changed),train,5)
        np.testing.assert_array_equal(mean,changed_mean)
        np.testing.assert_array_equal(scale,changed_scale)
        np.testing.assert_allclose(components,changed_components,atol=1e-12)
        np.testing.assert_array_equal(variance,changed_variance)


if __name__=="__main__":unittest.main()
