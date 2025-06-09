## 2025_6_8 release
We have updated protloc_mex_X to version 0.0.27 (the stable version for paper reproduction remains 0.0.17).
The main update is as follows:

User-specified CUDA device support:
protloc_mex_X now allows users to specify the CUDA device number for model inference.
For details, please refer to the PyPI page: https://pypi.org/project/protloc-mex-x/

⚠️ Due to technical limitations, even when a specific CUDA device is assigned, a small portion of GPU memory on CUDA:0 may still be occupied in certain scenarios.

## 2025_5_30 release

Leveraging the advances introduced by the newest [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) model, we can  generalize our [feature-representation methods]( https://doi.org/10.1093/bib/bbad534)—including CLS/EOS token embeddings, global mean pooling, and residue-specific extraction—to this state-of-the-art model. 

Please refer to the https://github.com/luozeyu1024/Protloc_mex_basline_model for more detail
