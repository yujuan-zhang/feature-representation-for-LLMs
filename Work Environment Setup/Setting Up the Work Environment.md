To ensure that you can replicate our work from the paper accurately, we recommend using the following work environment:

- Python version: 3.9.7
- protloc-mex-x version: 0.0.17
- protloc-mex1 version: 0.0.16

### Work Environment Setup

First, create a new conda environment. For Windows systems, it is recommended to use the conda Prompt for this task. On Linux systems, use the Terminal. (You can also modify the environment name as needed; here, we use "myenv" as an example):

```bash
conda create -n myenv python=3.9.7
```

Then, activate the environment you just created:

```bash
conda activate myenvs
```

Next, install a version of [PyTorch](https://pytorch.org/) that matches your device's configuration. If possible, we suggest using `"torch == 1.12.1"` to prevent potential compatibility issues.

Then, use pip to install 'protloc_mex1' within this environment:

```bash
pip install protloc_mex1==0.0.16
```

Finally, use pip to install 'protloc_mex_X' within this environment:

```bash
pip install protloc_mex_X==0.0.17
```

Other Python packages that are not automatically installed but also required (for a smoother replication of the work, we recommend using specific versions of libraries):

```python
dependencies = [
       "scikit-learn==1.2.2",
       "captum== 0.6.0",
       "transformers==4.26.1",
       "tqdm==4.63.0",
       "re==2.2.1"
]
```

With this, our basic work environment is set up. For more detail and related requirements, please refer to the information on [GitHub](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/README.md) and in our paper.
