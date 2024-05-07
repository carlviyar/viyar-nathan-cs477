# CS477 Final Project for Carl Viyar and Christopher Nathan

We ran our Jupyter notebook on the Grace Computer Cluster maintained by the Yale Center for Research Computing. To run the code, simply run each cell in order. The specifics for our computing environment were as follows:
- 10 CPU cores of 8GB each
- 3 GPUs of 64GB total
The HPC allows for job submission, so we used `papermill` to run the jupyter notebook asynchronously within a `conda` environment containing our necessary libraries.

Set up the environment using:

`module load miniconda`

`conda create -n [env_name] [packages separated by spaces]`

Run the job using:

`module load miniconda`

`conda activate [env_name]`

`conda install papermill`

`papermill path/to/notebook/final_project.ipynb path/to/output/output.ipynb`

The following libraries were used and installed (see the first cell the notebook):
- torch (2.3.0)
- numpy (1.26.4)
- datasets (2.19.0)
- torchtext (0.18.0)
- tqdm (4.66.2)
- matplotlib (3.8.4)
- huggingface_hub (0.22.2)
- transformers (4.40.1)
- [flair](https://flairnlp.github.io/) (0.13.1)
- scipy==1.12.0 (1.12.0)

