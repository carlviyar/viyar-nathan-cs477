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

Dependencies included with the installation of the above packages:
- filelock (from torch) (3.13.4)
- typing-extensions>=4.8.0 (from torch) (4.11.0)
- sympy (from torch) (1.12)
- networkx (from torch) (3.3)
- jinja2 (from torch) (3.1.3)
- fsspec (from torch) (2024.3.1)
- nvidia-cuda-nvrtc-cu12==12.1.105 (from torch) (12.1.105)
- nvidia-cuda-runtime-cu12==12.1.105 (from torch) (12.1.105)
- nvidia-cuda-cupti-cu12==12.1.105 (from torch) (12.1.105)
- nvidia-cudnn-cu12==8.9.2.26 (from torch) (8.9.2.26)
- nvidia-cublas-cu12==12.1.3.1 (from torch) (12.1.3.1)
- nvidia-cufft-cu12==11.0.2.54 (from torch) (11.0.2.54)
- nvidia-curand-cu12==10.3.2.106 (from torch) (10.3.2.106)
- nvidia-cusolver-cu12==11.4.5.107 (from torch) (11.4.5.107)
- nvidia-cusparse-cu12==12.1.0.106 (from torch) (12.1.0.106)
- nvidia-nccl-cu12==2.20.5 (from torch) (2.20.5)
- nvidia-nvtx-cu12==12.1.105 (from torch) (12.1.105)
- nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.- 107->torch) (12.4.127)
- pyarrow>=12.0.0 (from datasets) (16.0.0)
- pyarrow-hotfix (from datasets) (0.6)
- dill<0.3.9,>=0.3.0 (from datasets) (0.3.8)
- pandas (from datasets) (2.2.2)
- requests>=2.19.0 (from datasets) (2.31.0)
- xxhash (from datasets) (3.4.1)
- multiprocess (from datasets) (0.70.16)
- aiohttp (from datasets) (3.9.5)
- packaging (from datasets) (24.0)
- pyyaml>=5.1 (from datasets) (6.0.1)
- contourpy>=1.0.1 (from matplotlib) (1.2.1)
- cycler>=0.10 (from matplotlib) (0.12.1)
- fonttools>=4.22.0 (from matplotlib) (4.51.0)
- kiwisolver>=1.3.1 (from matplotlib) (1.4.5)
- pillow>=8 (from matplotlib) (10.3.0)
- pyparsing>=2.3.1 (from matplotlib) (3.1.2)
- python-dateutil>=2.7 (from matplotlib) (2.9.0)
- regex!=2019.12.17 (from transformers) (2024.4.16)
- tokenizers<0.20,>=0.19 (from transformers) (0.19.1)
- safetensors>=0.4.1 (from transformers) (0.4.3)
- boto3>=1.20.27 (from flair) (1.34.98)
- bpemb>=0.3.2 (from flair) (0.3.5)
- conllu>=4.0 (from flair) (4.5.3)
- deprecated>=1.2.13 (from flair) (1.2.14)
- ftfy>=6.1.0 (from flair) (6.2.0)
- gdown>=4.4.0 (from flair) (5.1.0)
- gensim>=4.2.0 (from flair) (4.3.2)
- janome>=0.4.2 (from flair) (0.5.0)
- langdetect>=1.0.9 (from flair) (1.0.9)
- lxml>=4.8.0 (from flair) (5.2.1)
- more-itertools>=8.13.0 (from flair) (10.2.0)
- mpld3>=0.3 (from flair) (0.5.10)
- pptree>=3.1 (from flair) (3.1)
- pytorch-revgrad>=0.2.0 (from flair) (0.2.0)
- scikit-learn>=1.0.2 (from flair) (1.4.2)
- segtok>=1.5.11 (from flair) (1.5.11)
- sqlitedict>=2.0.0 (from flair) (2.1.0)
- tabulate>=0.8.10 (from flair) (0.9.0)
- transformer-smaller-training-vocab>=0.2.3 (from flair) (0.4.0)
- urllib3<2.0.0,>=1.0.0 (from flair) (1.26.18)
- wikipedia-api>=0.5.7 (from flair) (0.6.0)
- semver<4.0.0,>=3.0.0 (from flair) (3.0.2)
- botocore<1.35.0,>=1.34.98 (from boto3>=1.20.27->flair) (1.34.98)
- jmespath<2.0.0,>=0.7.1 (from boto3>=1.20.27->flair) (1.0.1)
- s3transfer<0.11.0,>=0.10.0 (from boto3>=1.20.27->flair) (0.10.1)
- sentencepiece (from bpemb>=0.3.2->flair) (0.2.0)
- wrapt<2,>=1.10 (from deprecated>=1.2.13->flair) (1.16.0)
- aiosignal>=1.1.2 (from aiohttp->datasets) (1.3.1)
- attrs>=17.3.0 (from aiohttp->datasets) (23.2.0)
- frozenlist>=1.1.1 (from aiohttp->datasets) (1.4.1)
- multidict<7.0,>=4.5 (from aiohttp->datasets) (6.0.5)
- yarl<2.0,>=1.0 (from aiohttp->datasets) (1.9.4)
- wcwidth<0.3.0,>=0.2.12 (from ftfy>=6.1.0->flair) (0.2.13)
- beautifulsoup4 (from gdown>=4.4.0->flair) (4.12.3)
- smart-open>=1.8.1 (from gensim>=4.2.0->flair) (6.4.0)
- six (from langdetect>=1.0.9->flair) (1.16.0)
- charset-normalizer<4,>=2 (from requests>=2.19.0->datasets) (3.3.2)
- idna<4,>=2.5 (from requests>=2.19.0->datasets) (3.7)
- certifi>=2017.4.17 (from requests>=2.19.0->datasets) (2024.2.2)
- joblib>=1.2.0 (from scikit-learn>=1.0.2->flair) (1.4.0)
- threadpoolctl>=2.0.0 (from scikit-learn>=1.0.2->flair) (3.5.0)
- protobuf (from transformers[sentencepiece]<5.0.0,>=4.18.0->flair) (5.26.1)
- MarkupSafe>=2.0 (from jinja2->torch) (2.1.5)
- pytz>=2020.1 (from pandas->datasets) (2024.1)
- tzdata>=2022.7 (from pandas->datasets) (2024.1)
- mpmath>=0.19 (from sympy->torch) (1.3.0)
- accelerate>=0.21.0 (from transformers[sentencepiece,torch]<5.0,>=4.- 1->transformer-smaller-training-vocab>=0.2.3->flair) (0.30.0)
- soupsieve>1.2 (from beautifulsoup4->gdown>=4.4.0->flair) (2.5)
- PySocks!=1.5.7,>=1.5.6 (from requests[socks]->gdown>=4.4.0->flair) (1.7.1)
- psutil (from accelerate>=0.21.0->transformers[sentencepiece,torch]- <5.0,>=4.1->transformer-smaller-training-vocab>=0.2.3->flair) (5.9.8)
