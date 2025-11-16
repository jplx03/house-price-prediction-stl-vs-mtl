# house-price-prediction-stl-vs-mtl
323 group project on house price prediction

dataset: kc_house_data.csv file taken from kaggle site: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data 

Overview: 
The project evaluates house price prediction performance under three learning approaches:
1. Ridge Regression (STL)
2. Random Forest (STL)
3. L2,1-regularised MTL

The 'House Sales in King County, USA' dataset is used, with additional preprocessing and feature engineering. The goal is to compare STL vs MTL under consistent evaluation setup. 

## Use of GenAI: 
We acknowledge that generative AI tools (e.g. ChatGPT, Perplexity) were used as supplementary aid during the development of the MTL model. Since the research paper referenced did not provide an official codebase, we relied on its mathematical formulation and used generative AI only to help clarify how certain components (such as the L₂,₁ norm, task-specific weight selection, optimisation loop, etc.) could be implemented in practice. <br>
The final MTL implementation was written, adapted, and refined by our group based on our own understanding of the paper. No code was copied directly without modification. AI-generated examples served only as conceptual guidance, which we restructured to fit our dataset, architecture design, and training procedure.


## To run the notebook:
To reproduce the exact outputs (preprocessing figures, clusters, model metrics and the final comparison table), run:
```
stl_vs_mtl_house_price_pred.ipynb
```
from the first cell to the last cell in a clean environment. 
All randomness is controlled via fixed seeds, so the results will be identical.


## Environment setup
required libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch

the dataset is stored in this repository: 
```
dataset/kc_house_data.csv
``

to load the dataset directly from github, run the cell:
```
import pandas as pd
url = "https://raw.githubusercontent.com/jplx03/house-price-prediction-stl-vs-mtl/main/dataset/kc_house_data.csv"
df = pd.read_csv(url)
```


## Reproducibility
All the figures and results in this project are exactly reproducible because we enforced a global random seed with the function definition in the code:
```
seed_everything(42)
```
This controls the randomness in NumPy, scikit-learn, PyTorch, CUDA and OS threads


