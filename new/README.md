# Smart Home Energy Consumption Prediction

A simple notebook-based ML assignment. Start with **01_Data_Cleaning_and_EDA.ipynb**, then **02_Model_Training_and_Evaluation.ipynb**. Both include saved outputs and explanations.

## Files

- `01_Data_Cleaning_and_EDA.ipynb`: problem, dataset, cleaning, EDA and feature engineering (PDF stages 1–7).
- `02_Model_Training_and_Evaluation.ipynb`: baselines, two ML algorithms, tuning, validation, results, limitations, viva notes and a guideline checklist (stages 8–12).
- `data/energydata_complete.csv`: original supplied UCI dataset, unchanged.
- `data/clean_energydata.csv`: validated sensor table with the two artificial random columns removed.
- `data/model_ready_energydata.csv`: eleven predictors, timestamps and next-interval target; six boundary/history rows excluded.

## Run

Use Python 3.12 or newer (verified with Python 3.14.8). The numerical library versions are pinned to the executed run. From this project folder:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook
```

Select the virtual environment's Python kernel. Open Notebook 1 and use **Run All**, then do the same for Notebook 2. Each notebook is self-contained; neither imports local Python scripts. Notebook 1 recreates the cleaned CSVs from the included original. Notebook 2 trains in memory and includes an editable prediction example. Its small search fits four forest configurations across three time folds; runtime depends on your machine.

All charts, tuning results, metrics and explanations are inside the notebooks. The second notebook includes the guideline checklist and a short viva explanation. No HTML pages, web server or model files are required.

The previous complex version is preserved in `../old/`. Use this `new/` folder for the simplified notebook assignment. Run the commands above from inside `new/` so notebook dataset paths resolve correctly.

Dataset: Candanedo, L. (2017), [Appliances Energy Prediction, UCI](https://doi.org/10.24432/C5VC8G), CC BY 4.0. See Notebook 1 for collection details and Notebook 2 for sources and AI assistance disclosure.
