# Smart Home Energy Consumption Prediction

Two versions of the same ML assignment are preserved here:

| Folder | Contents | Use |
| --- | --- | --- |
| [new/](new/) | Two executed notebooks, original and cleaned datasets, minimal dependencies | Start here for the simplified assignment |
| [old/](old/) | Previous complete application, scripts, models, reports and HTML pages | Original version retained for reference |

## Run the simplified version

```bash
cd new
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook
```

Open **01_Data_Cleaning_and_EDA.ipynb**, then **02_Model_Training_and_Evaluation.ipynb**. Saved charts and results are included. Run notebooks with `new/` as the working directory. The notebooks contain the ML workflow, explanations, tuning, evaluation, local prediction example and guideline checklist.

See [new/README.md](new/README.md) for dataset descriptions and setup details. To use the previous application, change into `old/` before following [its instructions](old/README.md).
