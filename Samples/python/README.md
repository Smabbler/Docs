This repository contains a simple example on how to use Smabbler Galaxia to extract a features from natural text, and use them to build and evaluate a Machine Learning model. In this particular case we extract information about symptoms and diseases from descriptions of veterinary cases. Other information can be extracted by selecting a different pre-existing Smabbler model ("algorithm_version") or creating one on your own to suit your specific needs on our website: https://www.smabbler.com/alpha (currently in alpha).

**Setup:**
`cd` into the repository, create a virtual environment, and install required dependencies:

`python3 -m venv .`
`source bin/activate`
`pip install -r requirements.txt`

To run this example you need to either get access to the API: ~~instruction link~~
or change the offline flag to `True` at the end of the script:
`offline = False`

**Usage:**
to run the example (it will take a few minutes to run, exceptions will be printed if processing of particular examples will fail):
`./example_1.py`

We have limited the number of datapoints to make this example fast to run. In order to run it on full dataset, comment out the following lines:
```
    df_tr = df_tr[df_tr['LabelID'].isin(['ASC99', 'ASC165'])].copy()
    df_tt = df_tt[df_tt['LabelID'].isin(['ASC99', 'ASC165'])].copy()
```

