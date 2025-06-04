<p align="center">
  <img src="https://github.com/Smabbler/Docs/blob/main/smabbler_logo.png" alt="Logo"/>
</p>

## <p align="center">Python Usage Example</p>

This repository contains a simple example of how to use **Smabbler Galaxia** to extract structured features from natural language text using a pre-existing model. The example processes veterinary case descriptions to identify symptoms and diseases.

To extract other types of information, you can select a different pre-existing model (`algorithm_version`) or create your own using your data via the Smabbler Portal: [https://beta.smabbler.com](https://beta.cloud.smabbler.com)

**Setup:**
`cd` into the repository, create a virtual environment, and install required dependencies:

```
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

To run this example you need to either get access to the API and edit the `example_1.py` file to add the API key: ADD_INSTRUCTIONS

**Usage:**
to run the example (it will take a few minutes to run; exceptions will be printed if processing of particular examples will fail --- the script will retry to process failed examples a few times):
```
./example_1.py
```
