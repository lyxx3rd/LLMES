# LLMES
LLMES: source code and datasets

## The QMS data
The QMS sample - MB_QMS loacted in LLMES/QMS

## How to Audit
### prepartion
1. unzip LLMES/QMS/MB.zip
2. run LLMES/Model/download_{model_name}.ipynb to download pre-trained model which you need.
3. run LLMES/Model/Classificater_train.py to train Classificater model
4. run Classificater_dev.ipynb to dev trained model(optional).

### Audit
1. run requirements_processing.py to deal with the selected requirements.
2. run processing_QMS.py to process QMS documents.
3. run enhanced_data_RAG.ipynb to enhanced sub_requirements and RAG relation chuncks
4. run Classificater.py to launch a request serve which is used for classify result to label
5. run Direct_Review.py or Expert_Syste_Review.py to Directly audit or ES auidt

LLM is function to use different LLM quickly for audit.