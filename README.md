# LingoVision

## Introduction

An Unknown Device Identification Method Based on LLM and RAG.

## Model Selected

* Embeding Model: mxbai-embed-large-v1-f16.llamafile in size of 5.4G
* Chatting Model: Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile in size of 667M

## Usage

### Run Model Servers

* Embeding Model Server:

```
mxbai-embed-large-v1-f16.llamafile --server --port 8080 --nobrowser
```

* Chatting Model Server:

```
Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile --server --port 8080 --nobrowser
```

### Create Index for RAG

1. Put chunked RAG data to `rag_data` directory in text format.
2. Delete `rag_index` directory if exists.
3. Execute `create_index.py` and generate index for RAG data.

### Test Performance

Execute `lingo_vision.py` to test the performance of device type identification. This will use test dataset in `dataset` directory and generate results to `result` directory.

### Generate Picture

Test results contains confusion matrix in CSV and classification report in JSON. We can transform them to picture by:

```
python conf_picture.py confusion_matrix.csv conf_output.png
```

```
python class_picture.py classification_report.json class_output.png
```