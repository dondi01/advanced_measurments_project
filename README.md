# Relevant Branches

* **main**: here is the code described in section 3-4-5-6-7 of the report
* **fault-detection**: the branch which is a variant of the main project where we experimented with using Convolutional Neural Networks for fault detection

---

## The main branch

* Most of the files are there to be able to run `_main.py_` and deal with particular steps delineated in section 3 to 5.
* `_Probabilistic_analysis.py_` is the file that implements the probabilistic model discussed in section 6.
* `_validation.py_` and `_calculate_validation_metrics.py_` are the files that where used to statistically validate the algoriithms proposed in the **main** branch

---

## The fault-detection branch

* `_keras_test.py_` is the file where models are trained and designed
* `_functions/ml_functions.py_` is the custom library implemented for most files in this branch, also used in the main branch for validation.
* `_create_windowed_dataset.py_` is the file where the datasets used are generated
* `_cnn_predict.py_` is the file where the trained models are actually tested

In general for this branch there are a lot of files for building, running and managing a custom docker container for harnessing the GPU processing power.
