# Error-related potential (ErrP) classification from EEG (2020)

This repository contains a full signal processing and machine learning pipeline for decoding error-related potentials (ErrPs), from 32-channel EEG data, in a BCI speller application built to
provide an alternative communication avenue to patients with severe motor disabilities. 

I am fortunate to have collaborated with Jonathan Madera and Jackson Lightfoot on this project, who built the pipeline with me.

preprocessing.m: does preprocessing on raw EEG and features engineering (extraction & selection), runs load_raw_data.m, process_raw_data.m, feature_extraction_process.m and feature_selection_process.m

ERP_classify_LDA.m, ERP_classify_NN.m, ERP_classify_SVM_v3.m: buils different classification models, tunes hyperparameters via cross validation and evaluates models on test set. 

Project_report.pdf: A report that summarizes the model's performance and our key findings
