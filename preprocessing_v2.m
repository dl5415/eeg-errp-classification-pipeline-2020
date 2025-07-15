%% ErrP Project Driver
clc; clear; 
% global values
subjects = {'Subject1', 'Subject2', 'Subject4', 'Subject5'};
sessions = {'Offline', 'S2', 'S3'};
% subjects = {'Subject1'};
% sessions = {'Offline'};

%% Load Raw Data

raw_data = struct;

for subject = 1:length(subjects)
    for session = 1:length(sessions)
        subject_name = subjects{subject};
        session_name = sessions{session};
        raw_data.(subject_name).(session_name) = load_raw_data(subject_name, session_name);
    end
end

save('data/raw_data.mat', 'raw_data');

%% Process Raw Data

load('data/raw_data.mat')
processed_data = struct;

for subject = 1:length(subjects)
    for session = 1:length(sessions)
        subject_name = subjects{subject};
        session_name = sessions{session};
        processed_data.(subject_name).(session_name) = process_raw_data(raw_data.(subject_name).(session_name));
    end
end

save('data/processed_data.mat', 'processed_data');

%% Feature Selection Process

frac_best_values = [0.02 0.03 0.06 0.08 0.09 0.10 0.15 0.20 0.22];
dataset_labels = {'size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7', 'size8', 'size9'};
load('data/processed_data.mat');
run_divisions = [
    [0 125 250 375 500]
    [0 125 250 375 499]
    [0 125 250 375 500]
    [0 125 250 375 499]];
training_data = struct;

% done only for offline sessions
for subject = 1:length(subjects)
    subject_name = subjects{subject};
    for dataset = 1:length(dataset_labels)
        frac_best = frac_best_values(dataset);
        label = dataset_labels{dataset};
        training_data.(subject_name).(label) = feature_selection_process(processed_data.(subject_name).Offline, run_divisions(subject,:), frac_best);
    end
end

save('data/training_data_v2.mat', 'training_data');

%% Feature Extraction Process

dataset_labels = {'size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7', 'size8', 'size9'};
load('data/processed_data.mat')
load('data/training_data_v2.mat')
test_data = struct;

% done only for online sessions
for subject = 1:length(subjects)
    for session = 2:length(sessions)
        subject_name = subjects{subject};
        session_name = sessions{session};
        for dataset = 1:length(dataset_labels)
            label = dataset_labels{dataset};
            test_data.(subject_name).(session_name).(label) = feature_extraction_process(processed_data.(subject_name).(session_name), training_data.(subject_name).(label).stable_indices);
        end
    end
end

save('data/test_data_v2.mat', 'test_data');
