%% Feature Extraction Process Function

function results = feature_extraction_process(processed_data, stable_indices)    
    % retrieve processed data
    fs = processed_data.fs;
    trial_timing = processed_data.trial_timing;
    trials = processed_data.trials;
    trig_vals = processed_data.trig_vals.' + 1;
    
    % extract stable features
    [features, labels] = extract_features(trials, fs, trial_timing);
    fisher = fsFisher(features, trig_vals);
    stable_features = features(:,stable_indices);
    stable_labels = labels(stable_indices);
    
    % store results
    results = struct;
    results.features = features;
    results.labels = labels;
    results.fisher = fisher;
    results.stable_indices = stable_indices;
    results.stable_features = stable_features;
    results.stable_labels = stable_labels;
    results.X = stable_features;
    results.y = trig_vals;
end
