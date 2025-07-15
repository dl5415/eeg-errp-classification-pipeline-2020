%% Feature Selection Process Function

function results = feature_selection_process(processed_data, run_divisions, frac_best)    
    % retrieve processed data
    fs = processed_data.fs;
    trial_timing = processed_data.trial_timing;
    trials = processed_data.trials;
    trig_vals = processed_data.trig_vals.' + 1;
    
    % extract features for each run (assumes 4 runs)
    best_indices = [];
    for i = 1:4
        run_trials = trials(run_divisions(i)+1:run_divisions(i+1),:,:);
        run_trig_vals = trig_vals(run_divisions(i)+1:run_divisions(i+1),:);
        [run_features, ~] = extract_features(run_trials, fs, trial_timing);
        
        % determine most discriminant features
        run_feat_size = size(run_features);
        num_best = round(frac_best*run_feat_size(2));
        run_fisher = fsFisher(run_features, run_trig_vals);
        best_indices(i,:) = run_fisher.fList(1:num_best); %#ok<AGROW>
    end
    
    % determine stable features (assumes 4 runs)
    stable_indices = intersect(intersect(best_indices(4,:), best_indices(3,:)), intersect(best_indices(2,:), best_indices(1,:)));
    [features, labels] = extract_features(trials, fs, trial_timing);
    fisher = fsFisher(features, trig_vals);
    stable_features = features(:,stable_indices);
    stable_labels = labels(stable_indices);

    % store results
    results = struct;
    results.features = features;
    results.labels = labels;
    results.fisher = fisher;
    results.frac_best = frac_best;
    results.stable_indices = stable_indices;
    results.stable_features = stable_features;
    results.stable_labels = stable_labels;
    results.X = stable_features;
    results.y = trig_vals;
end
