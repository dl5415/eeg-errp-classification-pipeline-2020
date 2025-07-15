%% Process Data Function

function results = process_raw_data(raw_data)
    % retrieve raw data
    session_name = raw_data.session_name;
    signal = raw_data.signal;
    fs = raw_data.fs;
    trig_pos = raw_data.trig_pos;
    trig_vals = raw_data.trig_vals;
    
    % spatial filter
    signal = car_filter_signal(signal);
    
    % frequency filter
    freq_range = [1 10];
    N = 3;
    signal = freq_filter_signal(signal, fs, freq_range, N);
    
    % get trials
    trial_timing = [.2 .8];
    t = (round(trial_timing(1)*fs):round(trial_timing(2)*fs))/fs;
    trials = get_trials(signal.', trig_pos, trial_timing, fs);
    
    % store results
    results = struct;
    results.session_name = session_name;
    results.fs = fs;
    results.trial_timing = trial_timing;
    results.t = t;
    results.trials = trials;
    results.trig_vals = trig_vals;
end

%% Helper Functions

function trials = get_trials(signal, trigger_position, trial_timing, fs)
    trial_indices = cat(1, trigger_position + round(trial_timing(1)*fs), trigger_position + round(trial_timing(2)*fs));
    trials = [];
    
    for i = 1:length(trigger_position)
        trial(1,:,:) = signal(:,trial_indices(1,i):trial_indices(2,i));
        trials = cat(1, trials, trial);
    end
end

function filtered = freq_filter_signal(signal, fs, freq_range, N)
    [B, A] = butter(N, freq_range*2/fs);
    filtered = filter(B, A, signal);
end

function filtered = car_filter_signal(signal)
    filtered = signal - mean(signal, 2);
end
