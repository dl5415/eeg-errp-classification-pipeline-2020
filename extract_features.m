%% Extract Features Function

function [features, labels] = extract_features(trials, fs, trial_timing)
    % global values
    electrodes = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
    downsamp_factor = 8;
    %PSD parameters
    freq_res = 0.25; 
    %Only interested in 1-10Hz
    num_bin = 10 / freq_res;
    interval_len = 0.05; %in s
    DWT = 'db8'; 
    %From wmaxlev to avoid corner effects
    level = 4;
    
    % extract features
    feat_avg = get_avg(trials); %MS
    feat_std = get_std(trials); %std
    feat_max = get_max(trials); %Max
    feat_min = get_min(trials); %Min
    [feat_samp, n_samples] = get_samp(trials, downsamp_factor); %Subsamples
    [feat_PSD, W] = PSD_estimate_v2 (trials, fs, num_bin, freq_res); %Extract PSD
    [feat_interval, feat_len] = interval_feat_v2 (trials, fs, interval_len); %Interval feat
    [feat_DWT] = DWT_feat_v2 (trials, DWT, level); %DWT features
    features = cat(2, feat_avg, feat_std, feat_max, feat_min, feat_samp, feat_PSD, feat_interval, feat_DWT);

    % feature labels
    labels_avg = strcat(electrodes, {' Avg'});
    labels_std = strcat(electrodes, {' Std'});
    labels_max = strcat(electrodes, {' Max'});
    labels_min = strcat(electrodes, {' Min'});
    labels_samp = [];
    for i = 1:n_samples
        point = 1000*(trial_timing(1) + (i-1)*downsamp_factor/fs);
        labels_samp = horzcat(labels_samp, strcat(electrodes, {' Samp'}, int2str(i), {' @'}, num2str(point, '%.0f'), 'ms')); %#ok<AGROW>
    end
    labels_PSD = [];
    for i = 1:length(electrodes)
        labels_PSD_temp = [];
        for j = 1:num_bin
            labels_PSD_temp{j} = strcat(electrodes{i}, ' bin_# ', int2str(j));
        end
        labels_PSD = horzcat(labels_PSD,labels_PSD_temp); 
    end
    labels_interval = [];
    for i = 1:length(electrodes)
        labels_interval_temp = [];
        for j = 1:feat_len
            labels_interval_temp{j} = strcat(electrodes{i}, ' inter_# ', int2str(j));
        end
        labels_interval = horzcat(labels_interval,labels_interval_temp); 
    end
    
    labels_DWT = [];
    for i = 1:length(electrodes)
        labels_DWT_temp = [];
        for j = 1:(size(trials, 3) + 3)
            labels_DWT_temp{j} = strcat(electrodes{i}, ' DWT_# ', int2str(j));
        end
        labels_DWT = horzcat(labels_DWT,labels_DWT_temp); 
    end
    
    labels = horzcat(labels_avg, labels_std, labels_max, labels_min, labels_samp, labels_PSD, labels_interval, labels_DWT);
end

%% Helper Functions

function [feature, n_samples] = get_samp(trials, factor)
    samp = trials(:,:,1:factor:end);
    samp_dim = size(samp);
    n_samples = samp_dim(3);
    
    feature = [];
    for i = 1:n_samples
        feature = cat(2, feature, samp(:,:,i));
    end
end

function feature = get_min(trials)
    feature = min(trials, [], 3);
end

function feature = get_max(trials)
    feature = max(trials, [], 3);
end

function feature = get_std(trials)
    feature = std(trials, 0, 3);
end

function feature = get_avg(trials)
    feature = mean(trials.^2, 3);
end
