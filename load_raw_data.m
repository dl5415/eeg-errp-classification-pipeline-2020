%% Load Raw Data Function

function results = load_raw_data(subject, session)
    % global values
    data_dir = 'ErrPSpeller';
    skip_last = ["Subject2 Offline", "Subject5 Offline"];
    
    % load data
    data_path = fullfile(data_dir, subject, session);
    data_path = strcat('./', data_path);
    [signal, event, header] = loadData(data_path);
    
    session_name = strcat(subject, {' '}, session);
    skip = 0;
    for i = 1:length(skip_last)
        if convertCharsToStrings(session_name) == skip_last(i)
            skip = 1;
        end
    end
    
    % store data
    results = struct;
    results.session_name = session_name;
    results.signal = signal(:,1:16);
    results.fs = header.EVENT.SampleRate;
    results.trig_pos = event.position(:,1:end-skip);
    results.trig_vals = event.type(:,1:end-skip);
