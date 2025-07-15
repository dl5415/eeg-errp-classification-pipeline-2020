function [Pxx_feat, W] =  PSD_estimate_v2 (data, Fs, num_bins, freq_res)


%Desire longer window for better freq res
win_length = 0.4;
%win_length = [0.2, 0.4, 0.5, 0.58];
L = ceil(win_length * Fs);
%noverlap_pc=[0 0.25 0.5 0.75];
%Introduce some overlap to reduce variance
noverlap_pc = 0.6;
%noverlap = ceil(L * noverlap_pc);
%Calculate nfft to reach pre-defined freq res
nfft = ceil(Fs / freq_res);
Pxx_feat = zeros(size(data, 1), num_bins * size(data, 2)); 

for i = 1:size(data, 1)
    Pxx_feat_temp = []; 
    for j = 1:size(data, 2)
        data_temp = reshape(data(i,j,:), [1, size(data, 3)]);
        [Pxx, W] = pwelch(data_temp, rectwin(L), ceil(L * noverlap_pc) , nfft, Fs);
        Pxx = 10*log10(Pxx);
        %Pick the first num_bins as features
        Pxx = Pxx(1:num_bins);
        W = W(1:num_bins);
        Pxx_feat_temp = cat(2, Pxx_feat_temp, Pxx');
    end
    Pxx_feat(i,:) = Pxx_feat_temp;
end

end




