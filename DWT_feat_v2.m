
function  features = DWT_feat_v2 (data, DWT, level)
    dwtmode('per');
    features = zeros(size(data, 1), (size(data, 3) + 3) * size(data, 2)); 
    for i = 1:size(data, 1) %trial
        interval_feat_temp_2 = [];
        for j = 1:size(data, 2) %channel
            sig = reshape(data(i,j,:), [1, size(data, 3)]);
            [DWT_out,~] = wavedec(sig,level,DWT);
            interval_feat_temp_2 = cat(2, interval_feat_temp_2, DWT_out);
     
        end
        
        features(i,:) = interval_feat_temp_2;
    end

        
end

