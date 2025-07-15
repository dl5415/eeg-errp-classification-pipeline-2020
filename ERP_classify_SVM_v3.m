%This is v2 of the ERP_classify_SVM script, without optimising the model (C
%and lambda), just optimising the feature space
close all
clear
clc
%Classification SVM (all subjects)
subjects = {'Subject1', 'Subject2', 'Subject4', 'Subject5'};
%subjects = {'Subject2'};
sessions = {'S2', 'S3'};
features_num = {'size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7', 'size8', 'size9'};
%CV parameters
KFold = 4; %Run wise
fold_size = 125; %Each run has 125 trials

%% CV experiment (on all subjects)(CHECKED)
%MATLAB builtin KFold does not respect temporality, hence need to parse
%training data manually. Also only metric available is classification
%accuracy (CV loss)
load('data/training_data_v2.mat')
for p = 1:length(subjects)
    for z = 1:length(features_num)
        X_train = training_data.(subjects{p}).(features_num{z}).X;
        Y_train = training_data.(subjects{p}).(features_num{z}).y;
        %Deal with empty features
        if (length(X_train) == 0)
            X_train = training_data.(subjects{p}).(features_num{z+1}).X;
        end
        %label 1 -> ErrP trial
        Y_train = Y_train - 1;
        for k = 1:KFold %KFold = 4, run wise
           
            if (isequal(subjects{p}, 'Subject2') && (k == 4)) || (isequal(subjects{p}, 'Subject5') && (k == 4)) 
                end_indx = k * fold_size - 1 ;
            else
                end_indx = k * fold_size ;
            end
             %Validation set
            test_data = X_train ((k-1) * fold_size + 1 : end_indx, :);
            test_data_label = Y_train ((k-1) * fold_size + 1 : end_indx, :);
            %Omit validation set and use the rest as train data
            train_data = X_train;
            train_data((k-1) * fold_size + 1 : end_indx, :) = [];
            train_data_label = Y_train;
            train_data_label((k-1) * fold_size + 1 : end_indx, :) = [];
            %RF = fitcsvm(train_data,train_data_label, 'KernelFunction','linear');
            RF=TreeBagger(50,train_data,train_data_label,'OOBPred','On','Method','Classification');
            [output, ~] = predict(RF,test_data);
            output = str2double(output);
            %Performance on this fold
            [c,cm,ind,per] = confusion(test_data_label',output');
            Test_acc = mean(double(test_data_label == output));
            test_performance.acc = Test_acc;
            %TPR -> sensitivity, TNR -> specificity
            test_performance.sens = cm(2,2)/sum(cm(2,:));
            test_performance.spec = cm(1,1)/sum(cm(1,:));
            test_performance.FPR = cm(1,2)/sum(cm(1,:));
            kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(test_data_label)^2;
            test_performance.kapp = (Test_acc - kappa_pe) / (1 - kappa_pe);
            %Store performance
            CV_Subject(p).features_num(z).Fold(k) = test_performance;
        end
    end
end

%Save 
save('CV_results_RF.mat', 'CV_Subject');
%save('CV_results_rbf.mat', 'CV_Subject');

%% Select optimal parameters (CHECKED)

load('CV_results_RF.mat')
%load('CV_results_rbf.mat')
%Compute mean kappa
mean_kappa = zeros(length(subjects), length(features_num));
for p = 1:length(subjects)
    for z = 1:length(features_num)
        
        kapp = [CV_Subject(p).features_num(z).Fold.kapp];
        mean_kappa(p,z) = mean(kapp);
        
    end
end
%find optimal parameter for each subject
kappa_max = max(mean_kappa,[], 2);
for p = 1:length(subjects)
    idx = find(mean_kappa == kappa_max(p));
    s = size(mean_kappa);
    [I,J] = ind2sub(s,idx);
    Subject_opt(p).opt_features_size = features_num{J(1)}; 
end
save('Subject_opt_para_RF.mat', 'Subject_opt');
%save('Subject_opt_para_rbf.mat', 'Subject_opt');

%% Train again with the optimal parameters and online classification (CHECKED)
load('Subject_opt_para_RF.mat')
%load('Subject_opt_para_rbf.mat')
load('data/training_data_v2.mat')
load('data/test_data_v2.mat')
for i = 1:length(subjects)
        %Train on offline with optimal parameters
        X_train = training_data.(subjects{i}).(Subject_opt(i).opt_features_size).X;
        Y_train = training_data.(subjects{i}).(Subject_opt(i).opt_features_size).y;
        %label 1 -> ErrP trial
        Y_train = Y_train - 1;
        % Training
        RF=TreeBagger(50,X_train,Y_train,'OOBPred','On','Method','Classification');
        [out, ~] = predict(RF,X_train);
        out = str2double(out);
        Train_acc = mean(double(out == Y_train));
        [c,cm,ind,per] = confusion(Y_train',out');
        Subject_CVed(i).train_performance.acc = Train_acc;
        %TPR -> sensitivity, TNR -> specificity
        Subject_CVed(i).train_performance.sens = cm(2,2)/sum(cm(2,:)); 
        Subject_CVed(i).train_performance.spec = cm(1,1)/sum(cm(1,:));
        Subject_CVed(i).train_performance.FPR = cm(1,2)/sum(cm(1,:));
        kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(Y_train)^2;
        Subject_CVed(i).train_performance.kapp = (Train_acc - kappa_pe) / (1 - kappa_pe);
        %Testing - Subjects remain constant
        
        for p = 1:length(sessions)
            X_test = test_data.(subjects{i}).(sessions{p}).(Subject_opt(i).opt_features_size).X;
            Y_test = test_data.(subjects{i}).(sessions{p}).(Subject_opt(i).opt_features_size).y;
            Y_test = Y_test - 1;
            % Online sessions
            [out1, ~] = predict(RF,X_test);
            out1 = str2double(out1);
            [c,cm,ind,per] = confusion(Y_test',out1');
            %confusionchart(Y_test, out1);
            Test_acc = mean(double(out1 == Y_test));
            Subject_CVed(i).test_performance.(sessions{p}).acc = Test_acc;
            %TPR -> sensitivity, TNR -> specificity
            Subject_CVed(i).test_performance.(sessions{p}).sens = cm(2,2)/sum(cm(2,:)); 
            Subject_CVed(i).test_performance.(sessions{p}).spec = cm(1,1)/sum(cm(1,:));
            Subject_CVed(i).test_performance.(sessions{p}).FPR = cm(1,2)/sum(cm(1,:));
            kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(Y_test)^2;
            Subject_CVed(i).test_performance.(sessions{p}).kapp = (Test_acc - kappa_pe) / (1 - kappa_pe);
            confusionchart(Y_test, out1);
            saveas(gcf, strcat('RF_', 'Online_',subjects{i}, '_', sessions{p}));
           
        end
end

save('Subject_online_results_RF.mat', 'Subject_CVed');
%save('Subject_online_results_rbf.mat', 'Subject_CVed');
%% Chance level (CHECKED)
% Chance level decoding
load('data/test_data_v2.mat')
for i = 1:length(subjects)
    for p = 1:length(sessions)
        Y_test = test_data.(subjects{i}).(sessions{p}).('size1').y;
        %ErrP -> label 1
        Y_test = Y_test - 1;
        for z = 1 : 10000
            %shuffle Y_test, keep the distribution the same. 
            random_labels = Y_test(randperm(length(Y_test)));
            %random_labels = randi([0 1], length(Y_test), 1);
            [c,cm,ind,per] = confusion(Y_test',random_labels');
            %confusionchart(Y_test, out1);
            Test_acc = mean(double(random_labels == Y_test));
            Chance_Subject(i).test_performance.(sessions{p}).iter(z).acc = Test_acc;
            %TPR -> sensitivity, TNR -> specificity
            Chance_Subject(i).test_performance.(sessions{p}).iter(z).sens = cm(2,2)/sum(cm(2,:)); 
            Chance_Subject(i).test_performance.(sessions{p}).iter(z).spec = cm(1,1)/sum(cm(1,:));
            Chance_Subject(i).test_performance.(sessions{p}).iter(z).FPR = cm(1,2)/sum(cm(1,:));
            kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(Y_test)^2;
            Chance_Subject(i).test_performance.(sessions{p}).iter(z).kapp = (Test_acc - kappa_pe) / (1 - kappa_pe);

        end
    end
end

%Calculate mean for of chance level results
chance_perf_mean.kappa = zeros(length(subjects), length(sessions));
chance_perf_mean.sens = zeros(length(subjects), length(sessions));
chance_perf_mean.spec = zeros(length(subjects), length(sessions));
chance_perf_mean.FPR = zeros(length(subjects), length(sessions));
chance_perf_mean.acc = zeros(length(subjects), length(sessions));

for i = 1:length(subjects)
    for p = 1:length(sessions)
        kapp = [Chance_Subject(i).test_performance.(sessions{p}).iter.kapp];
        %kapp = rmoutliers(kapp);
        chance_perf_mean.kappa(i,p) = mean(kapp);
        sens = [Chance_Subject(i).test_performance.(sessions{p}).iter.sens];
        chance_perf_mean.sens (i,p) = mean(sens);
        spec = [Chance_Subject(i).test_performance.(sessions{p}).iter.spec];
        chance_perf_mean.spec (i,p) = mean(spec);
        FPR = [Chance_Subject(i).test_performance.(sessions{p}).iter.FPR];
        chance_perf_mean.FPR(i,p) = mean(FPR);
        acc = [Chance_Subject(i).test_performance.(sessions{p}).iter.acc];
        chance_perf_mean.acc(i,p) = mean(acc);
    end
end

save('Subject_online_chance_perf.mat', 'chance_perf_mean');
%% Data visualisation
%Plot 1, bar plot Kappa across subjects and sessions
figure;
load('Subject_online_results_RF.mat')
kappa_plot = zeros(length(subjects), length(sessions)+1);
for i = 1:length(subjects)
    kappa_plot(i,1) = Subject_CVed(i).train_performance.kapp;
    for p = 1:length(sessions)
        kappa_plot(i,p+1) = Subject_CVed(i).test_performance.(sessions{p}).kapp; 
    end
end
subjects_temp = categorical(subjects);
subjects_temp = reordercats(subjects_temp, subjects);
b = bar(subjects_temp, kappa_plot);
hold on
set(b, {'DisplayName'}, {'Offline', 'S2', 'S3'}');
legend()
ylabel('Kappa', 'fontweight', 'bold');
title('ErrP Classification Performance'); 
set(gca, 'FontSize', 12);

%Add chance decoding results

figure;
%Plot 2, scatterplot of TPR vs FPR
TPR_plot = zeros(length(subjects), length(sessions)+1);
FPR_plot = zeros(length(subjects), length(sessions)+1);
%Labels = zeros(length(subjects), length(sessions)+1);
for i = 1:length(subjects)
    TPR_plot(i,1) = Subject_CVed(i).train_performance.sens;
    FPR_plot(i,1) = Subject_CVed(i).train_performance.FPR;
    Labels{i,1} = strcat(subjects{i}, '_Offline');
    for p = 1:length(sessions)
        TPR_plot(i,p+1) = Subject_CVed(i).test_performance.(sessions{p}).sens; 
        FPR_plot(i,p+1) = Subject_CVed(i).test_performance.(sessions{p}).FPR;
        Labels{i,p+1} = strcat(subjects{i}, '_', sessions{p});
    end
end
TPR_plot = reshape(TPR_plot, [1,12]);
FPR_plot = reshape(FPR_plot, [1,12]);
Labels = reshape(Labels, [1,12]);
Mark_color = {'r', 'g', 'b', 'm'};
Mark_type = {'h', '<', 'o'};
k = 1;
for i = 1:length(subjects)
    for p = 1:length(sessions)+1
        scatter(FPR_plot(k), TPR_plot(k), 150, 'filled', Mark_color{i}, Mark_type{p});
        k = k + 1;
        hold on
    end
end

%% Miscellanous
% for k = 1:KFold
%             fold_size = ceil(length(X_train) / KFold);
%             %Validation set
%             test_data = X_train ((k-1) * fold_size + 1 : k * fold_size, :);
%             train_data = X_train;
%             %Omit validation set and use the rest as train data
%             train_data((k-1) * fold_size + 1 : k * fold_size, :) = [];
% end


% error_cross = mean(error_cross);
% error_cross_min = min(min(min(error_cross)));
% idx = find(error_cross == error_cross_min);
% s = size(error_cross);
% [I,J,K] = ind2sub(s,idx);
% opt_step = step(J(1));
% opt_lambda = lambda(K(1));














