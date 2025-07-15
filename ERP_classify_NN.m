close all
clear
clc

subjects = {'Subject1', 'Subject2', 'Subject4', 'Subject5'};
%subjects = {'Subject2'};
sessions = {'S2', 'S3'};
features_num = {'small', 'medium', 'large'};
%Classification NN (all subjects) 
% Hidden Layer Optimization

KFold = 10;
hidden_layer_size = [10 25 50 100]; % Arbitrary value
lambda = [1e-3 1e0 1e3 1e6]; % Regularization hyperparameter can modify
num_labels = 2; % Number of classes ie Correct/Error trials
load('data/training_data.mat')
for p = 1:length(subjects)
    for z = 1:length(features_num)
        X_train = training_data.(subjects{p}).(features_num{z}).X;
        Y_train = training_data.(subjects{p}).(features_num{z}).y;
        %label 1 -> ErrP trial
        %Y_train = Y_train;= - 1;
        fold_size = floor(length(Y_train) / KFold);
        for i = 1:length(hidden_layer_size) 
            for j = 1:length(lambda) 
                for k = 1:KFold
                    %Validation set
                    test_data = X_train ((k-1) * fold_size + 1 : k * fold_size, :);
                    test_data_label = Y_train ((k-1) * fold_size + 1 : k * fold_size, :);
                    %Omit validation set and use the rest as train data
                    train_data = X_train;
                    train_data((k-1) * fold_size + 1 : k * fold_size, :) = [];
                    train_data_label = Y_train;
                    train_data_label((k-1) * fold_size + 1 : k * fold_size, :) = [];
                    input_layer_size = size(train_data,2);
                    % Starting NN Learning
                    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size(i));
                    initial_Theta2 = randInitializeWeights(hidden_layer_size(i), num_labels);
                    % Unroll parameters
                    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
                    % Implement and Train NN
                    options = optimset('MaxIter', 50);
                    % Create "short hand" for the cost function to be minimized
                    costFunction = @(p) nnCostFunction(p, ...
                        input_layer_size, ...
                        hidden_layer_size(i), ...
                        num_labels, train_data, train_data_label, lambda(j));
                    
                    % Now, costFunction is a function that takes in only one argument (the
                    % neural network parameters)
                    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
                    % Obtain Theta1 and Theta2 back from nn_params
                    Theta1 = reshape(nn_params(1:hidden_layer_size(i) * (input_layer_size + 1)), ...
                        hidden_layer_size(i), (input_layer_size + 1));
                    Theta2 = reshape(nn_params((1 + (hidden_layer_size(i) * (input_layer_size + 1))):end), ...
                        num_labels, (hidden_layer_size(i) + 1));
                    output = predict_nn(Theta1, Theta2, test_data);
                    %Performance on this fold
                    [c,cm,ind,per] = confusion(test_data_label' - 1,output' - 1);
                    Test_acc = mean(double(test_data_label == output));
                    test_performance.acc = Test_acc;
                    %TPR -> sensitivity, TNR -> specificity
                    test_performance.sens = cm(2,2)/sum(cm(2,:)); 
                    test_performance.spec = cm(1,1)/sum(cm(1,:));
                    kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(test_data_label)^2;
                    test_performance.kapp = (Test_acc - kappa_pe) / (1 - kappa_pe);
                    %Store performance
                    CV_Subject_NN_Rerun(p).features_num(z).HL(i).Lambda(j).Fold(k) = test_performance;
                end      
            end
        end
    end
end
save('CV_results_NN_Rerun.mat', 'CV_Subject_NN_Rerun');

%% Select optimal parameters
load('CV_results_NN_Rerun.mat')
mean_kappa = zeros(length(subjects), length(features_num), length(hidden_layer_size), length(lambda));
for p = 1:length(subjects)
    for z = 1:length(features_num)
        for i = 1:length(hidden_layer_size) 
            for j = 1:length(lambda) 
                kapp = [CV_Subject_NN_Rerun(p).features_num(z).HL(i).Lambda(j).Fold.kapp];
                mean_kappa(p,z,i,j) = mean(kapp);
            end
        end
    end
end

kappa_max = max(mean_kappa,[],[2 3 4]);
for p = 1:length(subjects)
    idx = find(mean_kappa == kappa_max(p));
    s = size(mean_kappa);
    [I,J,K,L] = ind2sub(s,idx);
    Subject_opt(p).opt_features_size = features_num{J(1)};
    Subject_opt(p).opt_HL = hidden_layer_size(K(1));
    Subject_opt(p).opt_Lambda = lambda(L(1));
end
save('Subject_opt_NN_P1.mat', 'Subject_opt');
save('mean_kappa_NN_P1.mat', 'mean_kappa');
%% Retrain and optimize on the folds using the HL and Lambda optimal parameters
% Might just re-run the first one with more folds instead afterwards to
% optimize all at once but for time sake, doing two suboptimal loops
load('Subject_opt_NN_P1.mat') % optimal parameters based on 3 sizes
load('CV_results_NN_Rerun_fold.mat')
subjects = {'Subject1', 'Subject2', 'Subject4', 'Subject5'};
%subjects = {'Subject2'};
sessions = {'S2', 'S3'};
features_num =  {'size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7', 'size8', 'size9', 'size10', 'size11', 'size12'};
%CV parameters
KFold = 4; %Run wise
fold_size = 125; %Each run has 125 trials
num_labels = 2; % Number of classes ie Correct/Error trials
hidden_layer_size = Subject_opt.opt_HL; % note used: will loop through subjects of struct
lambda = Subject_opt.opt_HL; % not used: will loop through subjects of struct
load('data/training_data_v2.mat')
for p = 1:length(subjects)
    for z = 10:length(features_num)
        X_train = training_data.(subjects{p}).(features_num{z}).X;
        Y_train = training_data.(subjects{p}).(features_num{z}).y;
        %label 1 -> ErrP trial
        %Y_train = Y_train;= - 1;
        % fold_size = floor(length(Y_train) / KFold);
        for k = 1:KFold
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
                    input_layer_size = size(train_data,2);
                    % Starting NN Learning
                    initial_Theta1 = randInitializeWeights(input_layer_size, Subject_opt(p).opt_HL);
                    initial_Theta2 = randInitializeWeights(Subject_opt(p).opt_HL, num_labels);
                    % Unroll parameters
                    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
                    % Implement and Train NN
                    options = optimset('MaxIter', 50);
                    % Create "short hand" for the cost function to be minimized
                    costFunction = @(u) nnCostFunction(u, ...
                        input_layer_size, ...
                        Subject_opt(p).opt_HL, ...
                        num_labels, train_data, train_data_label, Subject_opt(p).opt_Lambda);
                    
                    % Now, costFunction is a function that takes in only one argument (the
                    % neural network parameters)
                    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
                    % Obtain Theta1 and Theta2 back from nn_params
                    Theta1 = reshape(nn_params(1:Subject_opt(p).opt_HL * (input_layer_size + 1)), ...
                        Subject_opt(p).opt_HL, (input_layer_size + 1));
                    Theta2 = reshape(nn_params((1 + (Subject_opt(p).opt_HL * (input_layer_size + 1))):end), ...
                        num_labels, (Subject_opt(p).opt_HL + 1));
                    output = predict_nn(Theta1, Theta2, test_data);
                    %Performance on this fold
                    [c,cm,ind,per] = confusion(test_data_label' - 1,output' - 1);
                    Test_acc = mean(double(test_data_label == output));
                    test_performance.acc = Test_acc;
                    %TPR -> sensitivity, TNR -> specificity
                    test_performance.sens = cm(2,2)/sum(cm(2,:)); 
                    test_performance.spec = cm(1,1)/sum(cm(1,:));
                    kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(test_data_label)^2;
                    test_performance.kapp = (Test_acc - kappa_pe) / (1 - kappa_pe);
                    %Store performance
                    CV_Subject_NN_Rerun_fold(p).features_num(z).Fold(k) = test_performance;
        end
    end
end
save('CV_results_NN_Rerun_fold12.mat', 'CV_Subject_NN_Rerun_fold');

%% Find optimal kappa based on CV of folds
load('CV_results_NN_Rerun_fold12.mat')
mean_kappa = zeros(length(subjects), length(features_num));
for p = 1:length(subjects)
    for z = 1:length(features_num)
        kapp = [CV_Subject_NN_Rerun_fold(p).features_num(z).Fold.kapp];
        mean_kappa(p,z) = mean(kapp);
    end
end

kappa_max = max(mean_kappa,[],2);
for p = 1:length(subjects)
    idx = find(mean_kappa == kappa_max(p));
    s = size(mean_kappa);
    [I,J] = ind2sub(s,idx);
    Subject_opt_NN_fold(p).opt_features_size = features_num{J(1)};
end
save('Subject_opt_NN_P2.mat', 'Subject_opt_NN_fold');
save('mean_kappa_NN_P2.mat', 'mean_kappa');
%% Plotting mean kappa vs feature space

feature_space = zeros(size(mean_kappa));
feature_space(1,1) = size(training_data.Subject1.size1.stable_labels,2);
feature_space(1,2) = size(training_data.Subject1.size2.stable_labels,2);
feature_space(1,3) = size(training_data.Subject1.size3.stable_labels,2);
feature_space(1,4) = size(training_data.Subject1.size4.stable_labels,2);
feature_space(1,5) = size(training_data.Subject1.size5.stable_labels,2);
feature_space(1,6) = size(training_data.Subject1.size6.stable_labels,2);
feature_space(1,7) = size(training_data.Subject1.size7.stable_labels,2);
feature_space(1,8) = size(training_data.Subject1.size8.stable_labels,2);
feature_space(1,9) = size(training_data.Subject1.size9.stable_labels,2);
feature_space(2,1) = size(training_data.Subject2.size1.stable_labels,2);
feature_space(2,2) = size(training_data.Subject2.size2.stable_labels,2);
feature_space(2,3) = size(training_data.Subject2.size3.stable_labels,2);
feature_space(2,4) = size(training_data.Subject2.size4.stable_labels,2);
feature_space(2,5) = size(training_data.Subject2.size5.stable_labels,2);
feature_space(2,6) = size(training_data.Subject2.size6.stable_labels,2);
feature_space(2,7) = size(training_data.Subject2.size7.stable_labels,2);
feature_space(2,8) = size(training_data.Subject2.size8.stable_labels,2);
feature_space(2,9) = size(training_data.Subject2.size9.stable_labels,2);
feature_space(3,1) = size(training_data.Subject4.size1.stable_labels,2);
feature_space(3,2) = size(training_data.Subject4.size2.stable_labels,2);
feature_space(3,3) = size(training_data.Subject4.size3.stable_labels,2);
feature_space(3,4) = size(training_data.Subject4.size4.stable_labels,2);
feature_space(3,5) = size(training_data.Subject4.size5.stable_labels,2);
feature_space(3,6) = size(training_data.Subject4.size6.stable_labels,2);
feature_space(3,7) = size(training_data.Subject4.size7.stable_labels,2);
feature_space(3,8) = size(training_data.Subject4.size8.stable_labels,2);
feature_space(3,9) = size(training_data.Subject4.size9.stable_labels,2);
feature_space(4,1) = size(training_data.Subject5.size1.stable_labels,2);
feature_space(4,2) = size(training_data.Subject5.size2.stable_labels,2);
feature_space(4,3) = size(training_data.Subject5.size3.stable_labels,2);
feature_space(4,4) = size(training_data.Subject5.size4.stable_labels,2);
feature_space(4,5) = size(training_data.Subject5.size5.stable_labels,2);
feature_space(4,6) = size(training_data.Subject5.size6.stable_labels,2);
feature_space(4,7) = size(training_data.Subject5.size7.stable_labels,2);
feature_space(4,8) = size(training_data.Subject5.size8.stable_labels,2);
feature_space(4,9) = size(training_data.Subject5.size9.stable_labels,2);
[foo,ind] = max(mean_kappa,[],2);
figure()
hold on
color = {'r','g','b','c'};
for p = 1:length(subjects)
    plot(feature_space(p,:), mean_kappa(p,:),'Color',color{p},'LineWidth',1.5)
    xline(feature_space(p,ind(p)),'Color',color{p},'LineStyle','--','LineWidth',1.5)
end
legend('Subject1 Kappa','Subject1 Opt Num', 'Subject2 Kappa','Subject2 Opt Num', 'Subject4 Kappa','Subject4 Opt Num', 'Subject5 Kappa','Subject5 Opt Num')
grid on
xlabel('Number of featuers')
ylabel('Mean Kappa Across CV Folds')
title('Feature Space Size Effect on Performance (Neural Network)')

%% NN Classification of Test Set using optimal folds
load('data/training_data_v2.mat')
load('data/test_data_v2.mat')
load('Subject_opt_NN_P2.mat') % optimal folds
load('Subject_opt_NN_P1.mat') % optimal hidden layer & regularization
for i = 1:length(subjects)
    X_train = training_data.(subjects{i}).(Subject_opt_NN_fold(i).opt_features_size).X;
    Y_train = training_data.(subjects{i}).(Subject_opt_NN_fold(i).opt_features_size).y;
    input_layer_size = size(X_train,2);
    % Starting NN Learning
    initial_Theta1 = randInitializeWeights(input_layer_size, Subject_opt(i).opt_HL);
    initial_Theta2 = randInitializeWeights(Subject_opt(i).opt_HL, num_labels);
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    % Implement and Train NN
    options = optimset('MaxIter', 50);
    % Create "short hand" for the cost function to be minimized
    costFunction = @(u) nnCostFunction(u, ...
        input_layer_size, ...
        Subject_opt(i).opt_HL, ...
        num_labels, X_train, Y_train, Subject_opt(i).opt_Lambda);
    
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:Subject_opt(i).opt_HL * (input_layer_size + 1)), ...
        Subject_opt(i).opt_HL, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (Subject_opt(i).opt_HL * (input_layer_size + 1))):end), ...
        num_labels, (Subject_opt(i).opt_HL + 1));
    % prediction on training set
    output = predict_nn(Theta1, Theta2, X_train);
    Train_acc = mean(double(output == Y_train));
    [c,cm,ind,per] = confusion(Y_train' - 1,output' - 1);
    Subject_CVed(i).train_performance.acc = Train_acc;
    %TPR -> sensitivity, TNR -> specificity
    Subject_CVed(i).train_performance.sens = cm(2,2)/sum(cm(2,:));
    Subject_CVed(i).train_performance.spec = cm(1,1)/sum(cm(1,:));
    Subject_CVed(i).train_performance.FPR = cm(1,2)/sum(cm(1,:));
    kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(Y_train)^2;
    Subject_CVed(i).train_performance.kapp = (Train_acc - kappa_pe) / (1 - kappa_pe);
    
    for p = 1:length(sessions)
        X_test = test_data.(subjects{i}).(sessions{p}).(Subject_opt_NN_fold(i).opt_features_size).X;
        Y_test = test_data.(subjects{i}).(sessions{p}).(Subject_opt_NN_fold(i).opt_features_size).y;
        % Online sessions
        output_test = predict_nn(Theta1, Theta2, X_test);
        Test_acc = mean(double(output_test == Y_test));
        [c,cm,ind,per] = confusion(Y_test' - 1,output_test' - 1);
        %confusionchart(Y_test, out1);
        Subject_CVed(i).test_performance.(sessions{p}).acc = Test_acc;
        %TPR -> sensitivity, TNR -> specificity
        Subject_CVed(i).test_performance.(sessions{p}).sens = cm(2,2)/sum(cm(2,:));
        Subject_CVed(i).test_performance.(sessions{p}).spec = cm(1,1)/sum(cm(1,:));
        Subject_CVed(i).test_performance.(sessions{p}).FPR = cm(1,2)/sum(cm(1,:));
        kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(Y_test)^2;
        Subject_CVed(i).test_performance.(sessions{p}).kapp = (Test_acc - kappa_pe) / (1 - kappa_pe);
        confusionchart(Y_test' - 1,output_test' - 1);
        saveas(gcf, strcat('NN_','Online_',subjects{i}, '_', sessions{p}));
    end
end
save('Subject_online_results_NN.mat', 'Subject_CVed');
%% Data visualization for kappa on all sessions
figure;
load('Subject_online_results_NN.mat')
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
title('ErrP Classification Performance NN'); 
set(gca, 'FontSize', 12);
grid on; grid minor;
%% Data visualization for TPR vs FPR
load('Subject_online_results_NN.mat')
figure()
hold on
mark = {'o','^','s'};
color = {'r','g','b','c'};
for p = 1:4
    scatter(Subject_CVed(p).train_performance.FPR,Subject_CVed(p).train_performance.sens,mark{1},'MarkerFaceColor',color{p},'MarkerEdgeColor',color{p},'LineWidth',3)
    scatter(Subject_CVed(p).test_performance.S2.FPR,Subject_CVed(p).test_performance.S2.sens,mark{2},'MarkerFaceColor',color{p},'MarkerEdgeColor',color{p},'LineWidth',3)
    scatter(Subject_CVed(p).test_performance.S3.FPR,Subject_CVed(p).test_performance.S3.sens,mark{3},'MarkerFaceColor',color{p},'MarkerEdgeColor',color{p},'LineWidth',3)
end
grid on
grid minor
x = 0:.1:1;
plot(x,x,'--k')
lgd = legend('Subject1 Off','Subject1 S2','Subject1 S3', 'Subject2 Off','Subject2 S2','Subject2 S3', 'Subject4 Off','Subject4 S2','Subject4 S3', 'Subject5 Off','Subject5 S2','Subject5 S3')

lgd.Location = 'northeastoutside';
ylabel('True Positive Rate')
xlabel('False Positive Rate')
title('TPR vs FPR NN Classifier')
%% Jacksons fixed feature space plotting

load('CV_results_NN_Rerun_fold12.mat')
load('data/training_data_v2.mat')
features_num = {'size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7', 'size8', 'size9', 'size10', 'size11', 'size12'};
% Compute mean kappa
mean_kappa = zeros(length(subjects), length(features_num));
num = zeros(length(subjects), length(features_num));
figure;
hold on
colors = {'r', 'g', 'b', 'c'};
for p = 1:length(subjects)
    for z = 1:length(features_num)
        num(p,z) = length(training_data.(subjects{p}).(features_num{z}).stable_labels);
        kapp = [CV_Subject_NN_Rerun_fold(p).features_num(z).Fold.kapp];
        mean_kappa(p,z) = mean(kapp);
    end
    plot(num(p,:), mean_kappa(p,:), colors{p}, 'DisplayName', strcat(subjects{p}, " Kappa"),'LineWidth',1.5)
    [~, arg] = max(mean_kappa(p,:));
    xl = xline(num(p,arg), strcat(colors{p}, '--'), int2str(num(p,arg)), 'DisplayName', strcat(subjects{p}, " Opt Num"),'LineWidth',1.5);
    xl.LabelHorizontalAlignment = 'center';
end
legend()
grid on
grid minor
title('Feature Space Size Effect on Performance (NN Classifier)')
xlabel('Number of Features')
ylabel('Mean Kappa Across CV Folds')
% set(gca, 'FontSize', 10);
% saveas(gcf, strcat('results/', 'lda_', 'feature_size'));