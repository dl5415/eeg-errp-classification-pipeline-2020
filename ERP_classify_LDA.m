%% ErrP Project LDA Classification

close all
clear
clc
subjects = {'Subject1', 'Subject2', 'Subject4', 'Subject5'};
sessions = {'S2', 'S3'};
features_num = {'size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7', 'size8', 'size9', 'size10'};
img_type = 'png';
% CV parameters
KFold = 4; % Run wise
fold_size = 125; % Each run has 125 trials

%% CV experiment (on all subjects)
% MATLAB builtin KFold does not respect temporality, hence need to parse
% training data manually. Also only metric available is classification
% accuracy (CV loss)

load('data/training_data_v2.mat')
for p = 1:length(subjects)
    for z = 1:length(features_num)
        
        X_train = training_data.(subjects{p}).(features_num{z}).X;
        Y_train = training_data.(subjects{p}).(features_num{z}).y;
        
        % Deal with empty features
        if (isempty(X_train))
            X_train = training_data.(subjects{p}).(features_num{z+1}).X;
        end
        % label 1 -> ErrP trial
        Y_train = Y_train - 1;
        
        for k = 1:KFold
            % Validation set
            if (isequal(subjects{p}, 'Subject2') && (k == 4)) || (isequal(subjects{p}, 'Subject5') && (k == 4)) 
                end_indx = k * fold_size - 1 ;
            else
                end_indx = k * fold_size ;
            end
            test_data = X_train((k-1) * fold_size + 1 : end_indx, :);
            test_data_label = Y_train((k-1) * fold_size + 1 : end_indx, :);
            
            % Omit validation set and use the rest as train data
            train_data = X_train;
            train_data((k-1) * fold_size + 1 : end_indx, :) = [];
            train_data_label = Y_train;
            train_data_label((k-1) * fold_size + 1 : end_indx, :) = [];
            
            lda_model = fitcdiscr(train_data, train_data_label);
            [pred_labels, pred_scores, ~] = predict(lda_model, test_data);
            
            % Performance on this fold
            [c, cm, ind, per] = confusion(test_data_label', pred_labels');
            acc = 1-c;
            test_performance.acc = acc;
            
            % TPR -> sensitivity, TNR -> specificity
            test_performance.sens = cm(2,2)/sum(cm(2,:));
            test_performance.spec = cm(1,1)/sum(cm(1,:));
            test_performance.FPR = cm(1,2)/sum(cm(1,:));
            
            % Kappa
            kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(test_data_label)^2;
            test_performance.kapp = (acc - kappa_pe) / (1 - kappa_pe);
            
            % AUC ROC
            [~, ~, ~, auc] = perfcurve(test_data_label, pred_scores(:,2), 1);
            test_performance.auc = auc;
            
            CV_Subject(p).features_num(z).Fold(k) = test_performance;  %#ok<*SAGROW>
        end
    end
end

save('results/cv_results_lda.mat', 'CV_Subject');

%% Select optimal parameters

load('results/cv_results_lda.mat')
% Compute mean kappa
mean_kappa = zeros(length(subjects), length(features_num));
for p = 1:length(subjects)
    for z = 1:length(features_num)  
        kapp = [CV_Subject(p).features_num(z).Fold.kapp];
        mean_kappa(p,z) = mean(kapp);
    end
end

% find optimal parameter for each subject
kappa_max = max(mean_kappa, [], 2);
for p = 1:length(subjects)
    idx = find(mean_kappa == kappa_max(p));
    s = size(mean_kappa);
    [I,J] = ind2sub(s,idx);
    Subject_opt(p).opt_features_size = features_num{J(1)};
end

save('results/opt_size_lda.mat', 'Subject_opt');

%% Train again with the optimal parameters and online classification

load('results/opt_size_lda.mat')
load('data/training_data_v2.mat')
load('data/test_data_v2.mat')

for i = 1:length(subjects)
    % Train on offline with optimal parameters
    X_train = training_data.(subjects{i}).(Subject_opt(i).opt_features_size).X;
    Y_train = training_data.(subjects{i}).(Subject_opt(i).opt_features_size).y;
    % label 1 -> ErrP trial
    Y_train = Y_train - 1;
    % Training
    lda_model = fitcdiscr(X_train, Y_train);
    % Training score
    [pred_labels, pred_scores, ~] = predict(lda_model, X_train);
    [c, cm, ind, per] = confusion(Y_train', pred_labels');
    acc = 1-c;
    Subject_CVed(i).train_performance.acc = acc;
    %TPR -> sensitivity, TNR -> specificity
    Subject_CVed(i).train_performance.sens = cm(2,2)/sum(cm(2,:)); 
    Subject_CVed(i).train_performance.spec = cm(1,1)/sum(cm(1,:));
    Subject_CVed(i).train_performance.FPR = cm(1,2)/sum(cm(1,:));
    kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(Y_train)^2;
    Subject_CVed(i).train_performance.kapp = (acc - kappa_pe) / (1 - kappa_pe);
    % AUC ROC
    [~, ~, ~, auc] = perfcurve(Y_train, pred_scores(:,2), 1);
    Subject_CVed(i).train_performance.auc = auc;
    
    % Testing - Subjects remain constant
    for p = 1:length(sessions)
        X_test = test_data.(subjects{i}).(sessions{p}).(Subject_opt(i).opt_features_size).X;
        Y_test = test_data.(subjects{i}).(sessions{p}).(Subject_opt(i).opt_features_size).y;
        Y_test = Y_test - 1;
        % Online sessions
        [pred_labels, pred_scores, ~] = predict(lda_model, X_test);
        [c,cm,ind,per] = confusion(Y_test', pred_labels');
        acc = 1-c;
        Subject_CVed(i).test_performance.(sessions{p}).acc = acc;
        % TPR -> sensitivity, TNR -> specificity
        Subject_CVed(i).test_performance.(sessions{p}).sens = cm(2,2)/sum(cm(2,:)); 
        Subject_CVed(i).test_performance.(sessions{p}).spec = cm(1,1)/sum(cm(1,:));
        Subject_CVed(i).test_performance.(sessions{p}).FPR = cm(1,2)/sum(cm(1,:));
        kappa_pe = ((cm(1,1)+cm(1,2)) * (cm(1,1)+cm(2,1)) + (cm(1,2)+cm(2,2)) * (cm(2,1)+cm(2,2))) / length(Y_test)^2;
        Subject_CVed(i).test_performance.(sessions{p}).kapp = (acc - kappa_pe) / (1 - kappa_pe);
        % AUC ROC
        [~, ~, ~, auc] = perfcurve(Y_test, pred_scores(:,2), 1);
        Subject_CVed(i).test_performance.(sessions{p}).auc = auc;
    end
end

save('results/online_results_lda.mat', 'Subject_CVed');

%% Plot 1, bar plot Kappa across subjects and sessions

figure;
hold on
load('results/online_results_lda.mat')
load('Subject_online_chance_perf.mat')
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
yline(0,'k','LineWidth',3, 'DisplayName', 'Chance Kappa' );
set(b, {'DisplayName'}, {'Offline', 'S2', 'S3'}');
legend()
ylabel('Kappa', 'fontweight', 'bold');
ylim([0 0.8])
title('ErrP Classification Performance (LDA Classifier)');
grid on; grid minor;
set(gca, 'FontSize', 12);
set(gcf, 'Position', [100 100 600 500])
% set(fig_kap, 'Position', [100 100 1000 450]);
saveas(gcf, strcat('figures/', 'lda_', 'kappa_bar'));
exportgraphics(gcf, strcat('figures/lda_kappa_bar.', img_type))

%% Plot 2, scatterplot of TPR vs FPR

load('results/online_results_lda.mat')
figure;
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
lgd = legend('Subject1 Off','Subject1 S2','Subject1 S3', 'Subject2 Off','Subject2 S2','Subject2 S3', 'Subject4 Off','Subject4 S2','Subject4 S3', 'Subject5 Off','Subject5 S2','Subject5 S3');
lgd.Location = 'northeastoutside';
ylabel('True Positive Rate')
xlabel('False Positive Rate')
title('TPR vs FPR (LDA Classifier)')
set(gca, 'FontSize', 10);
set(gcf, 'Position', [100 100 600 400]);
saveas(gcf, strcat('figures/', 'lda_', 'tpr_fpr'));
exportgraphics(gcf, strcat('figures/lda_tpr_fpr.', img_type))

%% Plot 3, Kappa vs. Number of Features

load('results/cv_results_lda.mat')
load('data/training_data_v2.mat')
% Compute mean kappa
mean_kappa = zeros(length(subjects), length(features_num));
num = zeros(length(subjects), length(features_num));
figure;
hold on
colors = {'r', 'g', 'b', 'c'};
for p = 1:length(subjects)
    for z = 1:length(features_num)
        num(p,z) = length(training_data.(subjects{p}).(features_num{z}).stable_labels);
        kapp = [CV_Subject(p).features_num(z).Fold.kapp];
        mean_kappa(p,z) = mean(kapp);
    end
    plot(num(p,:), mean_kappa(p,:), colors{p}, 'DisplayName', strcat(subjects{p}, " Kappa"))
    [~, arg] = max(mean_kappa(p,:));
    xl = xline(num(p,arg), strcat(colors{p}, '--'), int2str(num(p,arg)), 'DisplayName', strcat(subjects{p}, " Opt Num"));
    xl.LabelHorizontalAlignment = 'center';
end
legend()
grid on
grid minor
title('Feature Space Size Effect on Performance (LDA Classifier)')
xlabel('Number of Features')
ylabel('Mean Kappa Across CV Folds')
set(gca, 'FontSize', 10);
set(gcf, 'Position', [100 100 800 400]);
saveas(gcf, strcat('figures/', 'lda_', 'feature_size'));
exportgraphics(gcf, strcat('figures/lda_feature_size.', img_type))

%% Grand Average and Topo

electrodes = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
load('data/processed_data.mat')
subject = 1;

figure
data = processed_data.(subjects{subject}).Offline;
channel = 4;
t_ind = plot_grand_averages(data.t, data.trials, data.trig_vals, channel, strcat(subjects{subject}, {' Offline Grand Average (Mean +/- SE) for '}, electrodes{channel}));
set(gcf, 'Position', [100 100 600 400]);
exportgraphics(gcf, strcat('figures\ga_', subjects{subject}, '.', img_type));

load('chanlocs16.mat')

figure
data = processed_data.(subjects{subject}).Offline;
channel = 4;
plot_topos(data.trials, data.trig_vals, t_ind, chanlocs16, strcat(int2str(round(1000*data.t(t_ind))), {' ms'}));
set(gcf, 'Position', [100 100 300 500]);
% sgtitle(strcat(subjects{subject}, {' Offline Topological Plot'}));
exportgraphics(gcf, strcat('figures\topo_', subjects{subject}, '.', img_type));

%% Functions

function plot_topos(trials, trigger_values, t_ind, chanlocs16, name)
    correct = trigger_values == 0;
    error = trigger_values == 1;
    
    correct_data = mean(trials(correct,:,t_ind));
    error_data = mean(trials(error,:,t_ind));
    
    disp(size(trials))
    disp(size(error_data))
    disp(size(correct_data))
    disp(error_data)
    
    lower = min([min(correct_data) min(error_data)]);
    upper = max([max(correct_data) max(error_data)]);
    diff = upper - lower;
    factor = .5;
    limits = [(lower - diff*factor) (upper + diff*factor)];
    
    subplot(2, 1, 1);
    topoplot(correct_data, chanlocs16, 'maplimits', limits);
    title(strcat({'Correct Trials Average at '}, name));
    cbar();
    
    subplot(2, 1, 2);
    topoplot(error_data, chanlocs16, 'maplimits', limits);
    title(strcat({'Error Trials Average at '}, name));
    cbar();
end

function t_ind = plot_grand_averages(t, trials, trigger_values, channel, name)
    trials = squeeze(trials(:,channel,:));
    
    correct_trials = trials(trigger_values == 0,:);
    error_trials = trials(trigger_values == 1,:);
    
    correct_avg = mean(correct_trials, 1);
    error_avg = mean(error_trials, 1);
    
    [~, t_ind] = min(error_avg);
    t_val = t(t_ind);
    
    correct_se = std(correct_trials, 0, 1)/sqrt(length(correct_trials));
    error_se = std(error_trials, 0, 1)/sqrt(length(error_trials));
    
    hold on
    plot(t, correct_avg, 'b', t, error_avg, 'r');
    plot(t, correct_avg - correct_se, '--b', t, correct_avg + correct_se, '--b', t, error_avg - error_se, '--r', t, error_avg + error_se, '--r');
    xline(t_val, '-.');
    title(name);
    ylabel('Signal Magnitude [\muV]');
    xlabel('Time [s]');
    xlim([t(1) t(end)]);
    legend({'Correct','Error'},'Location','southeast');
end
