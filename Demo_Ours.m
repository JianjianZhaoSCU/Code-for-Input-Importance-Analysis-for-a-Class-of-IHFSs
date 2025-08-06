%% Friedman Dataset
clc;clear all;close all;rng(9);
%% Data Generation and Process
load friedman
n_samples=size(friedman,1);
input = friedman(:,1:5);
output = friedman(:,6);
input=zscore(input);
output=zscore(output);
a = eps;
b = 1.0;
input_min = min(input);
input_max = max(input);
input = a + ((input - input_min) * (b - a)) ./ (input_max - input_min);
output_min = min(output);
output_max = max(output);
output = a + ((output - output_min) * (b - a)) ./ (output_max - output_min);
[coeff, score, latent, tsquared, explained] = pca(input);
explained_variance_ratio = cumsum(latent) / sum(latent);
threshold = 0.95;
best_k = find(explained_variance_ratio >= threshold, 1);
fprintf('---------------------------------------------------------------------------------\n');
fprintf('Optimal number of principal components: %d\n', best_k);
cv_partition = cvpartition(n_samples, 'HoldOut', 0.2);
train_idx = training(cv_partition);
test_idx = test(cv_partition);
train_data = score(train_idx, :);
train_output = output(train_idx);
test_data = score(test_idx, :);
test_output = output(test_idx);
[N1,~]=size(train_data);
[N2,~]=size(test_data);
ntrain=N1;
ntest=N2;
x_1=train_data(:,1);x_1t=test_data(:,1);
x_2=train_data(:,2);x_2t=test_data(:,2);
x_3=train_data(:,3);x_3t=test_data(:,3);
x_4=train_data(:,4);x_4t=test_data(:,4);
x_5=train_data(:,5);x_5t=test_data(:,5);
%% Input Importance Ranking
mutualInfos = zeros(1, 5);
wind_size=ceil(log2(n_samples)+1);
for i = 1:5
    mutualInfos(i) = calmi(score(:, i), output,wind_size);
end
fprintf('---------------------------------------------------------------------------------\n');
fprintf('NMI-1: %.4f\n', mutualInfos(1));
fprintf('NMI-2: %.4f\n', mutualInfos(2));
fprintf('NMI-3: %.4f\n', mutualInfos(3));
fprintf('NMI-4: %.4f\n', mutualInfos(4));
fprintf('NMI-5: %.4f\n', mutualInfos(5));
[sorted_MI, sorted_idx] = sort(mutualInfos, 'descend');
topk_idx = sorted_idx(1:5);
X_topk = score(:, topk_idx);
k = size(X_topk, 2);
MI_matrix = zeros(k, k);
for i = 1:k
    for j = i+1:k
        MI_matrix(i, j) = calmi(X_topk(:, i), X_topk(:, j), wind_size);
        MI_matrix(j, i) = MI_matrix(i, j);
    end
end
r_thres = 0.5;
redundant = false(1, k);
for i = 1:k
    others = setdiff(1:k, i);
    avg_mi = mean(MI_matrix(i, others));
    if avg_mi > r_thres
        redundant(i) = true;
    end
end
final_idx = topk_idx(~redundant);
fprintf('---------------------------------------------------------------------------------\n');
fprintf('Selected input ranking:\n');
fprintf('---------------------------------------------------------------------------------\n');
disp(final_idx);
%% Training and Testing
mm=20;
inputs = {'x_5', 'x_4', 'x_3', 'x_2', 'x_1'};
all_permutations = perms(inputs);
results = cell(size(all_permutations, 1), 2);
%-------------------------------------------------------%
%----------------------training-------------------------%
for perm_idx = 1:size(all_permutations, 1)
    current_order = all_permutations(perm_idx, :);
    % fprintf('Current input order: %s\n', strjoin(current_order, ', '));
    for i = 1:ntrain
        temp1 = eval(current_order{1});
        temp2 = eval(current_order{2});
        temp3 = eval(current_order{3});
        temp4 = eval(current_order{4});
        temp5 = eval(current_order{5});
        x11(i,1) = temp1(i,1);
        x12(i,1) = temp2(i,1);
        x13(i,1) = temp3(i,1);
        x14(i,1) = temp4(i,1);
        x15(i,1) = temp5(i,1);
        y(i,1) = train_output(i,1);
    end
    sx1=[x11,x12];
    %-------------------the first layer---------------------%
    tic;
    [zb11 rg11]=wmdeepzb(mm,sx1,y);
    % 'Level 1 training done '
    yy_1=wmdeepyy(mm,zb11,rg11,sx1);
    %-------------------the second layer--------------------%
    sx2(1:ntrain,1)=yy_1;
    sx2(:,2)=x13;
    [zb21 rg21]=wmdeepzb(mm,sx2,y);
    % 'Level 2 training done '
    yy_2=wmdeepyy(mm,zb21,rg21,sx2);
    %-------------------the third layer--------------------%
    sx3(1:ntrain,1)=yy_2;
    sx3(:,2)=x14;
    [zb31 rg31]=wmdeepzb(mm,sx3,y);
    % 'Level 3 training done '
    yy_3=wmdeepyy(mm,zb31,rg31,sx3);
    %-------------------the fourth layer--------------------%
    sx4(1:ntrain,1)=yy_3;
    sx4(:,2)=x15;
    [zb41 rg41]=wmdeepzb(mm,sx4,y);
    % 'Level 4 training done '
    % 'Training done'
    shfs_elapsedtime=toc;
    yy_train=wmdeepyy(mm,zb41,rg41,sx4);
    % fprintf('Traning Time：%.4f s\n', shfs_elapsedtime);
    %-------------------------------------------------------%
    %-----------------------testing-------------------------%
    %-------------------the first layer---------------------%
    for i = 1:ntest
        temp1t = eval([current_order{1}, 't']);
        temp2t = eval([current_order{2}, 't']);
        temp3t = eval([current_order{3}, 't']);
        temp4t = eval([current_order{4}, 't']);
        temp5t = eval([current_order{5}, 't']);
        x11t(i,1) = temp1t(i,1);
        x12t(i,1) = temp2t(i,1);
        x13t(i,1) = temp3t(i,1);
        x14t(i,1) = temp4t(i,1);
        x15t(i,1) = temp5t(i,1);
        yt(i,1) = test_output(i,1);
    end
    x1t=[x11t,x12t];
    yy_1t=wmdeepyy(mm,zb11,rg11,x1t);
    % 'Level 1 computing done'
    %-------------------the second layer--------------------%
    x2t(1:ntest,1)=yy_1t;
    x2t(:,2)=x13t;
    yy_2t=wmdeepyy(mm,zb21,rg21,x2t);
    % 'Level 2 computing done'
    %-------------------the third layer--------------------%
    x3t(1:ntest,1)=yy_2t;
    x3t(:,2)=x14t;
    yy_3t=wmdeepyy(mm,zb31,rg31,x3t);
    % 'Level 3 computing done'
    %-------------------the fourth layer--------------------%
    x4t(1:ntest,1)=yy_3t;
    x4t(:,2)=x15t;
    yy_test=wmdeepyy(mm,zb41,rg41,x4t);
    % 'Level 4 computing done'
    % 'computing done'

    zb11_matrix=reshape(zb11,mm,mm);
    zb21_matrix=reshape(zb21,mm,mm);
    zb31_matrix=reshape(zb31,mm,mm);
    zb41_matrix=reshape(zb41,mm,mm);
    aver_sen_1=(mm-1)^(5-1)*calculate_sum(zb11_matrix,mm)*calculate_sum(zb21_matrix,mm)...
        *calculate_sum(zb31_matrix,mm)*calculate_sum(zb41_matrix,mm)/2^(5-1);
    aver_sen_3=(mm-1)^(5-2)*calculate_sum(zb21_matrix,mm)...
        *calculate_sum(zb31_matrix,mm)*calculate_sum(zb41_matrix,mm)/2^(5-2);
    aver_sen_4=(mm-1)^(5-3)*calculate_sum(zb31_matrix,mm)*calculate_sum(zb41_matrix,mm)/2^(5-3);
    aver_sen_5=(mm-1)^(5-4)*calculate_sum(zb41_matrix,mm)/2^(5-4);
    disp(['AS-1: ', num2str(aver_sen_1)]);
    disp(['AS-2: ', num2str(aver_sen_3)]);
    disp(['AS-3: ', num2str(aver_sen_4)]);
    disp(['AS-4: ', num2str(aver_sen_5)]);
    fprintf('---------------------------------------------------------------------------------\n');
    % RMSE
    for i=1:ntrain
        e_train(i)=y(i)-yy_train(i);
    end;
    err_train=0;
    for i=1:ntrain
        err_train=err_train+e_train(i)^2;
    end;
    err_train=sqrt(err_train/ntrain);
    for i=1:ntest
        e_test(i)=yt(i)-yy_test(i);
    end;
    err_test=0;
    for i=1:ntest
        err_test=err_test+e_test(i)^2;
    end;
    err_test=sqrt(err_test/ntest);
    results{perm_idx, 1} = current_order;
    results{perm_idx, 2} = [err_train, err_test];
    fprintf('Order: %s, Train Error: %.4f, Test Error: %.4f\n', strjoin(current_order, ', '), err_train, err_test);
    fprintf('---------------------------------------------------------------------------------\n');
%     figure('Position', [100, 100, 800, 400]);
%     plot(yy_test)
%     hold on
%     plot(yt)
%     xlabel('Samples', 'FontSize', 12, 'FontWeight', 'bold');
%     ylabel('Output', 'FontSize', 12, 'FontWeight', 'bold');
%     title('FRI dataset', 'FontSize', 12, 'FontWeight', 'bold');
%     ax = gca;
%     ax.TickLength = [0.02, 0.02];
%     ax.XColor = 'k'; 
%     ax.YColor = 'k';
%     set(ax, 'FontSize', 12, 'FontWeight', 'bold');
end

rmse_values = results(:, 2); 
rmse_values = cell2mat(rmse_values(:, 1));
train_rmse = rmse_values(:,1);
test_rmse = rmse_values(:,2);
[train_rmse_max, train_rmse_max_idx] = max(train_rmse);
[train_rmse_min, train_rmse_min_idx] = min(train_rmse);
[test_rmse_max, test_rmse_max_idx] = max(test_rmse);
[test_rmse_min, test_rmse_min_idx] = min(test_rmse);

fprintf('train_rmse_max: %f, Index: %d\n', train_rmse_max, train_rmse_max_idx);
fprintf('train_rmse_min: %f, Index: %d\n', train_rmse_min, train_rmse_min_idx);
fprintf('test_rmse_max: %f, Index: %d\n', test_rmse_max, test_rmse_max_idx);
fprintf('test_rmse_min: %f, Index: %d\n', test_rmse_min, test_rmse_min_idx);
%% Auxiliary Functions
function mi = calmi(u1, u2, wind_size)
if size(u1, 2) > 1, u1 = u1'; end
if size(u2, 2) > 1, u2 = u2'; end
n = wind_size;
x = [u1, u2];
[xrow, xcol] = size(x);
bin = zeros(xrow, xcol);
pmf = zeros(n, 2);
for i = 1:2
    minx = min(x(:,i));
    maxx = max(x(:,i));
    binwidth = (maxx - minx) / n;
    edges = minx + binwidth * (0:n);
    histcEdges = [-Inf edges(2:end-1) Inf];
    [occur, bin(:,i)] = histc(x(:,i), histcEdges, 1);
    pmf(:,i) = occur(1:n) ./ xrow;
end
jointOccur = accumarray(bin, 1, [n, n]);
jointPmf = jointOccur ./ xrow;
Hx = -sum(pmf(:,1) .* log2(pmf(:,1) + eps)); % H(X)
Hy = -sum(pmf(:,2) .* log2(pmf(:,2) + eps)); % H(Y)
Hxy = -sum(jointPmf(:) .* log2(jointPmf(:) + eps)); % H(X, Y)
MI = Hx + Hy - Hxy;
mi = MI / sqrt(Hx * Hy);
end

function sum_elements = calculate_sum(matrix,mm)
    sub_matrix = matrix(mm/2:(mm/2)+1, mm/2:(mm/2)+1);
    sum_elements = sum(sub_matrix(:));

end
