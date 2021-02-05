cd 'C:/Users/Steve/Desktop/Adaptive Filtering Project'
load ("ZachDatalnAGBlnDlnDHlnD2HlnD2HW.txt")
size (ZachDatalnAGBlnDlnDHlnD2HlnD2HW)
corr (ZachDatalnAGBlnDlnDHlnD2HlnD2HW)

%%% Splitting The Data into Two
P = 0.80 ;
n = 482
idx = randperm(n)  ;
Train_Data =  ZachDatalnAGBlnDlnDHlnD2HlnD2HW(idx(1:round(P*n)),:)
Test_Data =  ZachDatalnAGBlnDlnDHlnD2HlnD2HW(idx(round(P*n)+1:end),:)

%%% Training Data
AGB_train = Train_Data (:, 1);
D_train = Train_Data (:, 2);
DH_train = Train_Data (:, 3);
D2H_train = Train_Data (:, 4);
D2HW_train = Train_Data (:, 5);

%%% Test Data
AGB_test = Test_Data (:, 1);
D_test = Test_Data (:, 2);
DH_test = Test_Data (:, 3);
D2H_test = Test_Data (:, 4);
D2HW_test = Test_Data (:, 2);

%%% Correlation Analysis of the Training Data
corr (AGB_train, D_train)
corr (AGB_train, DH_train)
corr (AGB_train, D2H_train)
corr (AGB_train, D2HW_train)
m_train = length (AGB_train)
l_train = ones (m_train, 1);

%%% Correlation Analysis of the Test Data
corr (AGB_test, D_test)
corr (AGB_test, DH_test)
corr (AGB_test, D2H_test)
corr (AGB_test, D2HW_test)
m_test = length (AGB_test)
l_test = ones (m_test, 1);

%%% X Variable . . .
X1_train = [l_train, D_train];
X2_train = [l_train, DH_train];
X3_train = [l_train, D2H_train];
X4_train = [l_train, D2HW_train];
X1_test = [l_test, D_test];
X2_test = [l_test, DH_test];
X3_test = [l_test, D2H_test];
X4_test = [l_test, D2HW_test];

%%% STOCHASTIC GRADIENT DESCENT %%%%%%%%
alpha = 0.001;
iterations = 200;

%% lnAGB vs lnD
%% Initializing variables . . .
theta10_vals = zeros (1, iterations);
theta11_vals = zeros (1, iterations);
theta1 = [0 ; 0];
J_vals1 = zeros (iterations);
J_history1 = zeros (iterations, 1);
theta_store1 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X1_train * theta1;
errors = h - AGB_train;
theta_change1 = (alpha * (X1_train' * errors)) / m_train;
theta1 = theta1 - theta_change1;
theta10_vals (i) = theta1 (1, :);
theta11_vals (i) = theta1 (2, :);
theta_store1 (:, i) = theta1;
J_history1(i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
J_vals1 (i, i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
end
%% End of Iterations.
%% End of lnAGB vs lnD

%% lnAGB vs lnDH
%% Initializing variables . . .
theta20_vals = zeros (1, iterations);
theta21_vals = zeros (1, iterations);
theta2 = [0 ; 0];
J_vals2 = zeros (iterations);
J_history2 = zeros (iterations, 1);
theta_store2 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X2_train * theta2;
errors = h - AGB_train;
theta_change2 = (alpha * (X2_train' * errors)) / m_train;
theta2 = theta2 - theta_change2;
theta20_vals (i) = theta2 (1, :);
theta21_vals (i) = theta2 (2, :);
theta_store2 (:, i) = theta2;
J_history2(i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
J_vals2 (i, i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
end
%% End of Iterations.
%% End of lnAGB vs lnDH

%% lnAGB vs lnD2H
%% Initializing variables . . .
theta30_vals = zeros (1, iterations);
theta31_vals = zeros (1, iterations);
theta3 = [0 ; 0];
J_vals3 = zeros (iterations);
J_history3 = zeros (iterations, 1);
theta_store3 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X3_train * theta3;
errors = h - AGB_train;
theta_change3 = (alpha * (X3_train' * errors)) / m_train;
theta3 = theta3 - theta_change3;
theta30_vals (i) = theta3 (1, :);
theta31_vals (i) = theta3 (2, :);
theta_store3 (:, i) = theta3;
J_history3(i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
J_vals3 (i, i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
end
%% End of Iterations.
%% End of lnAGB vs lnD2H

%% lnAGB vs lnD2HW
%% Initializing variables . . .
theta40_vals = zeros (1, iterations);
theta41_vals = zeros (1, iterations);
theta4 = [0 ; 0];
J_vals4 = zeros (iterations);
J_history4 = zeros (iterations, 1);
theta_store4 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X4_train * theta4;
errors = h - AGB_train;
theta_change4 = (alpha * (X4_train' * errors)) / m_train;
theta4 = theta4 - theta_change4;
theta40_vals (i) = theta4 (1, :);
theta41_vals (i) = theta4 (2, :);
theta_store4 (:, i) = theta4;
J_history4(i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
J_vals4 (i, i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
end
%% End of Iterations.
%% End of lnAGB vs lnD2HW


SGD_Loss1 = J_history1;
SGD_theta10 = theta10_vals;
SGD_theta11 = theta11_vals;
SGD_thetaStore1 = theta_store1;
SGD_theta1 = theta1;

SGD_Loss2 = J_history2;
SGD_theta20 = theta20_vals;
SGD_theta21 = theta21_vals;
SGD_thetaStore2 = theta_store2;
SGD_theta2 = theta2;

SGD_Loss3 = J_history3;
SGD_theta30 = theta30_vals;
SGD_theta31 = theta31_vals;
SGD_thetaStore3 = theta_store3;
SGD_theta3 = theta3;

SGD_Loss4 = J_history4;
SGD_theta40 = theta40_vals;
SGD_theta41 = theta41_vals;
SGD_thetaStore4 = theta_store4;
SGD_theta4 = theta4;

min (SGD_Loss1)
max (SGD_Loss1)
SGD_theta1

min (SGD_Loss2)
max (SGD_Loss2)
SGD_theta2

min (SGD_Loss3)
max (SGD_Loss3)
SGD_theta3

min (SGD_Loss4)
max (SGD_Loss4)
SGD_theta4

AGBSGD_cap1 = X1_test * SGD_theta1;
AGBSGD_cap2 = X2_test * SGD_theta2;
AGBSGD_cap3 = X3_test * SGD_theta3;
AGBSGD_cap4 = X4_test * SGD_theta4;

%%%%% Plotting Loss
epoch = [1:iterations];
plot (epoch, SGD_Loss1, '-')
hold on;
plot (epoch, SGD_Loss2, '-')
plot (epoch, SGD_Loss3, '-')
plot (epoch, SGD_Loss4, '-')
xlabel('Iteration'); ylabel('Loss');

%%% RUN STUDENT'S T-TEST DATA %%%%%
pkg load statistics
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap1')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap2')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap3')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap4')


%%%% Root Mean Square Error Test
RMSE_SGD1 = sqrt (sum((AGBSGD_cap1 - AGB_test).^2)/m_test)
RMSE_SGD2 = sqrt (sum((AGBSGD_cap2 - AGB_test).^2)/m_test)
RMSE_SGD3 = sqrt (sum((AGBSGD_cap3 - AGB_test).^2)/m_test)
RMSE_SGD4 = sqrt (sum((AGBSGD_cap4 - AGB_test).^2)/m_test)
