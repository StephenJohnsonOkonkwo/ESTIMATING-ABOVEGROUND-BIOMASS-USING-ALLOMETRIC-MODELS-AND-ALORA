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

%%% Full Data
AGB = ZachDatalnAGBlnDlnDHlnD2HlnD2HW (:, 1);
D = ZachDatalnAGBlnDlnDHlnD2HlnD2HW (:, 2);
DH = ZachDatalnAGBlnDlnDHlnD2HlnD2HW (:, 3);
D2H = ZachDatalnAGBlnDlnDHlnD2HlnD2HW (:, 4);
D2HW = ZachDatalnAGBlnDlnDHlnD2HlnD2HW (:, 5);

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
%% Initializing variables . . .
alpha = 0.001;
iterations = 200;
%% Initializing decays . . . .
beta = 0.9;
beta_1 = 0.9;
beta_2 = 0.999;

%%%%%% STOCHASTIC GRADIENT DESCENT RMS PROPAGATION %%%%%%%

%% lnAGB vs lnD
%% Initializing variables . . .
theta10_vals = zeros (1, iterations);
theta11_vals = zeros (1, iterations);
theta1 = [0 ; 0];
J_vals1 = zeros (iterations);
J_history1 = zeros (iterations, 1);
theta_store1 = zeros (2, iterations);

%% Initializing moments . . . .
momentum1 = [0; 0]; % Momentum
momentum_store1 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X1_train * theta1;
errors = h - AGB_train;
momentum1 = (beta * momentum1) + (1 - beta) * (((X1_train' * errors) / m_train).^2);
theta_change1 = pinv (sqrt (abs (momentum1)) + 10e-8) * (alpha * ((X1_train' * errors) / m_train));
theta1 = theta1 - theta_change1;
momentum_store1 (:, i) = momentum1;
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

%% Initializing moments . . . .
momentum2 = [0; 0]; % Momentum
momentum_store2 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X2_train * theta2;
errors = h - AGB_train;
momentum2 = (beta * momentum2) + (1 - beta) * (((X2_train' * errors) / m_train).^2);
theta_change2 = pinv (sqrt (abs (momentum2)) + 10e-8) * (alpha * ((X2_train' * errors) / m_train));
theta2 = theta2 - theta_change2;
momentum_store2 (:, i) = momentum2;
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

%% Initializing moments . . . .
momentum3 = [0; 0]; % Momentum
momentum_store3 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X3_train * theta3;
errors = h - AGB_train;
momentum3 = (beta * momentum3) + (1 - beta) * (((X3_train' * errors) / m_train).^2);
theta_change3 = pinv (sqrt (abs (momentum3)) + 10e-8) * (alpha * ((X3_train' * errors) / m_train));
theta3 = theta3 - theta_change3;
momentum_store3 (:, i) = momentum3;
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

%% Initializing moments . . . .
momentum4 = [0; 0]; % Momentum
momentum_store4 = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X4_train * theta4;
errors = h - AGB_train;
momentum4 = (beta * momentum4) + (1 - beta) * (((X4_train' * errors) / m_train).^2);
theta_change4 = pinv (sqrt (abs (momentum4)) + 10e-8) * (alpha * ((X4_train' * errors) / m_train));
theta4 = theta4 - theta_change4;
momentum_store4 (:, i) = momentum4;
theta40_vals (i) = theta4 (1, :);
theta41_vals (i) = theta4 (2, :);
theta_store4 (:, i) = theta4;
J_history4(i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
J_vals4 (i, i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
end
%% End of Iterations.
%% End of lnAGB vs lnD2HW

RMSProp_Loss1 = J_history1;
RMSProp_theta10 = theta10_vals;
RMSProp_theta11 = theta11_vals;
RMSProp_thetaStore1 = theta_store1;
RMSProp_theta1 = theta1;

RMSProp_Loss2 = J_history2;
RMSProp_theta20 = theta20_vals;
RMSProp_theta21 = theta21_vals;
RMSProp_thetaStore2 = theta_store2;
RMSProp_theta2 = theta2;

RMSProp_Loss3 = J_history3;
RMSProp_theta30 = theta30_vals;
RMSProp_theta31 = theta31_vals;
RMSProp_thetaStore3 = theta_store3;
RMSProp_theta3 = theta3;

RMSProp_Loss4 = J_history4;
RMSProp_theta40 = theta40_vals;
RMSProp_theta41 = theta41_vals;
RMSProp_thetaStore4 = theta_store4;
RMSProp_theta4 = theta4;

min (RMSProp_Loss1)
max (RMSProp_Loss1)
RMSProp_theta1

min (RMSProp_Loss2)
max (RMSProp_Loss2)
RMSProp_theta2

min (RMSProp_Loss3)
max (RMSProp_Loss3)
RMSProp_theta3

min (RMSProp_Loss4)
max (RMSProp_Loss4)
RMSProp_theta4

AGBSGD_cap1RMSProp = X1_test * RMSProp_theta1;
AGBSGD_cap2RMSProp = X2_test * RMSProp_theta2;
AGBSGD_cap3RMSProp = X3_test * RMSProp_theta3;
AGBSGD_cap4RMSProp = X4_test * RMSProp_theta4;

%%%%% Plotting RMSProp Loss
epoch = [1:iterations];
figure ()
plot (epoch, RMSProp_Loss1, '-', 'Color', 'blue')
hold on;
plot (epoch, RMSProp_Loss2, '-', 'Color', 'green')
plot (epoch, RMSProp_Loss3, '-', 'Color', 'red')
plot (epoch, RMSProp_Loss4, '-', 'Color', 'yellow')
title ('')
legend ('{\fontsize{20}ln(AGB) vs ln(D)}','{\fontsize{20}ln(AGB) vs ln(DH)}',...
        '{\fontsize{20}ln(AGB) vs ln((D^2)H)}','{\fontsize{20}ln(AGB) vs ln((D^2)HW)}',...
        'location','northeastoutside')
%legend boxoff
xlabel('{\fontsize{25}Iterations}'); ylabel('{\fontsize{25}Loss}');
grid on

%%% END OF STOCHASTIC GRADIENT DESCENT RMS PROPAGATION %%%%%%

%%%%%% STOCHASTIC GRADIENT DESCENT ADAM %%%%%%%

%% lnAGB vs lnD
%% Initializing variables . . .
theta10_vals = zeros (1, iterations);
theta11_vals = zeros (1, iterations);
theta1 = [0 ; 0];
J_vals1 = zeros (iterations);
J_history1 = zeros (iterations, 1);
theta_store1 = zeros (2, iterations);

%% Initializing moments . . . .
moment_11 = [0; 0]; % First moment
moment_12 = [0; 0]; % Second moment
m11_store = zeros (2, iterations);
m12_store = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X1_train * theta1;
errors = h - AGB_train;
moment_11 = (beta_1 * moment_11) + (1 - beta_1) * ((X1_train' * errors) / m_train);
moment_12 = (beta_2 * moment_12) + (1 - beta_2) * ((X1_train' * errors) / m_train);
theta_change1 = pinv (sqrt (abs (moment_12)) + 10e-8) * (alpha * moment_11);
theta1 = theta1 - theta_change1;
m11_store (:, i) = moment_11;
m12_store (:, i) = moment_12;
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

%% Initializing moments . . . .
moment_21 = [0; 0]; % First moment
moment_22 = [0; 0]; % Second moment
m21_store = zeros (2, iterations);
m22_store = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X2_train * theta2;
errors = h - AGB_train;
moment_21 = (beta_1 * moment_21) + (1 - beta_1) * ((X2_train' * errors) / m_train);
moment_22 = (beta_2 * moment_22) + (1 - beta_2) * ((X2_train' * errors) / m_train);
theta_change2 = pinv (sqrt (abs (moment_22)) + 10e-8) * (alpha * moment_21);
theta2 = theta2 - theta_change2;
m21_store (:, i) = moment_21;
m22_store (:, i) = moment_22;
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

%% Initializing moments . . . .
moment_31 = [0; 0]; % First moment
moment_32 = [0; 0]; % Second moment
m31_store = zeros (2, iterations);
m32_store = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X3_train * theta3;
errors = h - AGB_train;
moment_31 = (beta_1 * moment_31) + (1 - beta_1) * ((X3_train' * errors) / m_train);
moment_32 = (beta_2 * moment_32) + (1 - beta_2) * ((X3_train' * errors) / m_train);
theta_change3 = pinv (sqrt (abs (moment_32)) + 10e-8) * (alpha * moment_31);
theta3 = theta3 - theta_change3;
m31_store (:, i) = moment_31;
m32_store (:, i) = moment_32;
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

%% Initializing moments . . . .
moment_41 = [0; 0]; % First moment
moment_42 = [0; 0]; % Second moment
m41_store = zeros (2, iterations);
m42_store = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X4_train * theta4;
errors = h - AGB_train;
moment_41 = (beta_1 * moment_41) + (1 - beta_1) * ((X4_train' * errors) / m_train);
moment_42 = (beta_2 * moment_42) + (1 - beta_2) * ((X4_train' * errors) / m_train);
theta_change4 = pinv (sqrt (abs (moment_42)) + 10e-8) * (alpha * moment_41);
theta4 = theta4 - theta_change4;
m41_store (:, i) = moment_41;
m42_store (:, i) = moment_42;
theta40_vals (i) = theta4 (1, :);
theta41_vals (i) = theta4 (2, :);
theta_store4 (:, i) = theta4;
J_history4(i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
J_vals4 (i, i) = (sum ((h - AGB_train).^2))/ (2 * m_train);
end
%% End of Iterations.
%% End of lnAGB vs lnD2HW

Adam_Loss1 = J_history1;
Adam_theta10 = theta10_vals;
Adam_theta11 = theta11_vals;
Adam_thetaStore1 = theta_store1;
Adam_theta1 = theta1;

Adam_Loss2 = J_history2;
Adam_theta20 = theta20_vals;
Adam_theta21 = theta21_vals;
Adam_thetaStore2 = theta_store2;
Adam_theta2 = theta2;

Adam_Loss3 = J_history3;
Adam_theta30 = theta30_vals;
Adam_theta31 = theta31_vals;
Adam_thetaStore3 = theta_store3;
Adam_theta3 = theta3;

Adam_Loss4 = J_history4;
Adam_theta40 = theta40_vals;
Adam_theta41 = theta41_vals;
Adam_thetaStore4 = theta_store4;
Adam_theta4 = theta4;

min (Adam_Loss1)
max (Adam_Loss1)
Adam_theta1

min (Adam_Loss2)
max (Adam_Loss2)
Adam_theta2

min (Adam_Loss3)
max (Adam_Loss3)
Adam_theta3

min (Adam_Loss4)
max (Adam_Loss4)
Adam_theta4

AGBSGD_cap1Adam = X1_test * Adam_theta1;
AGBSGD_cap2Adam = X2_test * Adam_theta2;
AGBSGD_cap3Adam = X3_test * Adam_theta3;
AGBSGD_cap4Adam = X4_test * Adam_theta4;

%%%%% Plotting Adam Loss
epoch = [1:iterations];
figure ()
plot (epoch, Adam_Loss1, '-', 'Color', 'blue')
hold on;
plot (epoch, Adam_Loss2, '-', 'Color', 'green')
plot (epoch, Adam_Loss3, '-', 'Color', 'red')
plot (epoch, Adam_Loss4, '-', 'Color', 'yellow')
title ('')
legend ('{\fontsize{20}ln(AGB) vs ln(D)}','{\fontsize{20}ln(AGB) vs ln(DH)}',...
        '{\fontsize{20}ln(AGB) vs ln((D^2)H)}','{\fontsize{20}ln(AGB) vs ln((D^2)HW)}',...
        'location','northeastoutside')
legend boxoff
xlabel('{\fontsize{25}Iterations}'); ylabel('{\fontsize{25}Loss}');
grid on
%%% END OF STOCHASTIC GRADIENT DESCENT ADAM %%%%%%

%%%%%%%%% ACCURACY TEST FOR RMSPROP %%%%%%%%%%%%%%%%
%%% RUN STUDENT'S T-TEST DATA %%%%%
pkg load statistics
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap1RMSProp')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap2RMSProp')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap3RMSProp')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap4RMSProp')

%%%%%% Error Test
E_RMSProp1 = AGBSGD_cap1RMSProp - AGB_test;
E_RMSProp2 = AGBSGD_cap2RMSProp - AGB_test;
E_RMSProp3 = AGBSGD_cap3RMSProp - AGB_test;
E_RMSProp4 = AGBSGD_cap4RMSProp - AGB_test;

%%%%%% mean Absolute Error Test
MAE_RMSProp1 = sum(abs(E_RMSProp1)) /m_test
MAE_RMSProp2 = sum(abs(E_RMSProp2)) /m_test
MAE_RMSProp3 = sum(abs(E_RMSProp3)) /m_test
MAE_RMSProp4 = sum(abs(E_RMSProp4)) /m_test

%%%%%% Mean Square Error Test
MSE_RMSProp1 = sum((AGBSGD_cap1RMSProp - AGB_test).^2)/m_test
MSE_RMSProp2 = sum((AGBSGD_cap2RMSProp - AGB_test).^2)/m_test
MSE_RMSProp3 = sum((AGBSGD_cap3RMSProp - AGB_test).^2)/m_test
MSE_RMSProp4 = sum((AGBSGD_cap4RMSProp - AGB_test).^2)/m_test

%%%% Root Mean Square Error Test
RMSE_RMSProp1 = sqrt (sum((AGBSGD_cap1RMSProp - AGB_test).^2)/m_test)
RMSE_RMSProp2 = sqrt (sum((AGBSGD_cap2RMSProp - AGB_test).^2)/m_test)
RMSE_RMSProp3 = sqrt (sum((AGBSGD_cap3RMSProp - AGB_test).^2)/m_test)
RMSE_RMSProp4 = sqrt (sum((AGBSGD_cap4RMSProp - AGB_test).^2)/m_test)

%%%%% Percentage Error Test
PE_RMSProp1 = 100 * ((AGBSGD_cap1RMSProp - AGB_test) ./ AGB_test);
PE_RMSProp2 = 100 * ((AGBSGD_cap2RMSProp - AGB_test) ./ AGB_test);
PE_RMSProp3 = 100 * ((AGBSGD_cap3RMSProp - AGB_test) ./ AGB_test);
PE_RMSProp4 = 100 * ((AGBSGD_cap4RMSProp - AGB_test) ./ AGB_test);

%%%%%% Mean Absolute Percentage Error test
MAPE_RMSProp1 = sum(abs(PE_RMSProp1)) /m_test
MAPE_RMSProp2 = sum(abs(PE_RMSProp2)) /m_test
MAPE_RMSProp3 = sum(abs(PE_RMSProp3)) /m_test
MAPE_RMSProp4 = sum(abs(PE_RMSProp4)) /m_test

%%%%%%%% ACCURACY TEST FOR ADAM %%%%%%%%%%%%%%%%%%%%%%
%%% RUN STUDENT'S T-TEST DATA %%%%%
pkg load statistics
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap1Adam')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap2Adam')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap3Adam')
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap4Adam')

%%%%%% Error Test
E_Adam1 = AGBSGD_cap1Adam - AGB_test;
E_Adam2 = AGBSGD_cap2Adam - AGB_test;
E_Adam3 = AGBSGD_cap3Adam - AGB_test;
E_Adam4 = AGBSGD_cap4Adam - AGB_test;

%%%%%% mean Absolute Error Test
MAE_Adam1 = sum(abs(E_Adam1)) /m_test
MAE_Adam2 = sum(abs(E_Adam2)) /m_test
MAE_Adam3 = sum(abs(E_Adam3)) /m_test
MAE_Adam4 = sum(abs(E_Adam4)) /m_test

%%%%%% Mean Square Error Test
MSE_Adam1 = sum((AGBSGD_cap1Adam - AGB_test).^2)/m_test
MSE_Adam2 = sum((AGBSGD_cap2Adam - AGB_test).^2)/m_test
MSE_Adam3 = sum((AGBSGD_cap3Adam - AGB_test).^2)/m_test
MSE_Adam4 = sum((AGBSGD_cap4Adam - AGB_test).^2)/m_test

%%%% Root Mean Square Error Test
RMSE_Adam1 = sqrt (sum((AGBSGD_cap1Adam - AGB_test).^2)/m_test)
RMSE_Adam2 = sqrt (sum((AGBSGD_cap2Adam - AGB_test).^2)/m_test)
RMSE_Adam3 = sqrt (sum((AGBSGD_cap3Adam - AGB_test).^2)/m_test)
RMSE_Adam4 = sqrt (sum((AGBSGD_cap4Adam - AGB_test).^2)/m_test)

%%%%% Percentage Error Test
PE_Adam1 = 100 * ((AGBSGD_cap1Adam - AGB_test) ./ AGB_test);
PE_Adam2 = 100 * ((AGBSGD_cap2Adam - AGB_test) ./ AGB_test);
PE_Adam3 = 100 * ((AGBSGD_cap3Adam - AGB_test) ./ AGB_test);
PE_Adam4 = 100 * ((AGBSGD_cap4Adam - AGB_test) ./ AGB_test);

%%%%%% Mean Absolute Percentage Error test
MAPE_Adam1 = sum(abs(PE_Adam1)) /m_test
MAPE_Adam2 = sum(abs(PE_Adam2)) /m_test
MAPE_Adam3 = sum(abs(PE_Adam3)) /m_test
MAPE_Adam4 = sum(abs(PE_Adam4)) /m_test


m = length (AGB)
F = [ones(m, 1), D(:)];
%[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
%AGBFit = F * p;
AGBFit1 = F * Adam_theta1;
AGBFit1Var = var (AGBFit1)
AGBVar = var(AGB)
%AGBFit3 = F * Adam_theta1;
figure ()
plot(D, AGB, '+b', D, AGBFit1, '-g',...
     D, AGBFit1 + 1.96 * sqrt (AGBVar), '--r',...
     D, AGBFit1 + 1.96 * sqrt (AGBFit1Var), '--k',...
     D, AGBFit1 - 1.96 * sqrt (AGBVar), '--r',...
     D, AGBFit1 - 1.96 * sqrt (AGBFit1Var), '--k')
title (' ')
legend ('{\fontsize{20}data}','{\fontsize{20} ln({AGB_{fit}}) = 0.90185 + 0.90185(ln(D))}','{\fontsize{20}+/-95% ln(AGB) values}',...
        '{\fontsize{20}+/- 95% ln({AGB_{fit}}) values}','location','northwest')
legend boxoff
xlabel('{\fontsize{25}ln(D)}'); ylabel('{\fontsize{25}ln(AGB)}');
grid on     


m = length (AGB)
F = [ones(m, 1), DH(:)];
%[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
%AGBFit = F * p;
AGBFit2 = F * Adam_theta2;
AGBFit2Var = var (AGBFit2)
AGBVar = var(AGB)
%AGBFit3 = F * Adam_theta1;
figure ()
plot(DH, AGB, '+b', DH, AGBFit2, '-g',...
     DH, AGBFit2 + 1.96 * sqrt (AGBVar), '--r',...
     DH, AGBFit2 + 1.96 * sqrt (AGBFit2Var), '--k',...
     DH, AGBFit2 - 1.96 * sqrt (AGBVar), '--r',...
     DH, AGBFit2 - 1.96 * sqrt (AGBFit2Var), '--k')
title (' ')
legend ('{\fontsize{20}data}','{\fontsize{20} ln({AGB_{fit}}) = 0.55252 + 0.0.55252(ln(DH))}','{\fontsize{20}+/-95% ln(AGB) values}',...
        '{\fontsize{20}+/- 95% ln({AGB_{fit}}) values}','location','south')
legend boxoff
xlabel('{\fontsize{25}ln(DH)}'); ylabel('{\fontsize{25}ln(AGB)}');
grid on     


m = length (AGB)
F = [ones(m, 1), D2H(:)];
%[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
%AGBFit = F * p;
AGBFit3 = F * Adam_theta3;
AGBFit3Var = var (AGBFit3)
AGBVar = var(AGB)
%AGBFit3 = F * Adam_theta1;
figure ()
plot(D2H, AGB, '+b', D2H, AGBFit3, '-g',...
     D2H, AGBFit3 + 1.96 * sqrt (AGBVar), '--r',...
     D2H, AGBFit3 + 1.96 * sqrt (AGBFit3Var), '--k',...
     D2H, AGBFit3 - 1.96 * sqrt (AGBVar), '--r',...
     D2H, AGBFit3 - 1.96 * sqrt (AGBFit3Var), '--k')
title (' ')
legend ('{\fontsize{20}data}','{\fontsize{20} ln({AGB_{fit}}) = 0.37358 + 0.37358(ln({D^2}H))}','{\fontsize{20}+/-95% ln(AGB) values}',...
        '{\fontsize{20}+/- 95% ln({AGB_{fit}}) values}','location','northwest')
legend boxoff
xlabel('{\fontsize{25}ln({D^2}H)}'); ylabel('{\fontsize{25}ln(AGB)}');
grid on     




m = length (AGB)
F = [ones(m, 1), D2HW(:)];
%[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
%AGBFit = F * p;
AGBFit4 = F * Adam_theta4;
AGBFit4Var = var (AGBFit4)
AGBVar = var(AGB)
%AGBFit3 = F * Adam_theta1;
figure ()
plot(D2HW, AGB, '+b', D2HW, AGBFit4, '-g',...
     D2HW, AGBFit4 + 1.96 * sqrt (AGBVar), '--r',...
     D2HW, AGBFit4 + 1.96 * sqrt (AGBFit4Var), '--k',...
     D2HW, AGBFit4 - 1.96 * sqrt (AGBVar), '--r',...
     D2HW, AGBFit4 - 1.96 * sqrt (AGBFit4Var), '--k')
title (' ')
legend ('{\fontsize{20}data}','{\fontsize{20} ln({AGB_{fit}}) = 0.39145 + 0.39145(ln({D^2}HW))}','{\fontsize{20}+/-95% ln(AGB) values}',...
        '{\fontsize{20}+/- 95% ln({AGB_{fit}}) values}','location','northwest')
legend boxoff
xlabel('{\fontsize{25}ln({D^2}HW)}'); ylabel('{\fontsize{25}ln(AGB)}');
grid on    