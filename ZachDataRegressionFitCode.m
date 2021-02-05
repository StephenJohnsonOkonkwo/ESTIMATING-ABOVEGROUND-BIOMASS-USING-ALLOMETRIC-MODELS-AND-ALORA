%%%% REGRESSION PLOTS
pkg load optim
m = length (AGB)
F = [ones(m, 1), D(:)];
[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
p
p_var
AGBFit = F * p;
figure ()
plot(D, AGB, '+b', D, AGBFit, '-g',...
      D, AGBFit + 1.96 * sqrt (e_var), '--r',...
      D, AGBFit + 1.96 * sqrt (fit_var), '--k',...
      D, AGBFit - 1.96 * sqrt (e_var), '--r',...
      D, AGBFit - 1.96 * sqrt (fit_var), '--k')
title ('')
legend ('{\fontsize{15}data}','{\fontsize{15}fit}','{\fontsize{15}+/-95% ln(AGB) values}',...
        '{\fontsize{15}+/- 95% fitted values}','location','northwest')
xlabel('{\fontsize{15}ln(D)}'); ylabel('{\fontsize{15}ln(AGB)}');
grid on


F = [ones(m, 1), DH(:)];
[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
p
AGBFit = F * p;
figure ()
plot(DH, AGB, '+b', DH, AGBFit, '-g',...
      DH, AGBFit + 1.96 * sqrt (e_var), '--r',...
      DH, AGBFit + 1.96 * sqrt (fit_var), '--k',...
      DH, AGBFit - 1.96 * sqrt (e_var), '--r',...
      DH, AGBFit - 1.96 * sqrt (fit_var), '--k')
title (' ')
legend ('{\fontsize{15}data}','{\fontsize{15}fit}','{\fontsize{15}+/-95% ln(AGB) values}',...
        '{\fontsize{15}+/- 95% fitted values}','location','northwest')
xlabel('{\fontsize{15}ln(DH)}'); ylabel('{\fontsize{15}ln(AGB)}');
grid on


F = [ones(m, 1), D2H(:)];
[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
p
AGBFit = F * p;
figure ()
plot(D2H, AGB, '+b', D2H, AGBFit, '-g',...
      D2H, AGBFit + 1.96 * sqrt (e_var), '--r',...
      D2H, AGBFit + 1.96 * sqrt (fit_var), '--k',...
      D2H, AGBFit - 1.96 * sqrt (e_var), '--r',...
      D2H, AGBFit - 1.96 * sqrt (fit_var), '--k')
title (' ')
legend ('{\fontsize{15}data}','{\fontsize{15}fit}','{\fontsize{15}+/-95% ln(AGB) values}',...
        '{\fontsize{15}+/- 95% fitted values}','location','northwest')
xlabel('{\fontsize{15}ln({D^2}H)}'); ylabel('{\fontsize{15}ln(AGB)}');
grid on


F = [ones(m, 1), D(:)];
[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
p
AGBFit = F * p;
figure ()
plot(D2HW, AGB, '+b', D2HW, AGBFit, '-g',...
      D2HW, AGBFit + 1.96 * sqrt (e_var), '--r',...
      D2HW, AGBFit + 1.96 * sqrt (fit_var), '--k',...
      D2HW, AGBFit - 1.96 * sqrt (e_var), '--r',...
      D2HW, AGBFit - 1.96 * sqrt (fit_var), '--k')
title (' ')
legend ('{\fontsize{15}data}','{\fontsize{15}fit}','{\fontsize{15}+/-95% ln(AGB) values}',...
        '{\fontsize{15}+/- 95% fitted values}','location','northwest')
xlabel('{\fontsize{15}ln({D^2}HW)}'); ylabel('{\fontsize{15}ln(AGB)}');
grid on



pkg load optim


m = length (AGB)
F = [ones(m, 1), D(:)];
%[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
%AGBFit = F * p;
AGBFit2 = F * RMSProp_theta1;
%AGBFit3 = F * Adam_theta1;
figure ()
plot(D, AGB, '+b', D, AGBFit2, '-g',...
      

E_RMSProp1var = var(E_RMSProp1)
AGBSGD_cap1RMSPropVar = var(AGBSGD_cap1RMSProp)
AGB_testVar = var (AGB_test)

m_test = length (AGB_test)
F = [ones(m_test, 1), D_test(:)];
%[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
%AGBFit = F * p;
AGBFit2 = F * RMSProp_theta1;
%AGBFit3 = F * Adam_theta1;
figure ()
plot(D_test, AGB_test, '+b', D_test, AGBFit2, '-g',...
     D_test, AGBFit2 + 1.96 * sqrt (AGB_testVar), '--r',...
     D_test, AGBFit2 + 1.96 * sqrt (AGBSGD_cap1RMSPropVar), '--k',...
     D_test, AGBFit2 - 1.96 * sqrt (AGB_testVar), '--r',...
     D_test, AGBFit2 - 1.96 * sqrt (AGBSGD_cap1RMSPropVar), '--k')



 


 


m = length (AGB)
F = [ones(m, 1), D2H(:)];
[p, e_var, r, p_var, fit_var] = LinearRegression (F, AGB);
AGBFit = F * p;
AGBFit2 = F * RMSProp_theta1;
AGBFit3 = F * Adam_theta1;
figure ()
plot(D2H, AGB, '+b', D2H, AGBFit, '-g',...
      D2H, AGBFit2, '-r',...
      D2H, AGBFit3, '-k')

      


   


hist(AGB)
xlabel('{\fontsize{25}ln(AGB)}'); ylabel('{\fontsize{25}Frequency}');
grid on

hist(D, "facecolor", 'purple')
xlabel('{\fontsize{25}ln(D)}'); ylabel('{\fontsize{25}Frequency}');
grid on

hist(DH, "facecolor", 'b')
xlabel('{\fontsize{25}ln(DH)}'); ylabel('{\fontsize{25}Frequency}');
grid on

hist(D2H, "facecolor", 'g')
xlabel('{\fontsize{25}ln({D^2}H)}'); ylabel('{\fontsize{25}Frequency}');
grid on

hist(D2HW, "facecolor", 'y')
xlabel('{\fontsize{25}ln({D^2}HW)}'); ylabel('{\fontsize{25}Frequency}');
grid on