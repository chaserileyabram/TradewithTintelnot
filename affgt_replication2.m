% Assignment: Topics in International Trade and Growth
% Replication of Antras, Fadeev, Fort, Gutierrez, and Tintelnot (2021)
% Import Tariffs and Global Sourcing
% Closed Economy
% Ignacia Cuevas

clear
clc

% Load data

load('/Users/Nacha/Documents/PhD Economics/Second year/Third quarter/Topics in International Trade and Growth/Assignment/AFFGT_data.mat')

%% Calibration

% Set parameters values

% A. Fixed Values

param.theta = 4; % Elasticity of substitution, input varieties
param.sigma = 4; % Elasticity of substitution, final-good varieties
param.f_d = 1; % Entry costs, final-good sector
param.f_u = 1; % Entry costs, input sector
param.A_d = 1; % US productivity in downstream sector
param.A_u = 1;  % US productivity in upstream sector

% B. Values Measured From Data

% 1 - alpha: captures the share of inputs in production, as the share of
% revenue used to pay for intermediate inputs in production. They compute
% the input revenue share using data for the US from the WIOD
% Population data from CEPII to calibrate the labor endowment in each
% country.
% All data is for 2007

param.alpha = 1-data.Input_share_downs(1); % From data
param.L = 10*(data.L(1)/sum(data.L)); % From data

% Targeted moments

%% Equilibrium

% I want: p_d, p_u, w, x_d, x_u, q_d, q_u, l_d, l_u, M_d, M_u

% Starting point

x_0(1:1,1) = unifrnd(0.05,0.5,1,1); % p_u
x_0(2:2,1) = unifrnd(0.05,0.5,1,1); % M_u
x_0(3:3,1) = unifrnd(0.05,0.5,1,1); % M_d

%% Solver

[x] = fsolve(@(x)fun(x,param),x_0);
