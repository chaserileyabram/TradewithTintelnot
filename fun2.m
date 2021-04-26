% Assignment: Topics in International Trade and Growth
% Replication of Antras, Fadeev, Fort, Gutierrez, and Tintelnot (2021)
% Import Tariffs and Global Sourcing
% Closed economy
% Ignacia Cuevas

function const = fun(x,param)

% 1. Parameters

theta = param.theta; % Elasticity of substitution, input varieties
sigma = param.sigma; % Elasticity of substitution, final-good varieties
f_d = param.f_d; % Entry costs, final-good sector
f_u = param.f_u; % Entry costs, input sector
A_d = param.A_d; % US productivity in downstream sector
A_u = param.A_u; % US productivity in upstream sector
L = param.L; % Population
alpha = param.alpha; % captures the share of inputs in production, as the share of
% revenue used to pay for intermediate inputs in production.
w=1; % wage normalized to 1

% 2. Functions

% Change endogenous variables p_u, M_u, M_d by X

mc_u = w/A_u; % Marginal cost upstream sector
p_u = (theta/(theta-1))*mc_u; % upstream sector's price
P_u = x(1)*(x(2)^(1/(1-theta))); %P_u = M_u^(1/(1-theta))*p_u; % upstream sector's price index
mc_d = (1/A_d)*((w^alpha)*(P_u)^(1-alpha))/(alpha^alpha*(1-alpha)^(1-alpha)); % Marginal cost downstream sector
p_d = sigma/(sigma-1)*mc_d; % Downstream sector's price
P_d = (x(3)^(1/(1-sigma)))*p_d; % Downstream sector's price index
q_d = w*L*(p_d^(-sigma)/P_d^(1-sigma)); % Quantity consumed of variety omega in downstream sector
x_d = (sigma-1)*f_d; % Output produced for sale by variety omega in downstream sector
l_d = alpha*(p_d*x_d)/w; % Labor force in downstream sector
l_u = theta*f_u/A_u; % Labor force in upstream sector
q_u = (1-alpha)*(mc_d*(f_d + x_d)/P_u^(1-theta))*x(1)^(-theta); % Quantity consumed of variety omega in upstream sector
x_u = (theta-1)*f_u; % Output produced for sale by variety omega in upstream sector

% 3. Equilibrium

% 3.1. Labor market clearing

MC_L = L - x(3)*l_d - x(2)*l_u;

% 3.2. Free entry

pi_d = (p_d - mc_d)*x_d - mc_d * f_d; %mc_d + p_d*x_d - mc_d*(x_d+f_d+1);

pi_u = x(1)*x_u - w*(theta*f_u/A_u);

% 3.3. Goods market clearing

MC_G_u = x_u - x(3)*q_u;
MC_G_d = q_d - x_d;

% p_u
%pu = p_u - x(1);

% M_u
%Mu = M_u - x(2);

% M_d
%Md = M_d - x(3);

% 4. Output

const = [MC_G_u;MC_G_d;pi_u];

end