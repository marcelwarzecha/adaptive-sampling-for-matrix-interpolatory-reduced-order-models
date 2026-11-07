function cfg = production_config(modelName)
%PRODUCTION_CONFIG Settings of all runs reported in the paper (Tables 1-3).
%   cfg = PRODUCTION_CONFIG('KelvinCell') or PRODUCTION_CONFIG('Beam').
%   The queues in build_queue.m change only the logging and, for the smoke
%   test, the number of iterations.

cfg.model.name = modelName;
cfg.model.numParams = 2;
cfg.model.damping = true;

switch modelName
    case 'KelvinCell'
        cfg.model.Bounds = [0.055 0.02; 0.08 0.045];      % [lx ly] [m]
        cfg.design.factors = 3;                            % 3^2 FFD initial design
        cfg.mor.ROM_order = 20;                            % local IRKA order
        cfg.mor.IRKA_f0 = linspace(-500, 500, 20);         % [Hz]
        cfg.loop.outerloop_maxit = 50;
        cfg.loop.outerloop_minit = 10;
        cfg.loop.min_dist_threshold = 0.025/50;            % d_min = 5e-4 m
        % Objective configurations
        cfg.objectives = struct( ...
            'id',    {'f300-400_k1', 'f300-400_k20', 'f200-400_k1', 'f200-400_k20'}, ...
            'f_opt', {linspace(300,400,101), linspace(300,400,101), linspace(200,400,201), linspace(200,400,201)}, ...
            'p',     {1, 20, 1, 20});
    case 'Beam'
        cfg.model.L = 1;
        cfg.model.Bounds = [0.01 0.01; 0.05 0.05];         % [t h] [m]
        cfg.design.factors = 2;                            % 2^2 FFD initial design
        cfg.mor.ROM_order = 6;
        cfg.mor.IRKA_f0 = linspace(-250, 250, 6);          % [Hz]
        cfg.loop.outerloop_maxit = 20;
        cfg.loop.outerloop_minit = 4;
        cfg.loop.min_dist_threshold = 0.025/20;            % d_min = 1.25e-3 m
        cfg.objectives = struct('id', {'f50-100_k1'}, 'f_opt', {linspace(50,100,51)}, 'p', {1});
    otherwise
        error('production_config:unknownModel', 'Unknown model "%s".', modelName);
end
cfg.model.lb = cfg.model.Bounds(1,:);
cfg.model.ub = cfg.model.Bounds(2,:);

% Model order reduction
cfg.mor.IRKA_tol = 1e-6;
cfg.mor.IRKA_maxiter = 10;
cfg.mor.SVD_retain_frac = 0.9995;

% Regression (SBR + Thompson sampling)
cfg.reg.method = 'SBR';
cfg.reg.parallel_training = true;       % fit the entry models in parfor, Thompson draws serially in a fixed order
% SBR settings, the same for every matrix entry and both models (SB1_fit, SB1_Estimate).
% Initialization as in Tipping's SB1_ExampleRegress.m: alpha_0 = 1/N^2, noise std
% noise_frac * std(Y).
cfg.reg.sbr.noise_frac = 0.1;           % initial noise std as a fraction of std(Y)
cfg.reg.sbr.maxits = 20;                % passes of the alpha/beta re-estimation
cfg.reg.sbr.min_delta_logalpha = 1;     % stop when no log(alpha) changes by more than this
cfg.reg.sbr.alpha_max = 1e9;            % prune basis functions with alpha above this
% Monomial basis (ModelTraining): total degree <= q, with the largest q whose number of
% terms does not exceed (number of samples - 1), q <= max_degree (paper Eq. (38)).
cfg.reg.sbr.max_degree = 6;

% Gradient-based optimization (fmincon)
cfg.opt.start_it = 10;                  % MaxIterations = min(start_it + 2(k-1), maxit)
cfg.opt.maxit = 20;
cfg.opt.OptimalityTolerance = 1e-6;
cfg.opt.StepTolerance = 1e-6;
cfg.opt.frequency_solver = 'pages';     % 'loop' | 'pages' (pagemldivide, faster)

% Outer loop: stop when |ln R_k - ln R_(k-1)| of the optimum values stays below
% outerloop_tol in stop_window consecutive iterations (at least outerloop_minit - 1
% iterations, at most outerloop_maxit).
cfg.loop.outerloop_tol = 5e-3;
cfg.loop.stop_window = 3;
cfg.loop.checkdistance_maxit = 5;

% Logging that does not influence the algorithm (excluded from all timings, uses no RNG)
cfg.log.true_objective_at_samples = true;   % FOM objective at every added sample
cfg.log.pd_grid_points = 3;                 % definiteness check on a 3x3 grid (+ at found optimum)
cfg.log.surface_points = 0;                 % >0: store TS and posterior-mean surrogate surfaces (n x n) every surface_every-th iteration
cfg.log.surface_every = 1;                  % iterations 1, every, 2*every, ...
cfg.log.final_testgrid_eval = strcmp(modelName, 'KelvinCell');  % posterior-mean pROM error on the 51x51 reference grid
cfg.log.testgrid_stride = 1;                % 1 = all 2601 reference points

% Finite-difference comparison (paper Table 8)
cfg.fd.MaxIterations = 100;
cfg.fd.ROM_order = 50;
cfg.fd.IRKA_f0 = linspace(-1000, 1000, 50); % [Hz]

% Environment
cfg.env.n_workers = 12;                     % paper Table 3: 12 parallel threads
end
