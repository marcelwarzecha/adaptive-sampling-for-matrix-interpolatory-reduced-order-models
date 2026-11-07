function res = run_adaptive(cfg, objIdx, seed)
%RUN_ADAPTIVE One run of the adaptive sampling loop (paper Algorithm 3).
%   res = RUN_ADAPTIVE(cfg, objIdx, seed) runs the loop for objective
%   cfg.objectives(objIdx), with rng(seed, 'twister').
%
%   Everything that is only recorded for the paper (FOM objective at the
%   samples, posterior-mean objective, definiteness checks, surrogate
%   surfaces, final error map) happens in blocks marked LOGGING: they use
%   no random numbers and their time is excluded from all reported runtimes.

rng(seed, 'twister');
obj = cfg.objectives(objIdx);
f_opt = obj.f_opt;
p = obj.p;
lb = cfg.model.lb;
ub = cfg.model.ub;
numParams = cfg.model.numParams;
t_log = 0;                                   % accumulated logging time

t_main = tic;
%% Initial sample set (full-factorial design) and FOMs
factors = cfg.design.factors;
S_FFD = fullfact(repmat(factors, 1, numParams));
S_unit = (S_FFD - ones(size(S_FFD))) ./ (factors - 1);
S = transformToPhysicalSpace(S_unit, cfg.model.Bounds);
numInitial = size(S, 1);

tic;
[K, D, M, F, G] = assemble_foms(cfg, S);
time.initial_sampling = toc;

%% Initial local reduction
tic;
[~, ~, ~, ~, ~, Vr] = IRKA_reduction(K, D, M, F, G, cfg.mor.IRKA_f0, cfg.mor.IRKA_tol, cfg.mor.IRKA_maxiter);
time.local_reduction = toc;
time.sample_adding = [];
time.reprojection = [];
time.training = [];
time.optimization = [];

%% LOGGING: true objective at the initial samples
tl = tic;
lnR_fom_samples = nan(numInitial, 1);
if cfg.log.true_objective_at_samples
    for i = 1:numInitial
        lnR_fom_samples(i) = fom_lnR(K{i}, D{i}, M{i}, F{i}, G{i}, f_opt, p);
    end
end
t_log = t_log + toc(tl);

%% Adaptive sampling loop
outerloop_eps = 1;
outerloop_count = 1;
outerloop_cost = [];
start_it = cfg.opt.start_it;
sample_origin = zeros(numInitial, 1);        % 0 = initial, k = added after iteration k
moved = false(numInitial, 1);
it = struct([]);
SAoptimalParams = [];

while ((outerloop_eps > cfg.loop.outerloop_tol) && (outerloop_count <= cfg.loop.outerloop_maxit)) ...
        || (outerloop_count < cfg.loop.outerloop_minit)

    %% Add optimum of the previous iteration (distance check) and its FOM
    if outerloop_count > 1
        tic;
        [newSample, dc] = distance_check(SAoptimalParams, S, cfg.loop.min_dist_threshold, ...
            cfg.loop.checkdistance_maxit, lb, ub);
        S = [S; newSample]; %#ok<AGROW>
        S_unit = [S_unit; transformToUnitSpace(newSample, cfg.model.Bounds)]; %#ok<AGROW>
        [K_new, D_new, M_new, F_new, G_new] = assemble_fom(cfg, newSample);
        K{end+1} = K_new; D{end+1} = D_new; M{end+1} = M_new; F{end+1} = F_new; G{end+1} = G_new;
        time.sample_adding(end+1) = toc;

        tic;
        [~, ~, ~, ~, ~, Vr_new] = IRKA_reduction(K_new, D_new, M_new, F_new, G_new, ...
            cfg.mor.IRKA_f0, cfg.mor.IRKA_tol, cfg.mor.IRKA_maxiter);
        Vr{end+1} = cell2mat(Vr_new);
        time.local_reduction(end+1) = toc;

        % LOGGING
        tl = tic;
        sample_origin(end+1, 1) = outerloop_count - 1; %#ok<AGROW>
        moved(end+1, 1) = dc.moved; %#ok<AGROW>
        it(outerloop_count - 1).distance_check = dc;
        lnR_fom_samples(end+1, 1) = NaN; %#ok<AGROW>
        if cfg.log.true_objective_at_samples
            lnR_fom_samples(end) = fom_lnR(K_new, D_new, M_new, F_new, G_new, f_opt, p);
        end
        t_log = t_log + toc(tl);
    end

    %% Global basis and reprojection
    tic;
    [R, ROM_order] = getGlobalBasis(Vr, cfg.mor.SVD_retain_frac);
    [Kr_t, Dr_t, Mr_t, Fr_t, Gr_t] = transformToGlobalBasis(K, D, M, F, G, R);
    [Kr, Dr, Mr, Fr, Gr] = Cell2Array(Kr_t, Dr_t, Mr_t, Fr_t, Gr_t);
    time.reprojection(end+1) = toc;

    %% Model training (SBR, one Thompson draw per entry)
    tic;
    prom = build_prom(cfg, S, S_unit, Kr, Dr, Mr, Fr, Gr);
    time.training(end+1) = toc;

    %% Gradient-based optimization on the Thompson-sampled pROM
    tic;
    objectiveFunction = @(z) EvaluateCostAndGradient(prom.C_ts, z, f_opt, p, ...
        cfg.opt.frequency_solver);
    maxIterations = min(start_it, cfg.opt.maxit);
    options = fmincon_options(cfg, maxIterations, true);
    z0 = lb + (ub - lb) .* rand(1, numParams);           % no warm start
    [SAoptimalParams, SAoptimalCost, exitflag, output] = fmincon(objectiveFunction, z0, ...
        [], [], [], [], lb, ub, [], options);
    time.optimization(end+1) = toc;

    %% LOGGING: iteration record
    tl = tic;
    k = outerloop_count;
    it(k).iteration = k;
    it(k).n_samples = size(S, 1);
    it(k).r = ROM_order;
    it(k).poly_order = prom.order;
    it(k).z0 = z0;
    it(k).p_hat = SAoptimalParams;
    it(k).lnR_ts = SAoptimalCost;
    it(k).lnR_mean = prom_lnR(prom, SAoptimalParams, false, f_opt, p);
    it(k).fmincon_maxit = maxIterations;
    it(k).fmincon_exitflag = exitflag;
    it(k).fmincon_iterations = output.iterations;
    it(k).fmincon_funccount = output.funcCount;
    mdls = [prom.Mdl_K, prom.Mdl_D, prom.Mdl_M];
    it(k).n_entry_models = numel(mdls);
    it(k).n_ts_skipped = sum(cellfun(@(m) isfield(m, 'ts_skipped') && m.ts_skipped, mdls));
    if cfg.log.pd_grid_points > 0
        g = linspace(0, 1, cfg.log.pd_grid_points);
        [G1, G2] = meshgrid(g, g);
        Zpd = [SAoptimalParams; transformToPhysicalSpace([G1(:), G2(:)], cfg.model.Bounds)];
        it(k).pd_points = Zpd;
        [it(k).min_eig_ts, it(k).max_eig_ts] = prom_min_eig(prom, Zpd, true);      % [K D M]
        [it(k).min_eig_mean, it(k).max_eig_mean] = prom_min_eig(prom, Zpd, false);
    end
    every = cfg.log.surface_every;
    if cfg.log.surface_points > 0 && (k == 1 || mod(k, every) == 0)
        g = linspace(0, 1, cfg.log.surface_points);
        [G1, G2] = meshgrid(g, g);
        Zs = transformToPhysicalSpace([G1(:), G2(:)], cfg.model.Bounds);
        it(k).surface_points = Zs;
        it(k).surface_lnR_ts = prom_lnR(prom, Zs, true, f_opt, p);
        it(k).surface_lnR_mean = prom_lnR(prom, Zs, false, f_opt, p);
    end
    t_log = t_log + toc(tl);

    %% Stopping criterion: largest change |ln R_k - ln R_(k-1)| of the optimum
    %  values over the last cfg.loop.stop_window iterations
    outerloop_cost = [outerloop_cost; SAoptimalCost]; %#ok<AGROW>
    it(k).eps_step = NaN;
    if outerloop_count >= 2
        it(k).eps_step = abs(outerloop_cost(end) - outerloop_cost(end-1));
        if k >= cfg.loop.stop_window + 1
            outerloop_eps = max([it(k - cfg.loop.stop_window + 1:k).eps_step]);
        end
    end
    it(k).eps = outerloop_eps;
    outerloop_count = outerloop_count + 1;
    start_it = start_it + 2;
end
time.total_wall = toc(t_main);
time.logging = t_log;
time.total = time.total_wall - t_log;          % comparable to "Total runtime" in the paper

%% LOGGING: final quantities (not timed)
res.objective = obj;
res.seed = seed;
res.samples.S = S;
res.samples.origin = sample_origin;
res.samples.moved = moved;
res.samples.lnR_fom = lnR_fom_samples;
res.iterations = it;
res.time = time;
res.final.n_iterations = numel(it);
res.final.n_initial = numInitial;
res.final.n_added = size(S, 1) - numInitial;
res.final.n_total = size(S, 1);
res.final.r = prom.r;
res.final.poly_order = prom.order;
res.final.p_hat = SAoptimalParams;
res.final.lnR_ts = SAoptimalCost;
res.final.lnR_mean = it(end).lnR_mean;
[Kf, Df, Mf, Ff, Gf] = assemble_fom(cfg, SAoptimalParams);
res.final.lnR_fom = fom_lnR(Kf, Df, Mf, Ff, Gf, f_opt, p);
if cfg.log.final_testgrid_eval
    res.testgrid = evaluate_testgrid(cfg, prom, cfg.log.testgrid_stride);
end
end
