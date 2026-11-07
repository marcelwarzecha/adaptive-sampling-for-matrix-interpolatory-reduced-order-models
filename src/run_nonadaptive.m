function res = run_nonadaptive(cfg, design, n, seed)
%RUN_NONADAPTIVE pROM from a fixed (non-adaptive) design, evaluated for all objectives.
%   design = 'FFD': n x n full-factorial grid;
%   design = 'LHS': n samples,
%   lhsdesign(n, 2, 'criterion', 'maximin', 'iterations', 1000).
%   The pROM is built with the identical pipeline as in the adaptive loop
%   (FOMs, IRKA, global basis, SBR with the same degree rule). For each
%   objective, one optimization (fmincon with analytical gradients) is run
%   on the posterior-mean pROM from a random starting point.

rng(seed, 'twister');
lb = cfg.model.lb;
ub = cfg.model.ub;

switch design
    case 'FFD'
        levels = [n n];
        S_FFD = fullfact(levels);
        S_unit = (S_FFD - 1) ./ (levels - 1);
    case 'LHS'
        S_unit = lhsdesign(n, 2, 'criterion', 'maximin', 'iterations', 1000);
    otherwise
        error('run_nonadaptive:design', 'Unknown design "%s".', design);
end
S = transformToPhysicalSpace(S_unit, cfg.model.Bounds);

t_build = tic;
tic;
[K, D, M, F, G] = assemble_foms(cfg, S);
time.initial_sampling = toc;
tic;
[~, ~, ~, ~, ~, Vr] = IRKA_reduction(K, D, M, F, G, cfg.mor.IRKA_f0, cfg.mor.IRKA_tol, cfg.mor.IRKA_maxiter);
time.local_reduction = toc;
tic;
[R, ~] = getGlobalBasis(Vr, cfg.mor.SVD_retain_frac);
[Kr_t, Dr_t, Mr_t, Fr_t, Gr_t] = transformToGlobalBasis(K, D, M, F, G, R);
[Kr, Dr, Mr, Fr, Gr] = Cell2Array(Kr_t, Dr_t, Mr_t, Fr_t, Gr_t);
time.reprojection = toc;
tic;
prom = build_prom(cfg, S, S_unit, Kr, Dr, Mr, Fr, Gr);
time.training = toc;
time.build_total = toc(t_build);

% Optimization on the posterior-mean pROM, once per objective
opt = struct([]);
for o = 1:numel(cfg.objectives)
    obj = cfg.objectives(o);
    objectiveFunction = @(z) EvaluateCostAndGradient(prom.C_mean, z, obj.f_opt, obj.p, ...
        cfg.opt.frequency_solver);
    z0 = lb + (ub - lb) .* rand(1, 2);
    t_opt = tic;
    [p_hat, lnR_hat, exitflag, output] = fmincon(objectiveFunction, z0, [], [], [], [], lb, ub, [], ...
        fmincon_options(cfg, cfg.opt.maxit, true));
    opt(o).objective_id = obj.id;
    opt(o).time = toc(t_opt);
    opt(o).z0 = z0;
    opt(o).p_hat = p_hat;
    opt(o).lnR_mean = lnR_hat;
    opt(o).fmincon_exitflag = exitflag;
    opt(o).fmincon_iterations = output.iterations;
    opt(o).fmincon_funccount = output.funcCount;
    % LOGGING: true objective at the found optimum (not timed)
    [Kf, Df, Mf, Ff, Gf] = assemble_fom(cfg, p_hat);
    opt(o).lnR_fom = fom_lnR(Kf, Df, Mf, Ff, Gf, obj.f_opt, obj.p);
end

res.design = design;
res.n = size(S, 1);
res.seed = seed;
res.samples.S = S;
res.r = prom.r;
res.poly_order = prom.order;
res.time = time;
res.optimization = opt;
if cfg.log.final_testgrid_eval
    res.testgrid = evaluate_testgrid(cfg, prom, cfg.log.testgrid_stride);
end
end
