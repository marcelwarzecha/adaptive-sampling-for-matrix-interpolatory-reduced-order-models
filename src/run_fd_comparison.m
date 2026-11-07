function res = run_fd_comparison(cfg, objIdx, level, seed)
%RUN_FD_COMPARISON fmincon with finite-difference gradients directly on the FOM or on a local ROM.
%   level = 'FOM': every objective evaluation assembles and solves the FOM.
%   level = 'ROM': every objective evaluation assembles the FOM and reduces
%   it with IRKA (order cfg.fd.ROM_order) before solving (non-parametric ROM).
%   The objective is ln R with the L_k norm of paper Eq. (2).

rng(seed, 'twister');
obj = cfg.objectives(objIdx);
lb = cfg.model.lb;
ub = cfg.model.ub;
z0 = lb + (ub - lb) .* rand(1, 2);

switch level
    case 'FOM'
        objectiveFunction = @(z) fd_objective_fom(cfg, z, obj.f_opt, obj.p);
    case 'ROM'
        objectiveFunction = @(z) fd_objective_rom(cfg, z, obj.f_opt, obj.p);
    otherwise
        error('run_fd_comparison:level', 'Unknown level "%s".', level);
end

t = tic;
[p_hat, lnR_hat, exitflag, output] = fmincon(objectiveFunction, z0, [], [], [], [], lb, ub, [], ...
    fmincon_options(cfg, cfg.fd.MaxIterations, false));
res.time.total = toc(t);

res.objective = obj;
res.level = level;
res.seed = seed;
res.z0 = z0;
res.p_hat = p_hat;
res.lnR = lnR_hat;
res.fmincon_exitflag = exitflag;
res.fmincon_iterations = output.iterations;
res.fmincon_funccount = output.funcCount;
if strcmp(level, 'ROM')
    % LOGGING: FOM objective at the ROM optimum (not timed)
    [K, D, M, F, G] = assemble_fom(cfg, p_hat);
    res.lnR_fom = fom_lnR(K, D, M, F, G, obj.f_opt, obj.p);
else
    res.lnR_fom = lnR_hat;
end
end

function lnR = fd_objective_fom(cfg, z, f_opt, p)
[K, D, M, F, G] = assemble_fom(cfg, z);
lnR = fom_lnR(K, D, M, F, G, f_opt, p);
end

function lnR = fd_objective_rom(cfg, z, f_opt, p)
[K, D, M, F, G] = assemble_fom(cfg, z);
[Kr, Dr, Mr, Fr, Gr] = IRKA_reduction({K}, {D}, {M}, {F}, {G}, cfg.fd.IRKA_f0, ...
    cfg.mor.IRKA_tol, cfg.mor.IRKA_maxiter);
lnR = fom_lnR(Kr{1}, Dr{1}, Mr{1}, Fr{1}, Gr{1}, f_opt, p);
end
