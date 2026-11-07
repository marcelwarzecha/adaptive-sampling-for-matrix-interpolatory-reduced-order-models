function res = replay_adaptive(cfg, resultFile, nStep)
%REPLAY_ADAPTIVE Accuracy near p* of the intermediate pROMs of a finished adaptive run.
%   res = REPLAY_ADAPTIVE(cfg, resultFile, nStep) loads an adaptive result
%   (run_adaptive) and rebuilds the posterior-mean pROM from its first n
%   samples, in the order they were added, for n = n_initial + minit - 2
%   (the earliest iteration at which the loop can stop) to the final N in
%   steps of nStep (default 1; the final N is always included). This is the
%   posterior-mean pROM the loop had after iteration k = n - n_initial + 1:
%   the posterior mean does not depend on the Thompson draws. Each pROM is
%   evaluated with evaluate_testgrid at the grid points within 2.5 mm of the
%   true optimum of the run's objective (all objectives' error measures).
%
%   Checks: ln R of the rebuilt pROM at the logged optimum of iteration k
%   must equal the logged posterior-mean value (res.replay(j).lnR_mean_diff),
%   and for n = N the errors must equal those of the stored final test-grid
%   evaluation (res.final_check). FOMs and local bases are computed as in
%   run_adaptive (initial design as one parallel batch, every added sample
%   on the client), so agreement is up to round-off.
if nargin < 3
    nStep = 1;
end
NEAR_RADIUS = 2.5e-3;                        % [m]

R = load(resultFile, 'res');
R = R.res;
obj = R.objective;
S = R.samples.S;
N = size(S, 1);
n0 = R.final.n_initial;

% Unit-space coordinates as in run_adaptive: FFD formula for the initial
% design, transformToUnitSpace for the added samples
factors = cfg.design.factors;
S_unit0 = (fullfact(repmat(factors, 1, cfg.model.numParams)) - 1) ./ (factors - 1);
assert(max(abs(transformToPhysicalSpace(S_unit0, cfg.model.Bounds) - S(1:n0, :)), [], 'all') < 1e-12, ...
    'replay_adaptive:initial', 'Initial samples of %s do not match cfg.design.', resultFile);
S_unit = [S_unit0; transformToUnitSpace(S(n0+1:end, :), cfg.model.Bounds)];

% Grid points near the true optimum of this objective
Te = load_testgrid(cfg);
f_eval = 1:500;
band = ismember(f_eval, round(obj.f_opt));
lnR_true = log(sum(abs(Te.y(f_eval(band), :)).^obj.p, 1).^(1/obj.p));
[~, i_star] = min(lnR_true);
p_star = Te.S(i_star, :);
idx = find(vecnorm(Te.S - p_star, 2, 2) <= NEAR_RADIUS);

% FOMs and local bases exactly as in run_adaptive: the initial design as one
% parallel batch, every added sample on the client. IRKA stops at a
% tolerance, so a basis computed on a worker (single-threaded BLAS) differs
% from the client's by more than round-off, and the global basis and the SBR
% fits amplify that (the order r and ln R can differ noticeably).
t_build = tic;
[K, D, M, F, G] = assemble_foms(cfg, S(1:n0, :));
[~, ~, ~, ~, ~, Vr] = IRKA_reduction(K, D, M, F, G, cfg.mor.IRKA_f0, cfg.mor.IRKA_tol, cfg.mor.IRKA_maxiter);
for i = n0+1:N
    [K{i}, D{i}, M{i}, F{i}, G{i}] = assemble_fom(cfg, S(i, :));
    [~, ~, ~, ~, ~, Vr_new] = IRKA_reduction(K{i}, D{i}, M{i}, F{i}, G{i}, ...
        cfg.mor.IRKA_f0, cfg.mor.IRKA_tol, cfg.mor.IRKA_maxiter);
    Vr{i} = cell2mat(Vr_new);
end
time.build = toc(t_build);

nList = unique([n0 + cfg.loop.outerloop_minit - 2 : nStep : N, N]);
nList = nList(nList >= n0);
rng(0, 'twister');                           % Thompson draws in build_prom (unused here)
rep = struct([]);
for j = 1:numel(nList)
    n = nList(j);
    k = n - n0 + 1;
    t_n = tic;
    [Rg, r] = getGlobalBasis(Vr(1:n), cfg.mor.SVD_retain_frac);
    [Kr_t, Dr_t, Mr_t, Fr_t, Gr_t] = transformToGlobalBasis(K(1:n), D(1:n), M(1:n), F(1:n), G(1:n), Rg);
    [Kr, Dr, Mr, Fr, Gr] = Cell2Array(Kr_t, Dr_t, Mr_t, Fr_t, Gr_t);
    prom = build_prom(cfg, S(1:n, :), S_unit(1:n, :), Kr, Dr, Mr, Fr, Gr);
    ev = evaluate_testgrid(cfg, prom, 1, idx);

    rep(j).n = n;
    rep(j).iteration = k;
    rep(j).r = r;
    rep(j).poly_order = prom.order;
    rep(j).meanRelErr = ev.meanRelErr;
    rep(j).bandRelErr = ev.bandRelErr;
    rep(j).lnR_prom = ev.lnR_prom;
    rep(j).lnR_mean_diff = NaN;
    if k <= numel(R.iterations)
        it = R.iterations(k);
        rep(j).lnR_mean_diff = prom_lnR(prom, it.p_hat, false, obj.f_opt, obj.p) - it.lnR_mean;
    end
    rep(j).time = toc(t_n);
    fprintf('[replay %s] n = %d (iteration %d): r = %d, order %d, median mean error near p* %.2f %%, ln R check %.1e (%.0f s)\n', ...
        obj.id, n, k, r, prom.order, 100 * median(ev.meanRelErr), rep(j).lnR_mean_diff, rep(j).time);
end

% Final pROM (n = N) against the stored test-grid evaluation of the run (full grid, same order)
assert(R.testgrid.stride == 1 && isequal(size(R.testgrid.S), size(Te.S)) && ...
    max(abs(R.testgrid.S - Te.S), [], 'all') < 1e-12, 'replay_adaptive:testgrid', ...
    'Stored test grid of %s is not the full reference grid.', resultFile);
final_check.max_abs_diff_meanRelErr = max(abs(rep(end).meanRelErr - R.testgrid.meanRelErr(idx)));
final_check.max_abs_diff_bandRelErr = max(abs(rep(end).bandRelErr - R.testgrid.bandRelErr(idx, :)), [], 'all');
fprintf('[replay %s] final pROM vs stored test grid: max |diff| mean error %.1e, band error %.1e\n', ...
    obj.id, final_check.max_abs_diff_meanRelErr, final_check.max_abs_diff_bandRelErr);

res.source = resultFile;
res.objective = obj;
res.seed = R.seed;
res.n_initial = n0;
res.n_final = N;
res.p_star = p_star;
res.grid_idx = idx;
res.grid_S = Te.S(idx, :);
res.objective_ids = ev.objective_ids;
res.replay = rep;
res.final_check = final_check;
res.time = time;
end
