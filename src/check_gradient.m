function res = check_gradient(cfg, seed)
%CHECK_GRADIENT Compare the adjoint gradient with direct sensitivities and central finite differences.
%   Builds a Thompson-sampled pROM from the initial design and evaluates,
%   at five random points, d(ln R)/dp from
%     (1) EvaluateCostAndGradient (adjoint, used in all runs),
%     (2) direct sensitivities: one additional solve per parameter and frequency,
%     (3) central finite differences of ln R.
%   (1) and (2) must agree to round-off, (1) and (3) to the FD truncation error.

rng(seed, 'twister');
obj = cfg.objectives(1);
factors = cfg.design.factors;
S_unit = (fullfact([factors factors]) - 1) ./ (factors - 1);
S = transformToPhysicalSpace(S_unit, cfg.model.Bounds);
[K, D, M, F, G] = assemble_foms(cfg, S);
[~, ~, ~, ~, ~, Vr] = IRKA_reduction(K, D, M, F, G, cfg.mor.IRKA_f0, cfg.mor.IRKA_tol, cfg.mor.IRKA_maxiter);
[R, ~] = getGlobalBasis(Vr, cfg.mor.SVD_retain_frac);
[Kr_t, Dr_t, Mr_t, Fr_t, Gr_t] = transformToGlobalBasis(K, D, M, F, G, R);
[Kr, Dr, Mr, Fr, Gr] = Cell2Array(Kr_t, Dr_t, Mr_t, Fr_t, Gr_t);
prom = build_prom(cfg, S, S_unit, Kr, Dr, Mr, Fr, Gr);
costgrad = @(z) EvaluateCostAndGradient(prom.C_ts, z, obj.f_opt, obj.p, cfg.opt.frequency_solver);

lb = cfg.model.lb; ub = cfg.model.ub;
h = 1e-6 * (ub - lb);
nPts = 5;
res.points = lb + (ub - lb) .* (0.1 + 0.8 * rand(nPts, 2));
res.grad_adjoint = zeros(nPts, 2);
res.grad_direct = zeros(nPts, 2);
res.grad_fd = zeros(nPts, 2);
for i = 1:nPts
    z = res.points(i, :);
    [c0, res.grad_adjoint(i, :)] = costgrad(z);
    res.grad_direct(i, :) = gradient_direct(prom.C_ts, z, obj.f_opt, obj.p, exp(c0));
    for j = 1:2
        e = zeros(1, 2); e(j) = h(j);
        res.grad_fd(i, j) = (costgrad(z + e) - costgrad(z - e)) / (2 * h(j));
    end
end
res.relerr_adjoint_vs_direct = max(abs(res.grad_adjoint - res.grad_direct), [], 2) ./ max(abs(res.grad_direct), [], 2);
res.relerr_adjoint_vs_fd = max(abs(res.grad_adjoint - res.grad_fd), [], 2) ./ max(abs(res.grad_fd), [], 2);
res.pass = all(res.relerr_adjoint_vs_direct < 1e-8) && all(res.relerr_adjoint_vs_fd < 1e-3);
fprintf('[check_gradient %s] max rel. diff adjoint vs direct: %.2e, vs FD: %.2e -> %s\n', ...
    cfg.model.name, max(res.relerr_adjoint_vs_direct), max(res.relerr_adjoint_vs_fd), string(res.pass));
end

function dRdp = gradient_direct(C, z, f_opt, p, cost)
% Direct sensitivities: dx_i/dp_k = K_dyn,i \ (-(dK + s_i dD + s_i^2 dM) x_i) for every
% parameter and frequency, then d(ln R)/dp_k = (1/R) 2 Re(sum_i lambda_i G dx_i/dp_k).
z_unit = transformToUnitSpace(z, C.Bounds);
[K, D, M, F, G] = EvaluateCompiledModels(C, z_unit);
[dK, dD, dM] = EvaluateCompiledDerivatives(C, z_unit);
n_f = numel(f_opt);
x = cell(1, n_f); y = zeros(1, n_f);
for i = 1:n_f
    s = 2 * pi * 1i * f_opt(i);
    x{i} = (K + s * D + s^2 * M) \ F;
    y(i) = G * x{i};
end
y_sum = sum(abs(y).^p);
d = numel(dK);
dRdp = zeros(1, d);
for i = 1:n_f
    s = 2 * pi * 1i * f_opt(i);
    Kd = K + s * D + s^2 * M;
    lambda = y_sum^(1/p - 1) * abs(y(i))^(p - 1) * 1/2 * conj(y(i)) / abs(y(i));
    for k = 1:d
        dRdp(k) = dRdp(k) + lambda * G * (Kd \ (-(dK{k} + s * dD{k} + s^2 * dM{k}) * x{i}));
    end
end
dRdp = (1 / cost) * 2 * real(dRdp);
end
