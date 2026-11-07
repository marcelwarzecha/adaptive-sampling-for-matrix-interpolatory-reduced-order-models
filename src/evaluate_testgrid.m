function ev = evaluate_testgrid(cfg, prom, stride, idx)
%EVALUATE_TESTGRID Posterior-mean pROM vs. full-order reference on the 51x51 grid.
%   Error measure of the paper's error maps: mean
%   relative transfer-function error over 1..500 Hz per grid point. In
%   addition, per objective configuration: relative error within the
%   objective band and ln R of pROM and FOM. stride > 1 evaluates every
%   stride-th grid point only (smoke test); idx (optional) evaluates the
%   given grid points only and overrides stride.
if nargin < 3
    stride = 1;
end
Te = load_testgrid(cfg);
if nargin < 4 || isempty(idx)
    idx = 1:stride:size(Te.S, 1);
end
f_eval = 1:500;
fi = round(f_eval);
assert(all(abs(Te.f(fi) - f_eval) < 1e-6), 'Unexpected frequency grid in reference data.');
yTe = Te.y(fi, idx);

nObj = numel(cfg.objectives);
n = numel(idx);
y = zeros(numel(f_eval), n);
chunk = 51;
for c0 = 1:chunk:n
    cols = c0:min(c0 + chunk - 1, n);
    [K, D, M, F, G] = prom_predict(prom, Te.S(idx(cols), :), false);
    yc = zeros(numel(f_eval), numel(cols));
    parfor j = 1:numel(cols)
        Kj = K(:,:,j); Dj = D(:,:,j); Mj = M(:,:,j); Fj = F(:,:,j); Gj = G(:,:,j);
        yj = zeros(numel(f_eval), 1);
        for k = 1:numel(f_eval)
            s = 2 * pi * 1i * f_eval(k);
            yj(k) = Gj * ((Kj + s * Dj + s^2 * Mj) \ Fj);
        end
        yc(:, j) = yj;
    end
    y(:, cols) = yc;
end

relErr = abs(yTe - y) ./ abs(yTe);
ev.S = Te.S(idx, :);
ev.meanRelErr = mean(relErr, 1)';
ev.objective_ids = {cfg.objectives.id};
ev.bandRelErr = zeros(n, nObj);
ev.lnR_prom = zeros(n, nObj);
ev.lnR_true = zeros(n, nObj);
for o = 1:nObj
    band = ismember(f_eval, round(cfg.objectives(o).f_opt));
    pk = cfg.objectives(o).p;
    ev.bandRelErr(:, o) = mean(relErr(band, :), 1)';
    ev.lnR_prom(:, o) = log(sum(abs(y(band, :)).^pk, 1).^(1/pk))';
    ev.lnR_true(:, o) = log(sum(abs(yTe(band, :)).^pk, 1).^(1/pk))';
end
ev.stride = stride;
end
