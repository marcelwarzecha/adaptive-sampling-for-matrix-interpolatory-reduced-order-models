function res = run_reference_grid(cfg, n)
%RUN_REFERENCE_GRID Full-order objective ln R on an n x n grid for all objectives of the model.
%   Used for the beam (the Kelvin cell reference is data/FFD_51x51_1000Hz.mat).
g1 = linspace(cfg.model.lb(1), cfg.model.ub(1), n);
g2 = linspace(cfg.model.lb(2), cfg.model.ub(2), n);
[P1, P2] = meshgrid(g1, g2);
Z = [P1(:), P2(:)];
nObj = numel(cfg.objectives);
lnR = zeros(size(Z, 1), nObj);
objectives = cfg.objectives;
t = tic;
parfor i = 1:size(Z, 1)
    [K, D, M, F, G] = assemble_fom(cfg, Z(i,:));
    row = zeros(1, nObj);
    for o = 1:nObj
        absy = zeros(size(objectives(o).f_opt));
        for k = 1:numel(absy)
            s = 2 * pi * 1i * objectives(o).f_opt(k);
            absy(k) = abs(G * ((K + s * D + s^2 * M) \ F));
        end
        row(o) = log(sum(absy.^objectives(o).p)^(1 / objectives(o).p));
    end
    lnR(i, :) = row;
end
res.time.total = toc(t);
res.Z = Z;
res.lnR = lnR;
res.objective_ids = {cfg.objectives.id};
[~, iopt] = min(lnR, [], 1);
res.p_star = Z(iopt, :);
end
