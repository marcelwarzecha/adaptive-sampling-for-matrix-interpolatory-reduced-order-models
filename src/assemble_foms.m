function [K, D, M, F, G] = assemble_foms(cfg, S)
%ASSEMBLE_FOMS Full-order models at all rows of S (parallel), returned as cell arrays.
n = size(S, 1);
K = cell(1, n); D = cell(1, n); M = cell(1, n); F = cell(1, n); G = cell(1, n);
parfor i = 1:n
    [K{i}, D{i}, M{i}, F{i}, G{i}] = assemble_fom(cfg, S(i,:));
end
end
