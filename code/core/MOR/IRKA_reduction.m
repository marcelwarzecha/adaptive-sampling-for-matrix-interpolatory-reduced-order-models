function [Kr, Dr, Mr, Fr, Gr, Vr] = IRKA_reduction(K, D, M, F, G, IRKA_f0,IRKA_tol,IRKA_maxiter)
%IRKA_REDUCTION 
%   reduces the entire sample set 
%   INPUT:
%   - K, D, M, F, G: Cell Structures containing the FOM Matrices nDOF x
%   nDOF x nSamples
%   - IRKA_f0, IRKA_tol, IRKA_maxiter: IRKA parameters
%
%   OUTPUT:
%   - Kr, Dr, Mr, Fr, Gr, Vr: Cell structure containing ROM matrices 

% a single sample may be passed as plain matrices
if ~iscell(K)
    K = {K}; D = {D}; M = {M}; F = {F}; G = {G};
end
numSamples = numel(K);
% convert input sample frequencies to rad/s
s_0 = 1i * 2 * pi * IRKA_f0;

% the samples are independent: parallel over samples, but a single sample
% stays on the client (no transfer of the FOM to a worker)
Kr = cell(1, numSamples); Dr = cell(1, numSamples); Mr = cell(1, numSamples);
Fr = cell(1, numSamples); Gr = cell(1, numSamples); Vr = cell(1, numSamples);
if numSamples > 1
    maxWorkers = Inf;
else
    maxWorkers = 0;
end
parfor (i = 1:numSamples, maxWorkers)
    fprintf("%u/%u ... \t", i, numSamples)
    % obtain ROM + reduced basis
    [Kr{i}, Dr{i}, Mr{i}, Fr{i}, Gr{i}, Vr{i}] = SO_IRKA_SISO(K{i}, D{i}, M{i}, F{i}, G{i}, ...
        s_0, IRKA_tol, IRKA_maxiter);
end
end
