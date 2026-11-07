function [logcost, grad] = EvaluateCostAndGradient(C, z, f_opt, p, solver)
%EVALUATECOSTANDGRADIENT Objective ln R(p) and its adjoint gradient on a compiled pROM (fmincon handle).
%   [logcost, grad] = EVALUATECOSTANDGRADIENT(C, z, f_opt, p, solver)
%   computes ln R and d(ln R)/dp (paper Section 4, Algorithm 2) for the pROM
%   compiled by CompileModels:
%     - operators and their derivatives come from one matrix product each
%       (EvaluateCompiledModels, EvaluateCompiledDerivatives);
%     - state and adjoint share one factorization per frequency: K_dyn is
%       complex symmetric, so eta_i = K_dyn,i.' \ (lambda_i G.') = lambda_i (K_dyn,i \ G.');
%     - the gradient contraction over all frequencies is a few matrix
%       products, without storing dK_dyn/dp for every frequency;
%     - no parfor: the reduced systems are small, and the frequency sums do
%       not depend on the order in which workers return.
%   F and G are parameter-independent (paper Section 5).
%
%   Inputs:
%       C      - compiled pROM (CompileModels)
%       z      - parameter point in physical units (1 x d)
%       f_opt  - frequencies of the objective [Hz]
%       p      - order of the norm
%       solver - 'loop' (default): mldivide frequency by frequency;
%                'pages': all K_dyn at once with pagemldivide

if nargin < 5
    solver = 'loop';
end

z_unit = transformToUnitSpace(z, C.Bounds);
[K, D, M, F, G] = EvaluateCompiledModels(C, z_unit);

% State x_i = K_dyn,i \ F and w_i = K_dyn,i \ G.' for all frequencies (columns)
s = 2 * pi * 1i * f_opt(:).';
n_f = numel(s);
B = [F, G.'];
switch solver
    case 'loop'
        X = complex(zeros(C.r, n_f));
        W = complex(zeros(C.r, n_f));
        for i = 1:n_f
            XW = (K + s(i) * D + s(i)^2 * M) \ B;
            X(:, i) = XW(:, 1);
            W(:, i) = XW(:, 2);
        end
    case 'pages'
        K_dyn = K + reshape(s, 1, 1, n_f) .* D + reshape(s.^2, 1, 1, n_f) .* M;
        XW = pagemldivide(K_dyn, repmat(B, 1, 1, n_f));
        X = reshape(XW(:, 1, :), C.r, n_f);
        W = reshape(XW(:, 2, :), C.r, n_f);
    otherwise
        error('EvaluateCostAndGradient:solver', 'Unknown solver "%s".', solver);
end

% Cost
y = G * X;
absy = abs(y);
y_sum = sum(absy.^p);
cost = y_sum^(1/p);
logcost = log(cost);

% Gradient
if nargout > 1
    [dK, dD, dM] = EvaluateCompiledDerivatives(C, z_unit);
    lambda = y_sum^(1/p - 1) * absy.^(p - 1) * 1/2 .* conj(y) ./ absy;   % Wirtinger derivative
    E = W .* lambda;                                                     % adjoint solutions eta_i
    d = numel(dK);
    dRdp = complex(zeros(1, d));
    for k = 1:d
        % sum_i eta_i.' * (-(dK + s_i dD + s_i^2 dM) * x_i)
        dRdp(k) = -sum(sum(E .* (dK{k} * X), 1) + s .* sum(E .* (dD{k} * X), 1) ...
            + s.^2 .* sum(E .* (dM{k} * X), 1));
    end
    grad = (1/cost) * 2 * real(dRdp);
end
end
