function [dK, dD, dM] = EvaluateCompiledDerivatives(C, z_unit)
%EVALUATECOMPILEDDERIVATIVES Parameter derivatives of the reduced K, D and M of a compiled pROM.
%   [dK, dD, dM] = EVALUATECOMPILEDDERIVATIVES(C, z_unit) returns 1 x d
%   cell arrays; dK{k} is the full symmetric derivative of K with respect
%   to the k-th physical parameter at the unit-space point z_unit (1 x d).
%   The basis is differentiated once for all entries (MonomialFeatures),
%   for any number of parameters.

d = size(C.terms, 2);
dx = C.Bounds(2, :) - C.Bounds(1, :);
dK = cell(1, d);
dD = cell(1, d);
dM = cell(1, d);
for k = 1:d
    dPhi = MonomialFeatures(z_unit, C.terms, k);
    % chain rule through both normalizations: dy/dp_k = dy_n/dz_k * (y_max - y_min) / (ub_k - lb_k)
    dK{k} = symmetricMatrix(C, (dPhi * C.K.W) .* C.K.yrng / dx(k));
    dD{k} = symmetricMatrix(C, (dPhi * C.D.W) .* C.D.yrng / dx(k));
    dM{k} = symmetricMatrix(C, (dPhi * C.M.W) .* C.M.yrng / dx(k));
end
end

function A = symmetricMatrix(C, y)
A = zeros(C.r);
A(C.upper) = y;
A = A + tril(A.', -1);
end
