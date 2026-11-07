function [K, D, M, F, G] = EvaluateCompiledModels(C, Z_unit)
%EVALUATECOMPILEDMODELS Reduced operators of a compiled pROM (CompileModels) at unit-space points.
%   [K, D, M, F, G] = EVALUATECOMPILEDMODELS(C, Z_unit) returns the full
%   symmetric K, D and M (r x r x n), F (r x 1 x n) and G (1 x r x n) at
%   the n rows of Z_unit. Only the upper triangles of K, D and M are
%   regressed; the lower triangles are obtained by symmetry.

Phi = MonomialFeatures(Z_unit, C.terms);
n = size(Z_unit, 1);
K = symmetricOperators(C, entries(C.K, Phi));
D = symmetricOperators(C, entries(C.D, Phi));
M = symmetricOperators(C, entries(C.M, Phi));
F = reshape(entries(C.F, Phi).', C.r, 1, n);
G = reshape(entries(C.G, Phi).', 1, C.r, n);
end

function Y = entries(op, Phi)
% predicted entries (n x numEntries), normalization of SysMdls reverted
Y = op.ymin + (Phi * op.W) .* op.yrng;
end

function A = symmetricOperators(C, Y)
n = size(Y, 1);
A = zeros(C.r * C.r, n);
A(C.upper(:), :) = Y.';
A = reshape(A, C.r, C.r, n);
for j = 1:n
    A(:,:,j) = A(:,:,j) + tril(A(:,:,j).', -1);
end
end
