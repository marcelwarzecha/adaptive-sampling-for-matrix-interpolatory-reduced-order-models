function Phi = MonomialFeatures(Z, terms, k)
%MONOMIALFEATURES Monomials of the points Z, or their derivative with respect to one parameter.
%   Phi = MONOMIALFEATURES(Z, terms) returns Phi(j,t) = prod_l Z(j,l)^terms(t,l)
%   (as x2fx(Z, terms)). Phi = MONOMIALFEATURES(Z, terms, k) returns the
%   derivative with respect to Z(:,k), i.e. the design matrix of the
%   differentiated polynomials.
%
%   Inputs:
%       Z     - query points (n x d)
%       terms - exponents (numTerms x d)
%       k     - (optional) parameter to differentiate with respect to

if nargin < 3
    k = 0;
end
Phi = ones(size(Z, 1), size(terms, 1));
for l = 1:size(terms, 2)
    e = terms(:, l).';
    if l == k
        % d/dz z^e = e z^(e-1); the clamp keeps 0^(-1) out of terms with e = 0
        Phi = Phi .* (e .* Z(:, l).^max(e - 1, 0));
    else
        Phi = Phi .* Z(:, l).^e;
    end
end
end
