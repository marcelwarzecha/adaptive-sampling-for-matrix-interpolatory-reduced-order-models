function [evMin, evMax] = prom_min_eig(prom, Z, use_ts)
%PROM_MIN_EIG Smallest and largest eigenvalues of the predicted reduced K, D and M at the points Z.
%   evMin and evMax are n x 3 with columns [K, D, M]. A predicted operator
%   is indefinite if its smallest eigenvalue is negative; judge this
%   relative to the largest one (evMin ./ evMax), since the reduced mass
%   matrix spans many orders of magnitude (rotational inertia).
[K, D, M] = prom_predict(prom, Z, use_ts);
n = size(Z, 1);
evMin = zeros(n, 3);
evMax = zeros(n, 3);
for j = 1:n
    A = {K(:,:,j), D(:,:,j), M(:,:,j)};
    for m = 1:3
        e = eig((A{m} + A{m}.') / 2);
        evMin(j, m) = min(e);
        evMax(j, m) = max(e);
    end
end
end
