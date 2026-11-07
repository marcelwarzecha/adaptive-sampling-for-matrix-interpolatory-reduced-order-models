function lnR = prom_lnR(prom, Z, use_ts, f_opt, p)
%PROM_LNR ln R of the pROM at the points Z (rows), posterior mean or Thompson-sampled.
%   Points are evaluated in parallel (parfor); the result does not depend on the pool.
[K, D, M, F, G] = prom_predict(prom, Z, use_ts);
n = size(Z, 1);
lnR = zeros(n, 1);
parfor j = 1:n
    Kj = K(:,:,j); Dj = D(:,:,j); Mj = M(:,:,j); Fj = F(:,:,j); Gj = G(:,:,j);
    absy = zeros(size(f_opt));
    for k = 1:numel(f_opt)
        s = 2 * pi * 1i * f_opt(k);
        absy(k) = abs(Gj * ((Kj + s * Dj + s^2 * Mj) \ Fj));
    end
    lnR(j) = log(sum(absy.^p)^(1/p));
end
end
