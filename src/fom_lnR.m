function lnR = fom_lnR(K, D, M, F, G, f_opt, p)
%FOM_LNR Log of the objective R = (sum_i |y(s_i)|^p)^(1/p) (paper Eq. 2) for one FOM.
%   Same definition as EvaluateCost/EvaluateCostAndGradient (fmincon minimizes ln R).
absy = zeros(size(f_opt));
parfor k = 1:numel(f_opt)
    s = 2 * pi * 1i * f_opt(k);
    x = (K + s * D + s^2 * M) \ F;
    absy(k) = abs(G * x);
end
lnR = log(sum(absy.^p)^(1/p));
end
