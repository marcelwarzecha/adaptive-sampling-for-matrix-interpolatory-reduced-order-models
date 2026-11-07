function [K, D, M, F, G] = prom_predict(prom, Z, use_ts)
%PROM_PREDICT Reduced operators at the points Z (rows, physical units).
%   use_ts = true uses the Thompson-sampled coefficients, false the
%   posterior mean. K, D and M are returned as full symmetric matrices
%   (the regression models only cover the upper triangle).
if use_ts
    C = prom.C_ts;
else
    C = prom.C_mean;
end
[K, D, M, F, G] = EvaluateCompiledModels(C, transformToUnitSpace(Z, prom.ModelParams.Bounds));
end
