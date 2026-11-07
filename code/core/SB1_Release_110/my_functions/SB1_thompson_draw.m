function Mdl = SB1_thompson_draw(Mdl, scale_factor)
%SB1_THOMPSON_DRAW One Thompson realization of the coefficients, beta_TS ~ N(weights, Sigma).
%   If SB1_Estimate returns no posterior covariance (all basis functions
%   pruned, or the Cholesky factorization failed in its first iteration),
%   no draw is possible: the posterior mean is used and ts_skipped is set.
%   Called by SB1_fit, or by SysMdls after a parallel fit (defer_thompson),
%   which keeps the order of the draws and thus the random numbers.
%   Entries that are constant over the samples (SB1_fit, constant) have no
%   uncertainty: they keep their value and draw no random numbers.
if isfield(Mdl, 'constant') && Mdl.constant
    Mdl.ts_skipped = false;
    Mdl.Sigma = zeros(numel(Mdl.weights));
    Mdl.TSCoefficients = Mdl.weights;
    return
end
Mdl.ts_skipped = isempty(Mdl.weights) || isempty(Mdl.Ui);
if Mdl.ts_skipped
    Mdl.Sigma = zeros(numel(Mdl.weights));
    Mdl.TSCoefficients = Mdl.weights;
else
    Mdl.Sigma = scale_factor*(Mdl.Ui * Mdl.Ui');
    Mdl.TSCoefficients = Mdl.weights + mvnrnd(zeros(size(Mdl.weights)), Mdl.Sigma, 1)';
end
end
