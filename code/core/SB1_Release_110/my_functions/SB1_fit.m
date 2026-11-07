function [Mdl] = SB1_fit(X, Y, options)
%SB1_FIT Sparse Bayesian regression (Tipping, SB1) of one normalized matrix entry.
%   Mdl = SB1_FIT(X, Y, options) fits the monomial basis options.PHI to the
%   entry values Y at the samples X. The same settings hold for every entry
%   and every model (options.sbr, set in cfg.reg.sbr):
%     - initialization as in Tipping's SB1_ExampleRegress.m: alpha_0 = 1/N^2
%       for all basis functions, noise standard deviation 10 % of std(Y)
%       (options.sbr.noise_frac), i.e. beta_0 = 1/(noise_frac*std(Y))^2;
%     - re-estimation in SB1_Estimate with options.sbr.maxits passes.
%   An entry that is constant over the samples is not regressed: it keeps
%   its value through the intercept, without posterior covariance (constant).
    Mdl.PHI = options.PHI;
    N = size(Mdl.PHI, 1);

    Y_std = std(Y);
    Mdl.diagnostics.Y_stats = struct('std', Y_std, 'range', range(Y), 'mean', mean(Y));

    if Y_std == 0
        Mdl.constant = true;
        Mdl.used = find(all(options.terms == 0, 2));
        Mdl.weights = Y(1);
        Mdl.ML = NaN; Mdl.alpha = []; Mdl.beta = Inf; Mdl.G = []; Mdl.Ui = [];
    else
        Mdl.constant = false;
        initAlpha = (1/N)^2;
        initBeta = 1/(options.sbr.noise_frac*Y_std)^2;
        [Mdl.weights,Mdl.used,Mdl.ML,Mdl.alpha,Mdl.beta,Mdl.G, Mdl.Ui] = ...
            SB1_Estimate(Mdl.PHI,Y,initAlpha,initBeta,options.sbr.maxits,options);
    end

    Mdl.ModelTerms = options.terms(Mdl.used,:);
    Mdl.Coefficients = Mdl.weights;

    % Calculate fit quality metrics
    y_pred = Mdl.PHI(:,Mdl.used)*Mdl.weights;
    Mdl.diagnostics.R2 = 1 - sum((Y - y_pred).^2)/sum((Y - mean(Y)).^2);
    Mdl.diagnostics.RMSE = sqrt(mean((Y - y_pred).^2));

    if options.thompson_sampling && ~defer_thompson(options)
        scale_factor = 1e0;
        Mdl = SB1_thompson_draw(Mdl, scale_factor);
    end
end

function tf = defer_thompson(options)
% options.defer_thompson = true: fit only; the caller draws later (SysMdls, parallel training)
tf = isfield(options, 'defer_thompson') && options.defer_thompson;
end
