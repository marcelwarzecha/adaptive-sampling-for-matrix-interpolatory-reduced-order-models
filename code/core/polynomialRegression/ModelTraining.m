function [Mdl_K, Mdl_D, Mdl_M, Mdl_F, Mdl_G,  Y_K_MinMax, Y_D_MinMax, Y_M_MinMax, ...
    Y_F_MinMax, Y_G_MinMax, options] = ModelTraining(S, S_unit, Kr, Dr, Mr, Fr, Gr, method, damping, ModelParams)
%MODELTRAINING Sparse Bayesian regression models of all entries of the reduced operators.
%   Every entry of the upper triangles of Kr, Dr and Mr and every entry of
%   Fr and Gr is normalized to [0, 1] and regressed over the unit-space
%   samples S_unit with sparse Bayesian regression (SBR, Tipping), followed
%   by one Thompson draw of the coefficients (SysMdls).
%
%   Basis: all monomials of total degree <= order, where order is the
%   largest degree whose number of terms, nchoosek(order+d, d), does not
%   exceed (number of samples - 1), capped at ModelParams.sbr.max_degree.
%
%   Inputs:
%       S, S_unit   - samples in physical and unit space (numSamples x d)
%       Kr, Dr, Mr  - reduced stiffness, damping and mass matrices (r x r x numSamples)
%       Fr, Gr      - reduced input (r x 1 x numSamples) and output (1 x r x numSamples)
%       method      - 'SBR'
%       damping     - true/false
%       ModelParams - model and SBR settings (model_params)
%
%   Outputs:
%       Mdl_K, ..., Mdl_G      - entry-wise models (cell arrays)
%       Y_K_MinMax, ...        - normalization of every entry
%       options                - basis (terms), degree (order) and Thompson-sampling flag

    assert(strcmp(method, 'SBR'), 'ModelTraining:method', 'Only method ''SBR'' is supported.');
    numParams = size(S,2);

    % Number of data points minus one
    N = size(S,1)-1;

    order = 0;
    while order < ModelParams.sbr.max_degree && nchoosek(order+1+numParams, numParams) <= N
        order = order + 1;
    end
    terms = fullfact(repmat(order+1, 1, numParams)) - 1;
    terms = terms(sum(terms, 2) <= order, :);

    normalize = true;
    options = struct('order', order, 'terms', terms, 'thompson_sampling', true);
    [Mdl_K, Mdl_D, Mdl_M, Mdl_F, Mdl_G, Y_K_MinMax, Y_D_MinMax, Y_M_MinMax, ...
        Y_F_MinMax, Y_G_MinMax] = SysMdls(S_unit, Kr, Dr, Mr, Fr, Gr, method, options, normalize, damping, ModelParams);
end
