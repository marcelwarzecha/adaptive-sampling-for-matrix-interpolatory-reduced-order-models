function [Mdl_K, Mdl_D, Mdl_M, Mdl_F, Mdl_G, Y_K_MinMax, Y_D_MinMax, Y_M_MinMax, ...
    Y_F_MinMax, Y_G_MinMax] = SysMdls(X, K, D, M, F, G, method, options, normalize, damping, ModelParams) %#ok<INUSD>
%SYSMDLS Entry-wise sparse Bayesian regression models of the reduced operators.
%
%   Inputs:
%       X           - samples in unit space (numSamples x d)
%       K, D, M     - reduced stiffness, damping and mass matrices (r x r x numSamples)
%       F, G        - reduced input (r x 1 x numSamples) and output (1 x r x numSamples)
%       method      - 'SBR'
%       options     - basis (options.terms) and Thompson-sampling flag (ModelTraining)
%       normalize   - true: every entry is mapped to [0, 1] before the regression
%       damping     - true/false
%       ModelParams - SBR settings (ModelParams.sbr) and parallel training flag
%
%   Outputs:
%       Mdl_K, ..., Mdl_G - one model per entry of the upper triangles of K, D
%                           and M and per entry of F and G (cell arrays)
%       Y_K_MinMax, ...   - normalization of every entry (2 x numEntries)

assert(strcmp(method, 'SBR') && normalize, 'SysMdls:method', ...
    'Only normalized sparse Bayesian regression (''SBR'') is supported.');

%% reorder entries

Y_K = symmMatrix2Vector(K);
Y_D = symmMatrix2Vector(D);
Y_M = symmMatrix2Vector(M);
Y_F = InputMatrix2Vector(F);
Y_G = OutputMatrix2Vector(G);

%% normalize

[Y_K, Y_K_MinMax] = normalizeEntries(Y_K);
[Y_D, Y_D_MinMax] = normalizeEntries(Y_D);
[Y_M, Y_M_MinMax] = normalizeEntries(Y_M);
[Y_F, Y_F_MinMax] = normalizeEntries(Y_F);
[Y_G, Y_G_MinMax] = normalizeEntries(Y_G);

%% Sparse Bayesian regression of every entry
assert(isfield(ModelParams, 'sbr'), 'SysMdls:sbr', ...
    'SBR settings missing: set ModelParams.sbr (cfg.reg.sbr, see production_config).');
options.sbr = ModelParams.sbr;
% Inititalize Sizes
n_K = size(Y_K, 2);
n_D = size(Y_D, 2);
n_M = size(Y_M, 2);
n_F = size(Y_F, 2);
n_G = size(Y_G, 2);

% Fit Sparse Bayesian models for each matrix entry
Mdl_K = cell(1, n_K);
Mdl_D = cell(1, n_D);
Mdl_M = cell(1, n_M);
Mdl_F = cell(1, n_F);
Mdl_G = cell(1, n_G);

% Compute Design Matrix
options.PHI = x2fx(X,options.terms);

if isfield(ModelParams, 'parallel_training') && ModelParams.parallel_training
    % The fits use no random numbers and run in parallel. The
    % Thompson draws follow serially in the order of the serial
    % loop below (K_i, D_i, M_i), so the random numbers are the same.
    fitOptions = options;
    fitOptions.defer_thompson = true;
    parfor i = 1:n_K
        Mdl_K{i} = rmfield(SB1_fit(X, Y_K(:, i), fitOptions), 'PHI');  % PHI: shared, not sent back
        Mdl_D{i} = rmfield(SB1_fit(X, Y_D(:, i), fitOptions), 'PHI');
        Mdl_M{i} = rmfield(SB1_fit(X, Y_M(:, i), fitOptions), 'PHI');
    end
    if options.thompson_sampling
        scale_factor = 1e0;                     % as in SB1_fit
        for i = 1:n_K
            Mdl_K{i} = SB1_thompson_draw(Mdl_K{i}, scale_factor);
            Mdl_D{i} = SB1_thompson_draw(Mdl_D{i}, scale_factor);
            Mdl_M{i} = SB1_thompson_draw(Mdl_M{i}, scale_factor);
        end
    end
else
    for i = 1:n_K
        % Stiffness Matrix
        Mdl_K{i} = SB1_fit(X, Y_K(:, i), options);

        % Damping Matrix            
        Mdl_D{i} = SB1_fit(X, Y_D(:, i), options);

        % Mass Matrix            
        Mdl_M{i} = SB1_fit(X, Y_M(:, i), options);
    end
end

% serial loops: an SB1_fit with Thompson draw must not run on a worker (unseeded random numbers)
% Constant entries get an intercept-only polyfitn model (degree 0): a
% fit of the full degree is rank-deficient when the samples are close
% together (e.g. on a bound) and only produces warnings, since constant entries are
% reconstructed exactly from the min/max normalization (range 0) anyway.
for i = 1:n_F
    epsilonF    = std(Y_F(:,i)) * 10/100; % Initial guess of 10% noise-to-signal
    if epsilonF == 0
         Mdl_F{i} = polyfitn(X, Y_F(:,i), 0);            % constant model, intercept only
    else
        Mdl_F{i} = SB1_fit(X, Y_F(:, i), options);
    end
end

for i = 1:n_G
    epsilonG    = std(Y_G(:,i)) * 10/100; % Initial guess of 10% noise-to-signal
    if epsilonG == 0
        Mdl_G{i} = polyfitn(X, Y_G(:,i), 0);            % constant model, intercept only
    else
        Mdl_G{i} = SB1_fit(X, Y_G(:, i), options);
    end
end
end % function

function [Y, Y_MinMax] = normalizeEntries(Y)
% Map every column of Y to [0, 1] between its entries of smallest and
% largest magnitude (first occurrence on ties); constant columns are set
% to 1 to prevent dividing by zero. Y_MinMax holds these two entries.
[~, indMin] = min(abs(Y), [], 1);
[~, indMax] = max(abs(Y), [], 1);
cols = 1:size(Y, 2);
Y_MinMax = [Y(sub2ind(size(Y), indMin, cols)); Y(sub2ind(size(Y), indMax, cols))];
constant = Y_MinMax(2, :) == Y_MinMax(1, :);
Y = (Y - Y_MinMax(1, :)) ./ (Y_MinMax(2, :) - Y_MinMax(1, :));
Y(:, constant) = 1;
end
