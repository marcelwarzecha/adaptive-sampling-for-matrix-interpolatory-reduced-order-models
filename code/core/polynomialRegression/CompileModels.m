function C = CompileModels(Mdl_K, Mdl_D, Mdl_M, Mdl_F, Mdl_G, Y_MinMax, terms, Bounds, use_ts)
%COMPILEMODELS Collect the entry-wise SBR models of a pROM in dense coefficient matrices.
%   C = COMPILEMODELS(Mdl_K, Mdl_D, Mdl_M, Mdl_F, Mdl_G, Y_MinMax, terms, Bounds, use_ts)
%   stores for every operator a matrix W (numTerms x numEntries) such that
%   the normalized entries at the unit-space points Z are
%   MonomialFeatures(Z, terms) * W. Predicting all entries (and their
%   derivatives) then takes one matrix product instead of one loop
%   iteration per entry (EvaluateCompiledModels, EvaluateCompiledDerivatives).
%
%   Inputs:
%       Mdl_K, ..., Mdl_G - models from ModelTraining (method 'SBR')
%       Y_MinMax          - {Y_K_MinMax, Y_D_MinMax, Y_M_MinMax, Y_F_MinMax, Y_G_MinMax}
%       terms             - monomial exponents of the SBR design matrix (trainingOptions.terms)
%       Bounds            - parameter bounds of the unit space (ModelParams.Bounds)
%       use_ts            - true: Thompson-sampled coefficients for K, D and M
%
%   Coefficients: TSCoefficients (use_ts) or weights (posterior mean) for
%   K, D and M, and the posterior mean (Coefficients) for F and G. An entry
%   whose basis functions were all pruned gets a zero column (normalized
%   value 0).

if use_ts
    kdmField = 'TSCoefficients';
else
    kdmField = 'weights';
end

C.terms = terms;
C.Bounds = Bounds;
C.use_ts = use_ts;
C.r = round(-0.5 + sqrt(0.25 + 2 * numel(Mdl_K)));
C.upper = triu(true(C.r));          % storage order of symmMatrix2Vector
assert(numel(Mdl_F) == C.r && numel(Mdl_G) == C.r, 'CompileModels:siso', ...
    'Only single-input single-output systems are supported.');

C.K = compileOperator(Mdl_K, Y_MinMax{1}, terms, kdmField);
C.D = compileOperator(Mdl_D, Y_MinMax{2}, terms, kdmField);
C.M = compileOperator(Mdl_M, Y_MinMax{3}, terms, kdmField);
C.F = compileOperator(Mdl_F, Y_MinMax{4}, terms, 'Coefficients');
C.G = compileOperator(Mdl_G, Y_MinMax{5}, terms, 'Coefficients');
end

function op = compileOperator(Mdl, Y_MinMax, terms, coeffField)
n = numel(Mdl);
W = zeros(size(terms, 1), n);
for e = 1:n
    m = Mdl{e};
    if isfield(m, 'used')
        rows = m.used;                  % SBR model: columns of the design matrix
    else
        % polyfitn model (constant entries): locate its terms in the SBR basis
        [found, rows] = ismember(m.ModelTerms, terms, 'rows');
        assert(all(found), 'CompileModels:terms', ...
            'Entry %d uses a monomial that is not in the SBR basis.', e);
    end
    W(rows, e) = m.(coeffField);
end
op.W = W;
op.ymin = Y_MinMax(1, :);
op.yrng = Y_MinMax(2, :) - Y_MinMax(1, :);
end
