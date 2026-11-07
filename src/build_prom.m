function prom = build_prom(cfg, S, S_unit, Kr, Dr, Mr, Fr, Gr)
%BUILD_PROM Train the regression models (ModelTraining) and bundle everything needed to evaluate the pROM.
%   Note: for SBR, ModelTraining draws one Thompson realization per entry
%   (mvnrnd inside SB1_fit), i.e. it consumes random numbers.
ModelParams = model_params(cfg);
[Mdl_K, Mdl_D, Mdl_M, Mdl_F, Mdl_G, Y_K_MinMax, Y_D_MinMax, Y_M_MinMax, ...
    Y_F_MinMax, Y_G_MinMax, trainingOptions] = ModelTraining(S, S_unit, Kr, Dr, Mr, Fr, Gr, ...
    cfg.reg.method, cfg.model.damping, ModelParams);

prom.S = S;
prom.S_unit = S_unit;
prom.Mdl_K = Mdl_K;
prom.Mdl_D = Mdl_D;
prom.Mdl_M = Mdl_M;
prom.Mdl_F = Mdl_F;
prom.Mdl_G = Mdl_G;
prom.Y_MinMax = {Y_K_MinMax, Y_D_MinMax, Y_M_MinMax, Y_F_MinMax, Y_G_MinMax};
prom.trainingOptions = trainingOptions;
prom.ModelParams = ModelParams;
prom.method = cfg.reg.method;
prom.r = size(Kr, 1);
prom.order = trainingOptions.order;

% Dense coefficient matrices for fast evaluation (part of the training time)
compile = @(use_ts) CompileModels(Mdl_K, Mdl_D, Mdl_M, Mdl_F, Mdl_G, prom.Y_MinMax, ...
    trainingOptions.terms, ModelParams.Bounds, use_ts);
prom.C_mean = compile(false);
prom.C_ts = [];
if trainingOptions.thompson_sampling
    prom.C_ts = compile(true);
end
end
