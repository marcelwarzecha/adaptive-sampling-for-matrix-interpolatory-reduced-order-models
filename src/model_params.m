function ModelParams = model_params(cfg)
%MODEL_PARAMS ModelParams struct in the format expected by the core routines.
ModelParams.Name = cfg.model.name;
ModelParams.numParams = cfg.model.numParams;
ModelParams.Bounds = cfg.model.Bounds;
ModelParams.lb = cfg.model.lb;
ModelParams.ub = cfg.model.ub;
ModelParams.parallel_training = cfg.reg.parallel_training;
ModelParams.sbr = cfg.reg.sbr;
if strcmp(cfg.model.name, 'Beam')
    ModelParams.L = cfg.model.L;
end
end
