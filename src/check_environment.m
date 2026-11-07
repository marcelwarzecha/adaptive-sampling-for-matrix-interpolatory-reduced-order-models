function res = check_environment(cfg)
%CHECK_ENVIRONMENT Toolboxes, parallel pool, required files and FOM assembly time.
res.matlab = version;
res.toolboxes = struct( ...
    'Statistics', license('test', 'Statistics_Toolbox'), ...
    'Optimization', license('test', 'Optimization_Toolbox'), ...
    'Parallel', license('test', 'Distrib_Computing_Toolbox'));
res.n_workers = 0;
if res.toolboxes.Parallel && ~isempty(ver('parallel'))
    pool = gcp('nocreate');
    if ~isempty(pool)
        res.n_workers = pool.NumWorkers;
    end
end
res.testgrid_found = isfile(cfg.paths.testgrid);
res.functions_found = struct( ...
    'KelvinCell', exist('KelvinCell', 'file') == 2, ...
    'FemModel', exist('FemModel', 'file') == 2, ...
    'SB1_Estimate', exist('SB1_Estimate', 'file') == 2, ...
    'polyfitn', exist('polyfitn', 'file') == 2);

cfgKC = production_config('KelvinCell');
t = tic;
[K, ~, ~, F] = assemble_fom(cfgKC, mean(cfgKC.model.Bounds, 1));
res.kelvin_cell_assembly_time = toc(t);
res.kelvin_cell_n_dof = size(K, 1);
res.kelvin_cell_sum_F = full(sum(F));
cfgB = production_config('Beam');
[Kb] = assemble_fom(cfgB, mean(cfgB.model.Bounds, 1));
res.beam_n_dof = size(Kb, 1);

% The Parallel Computing Toolbox is reported but not required (serial mode, run_all 'workers', 0).
res.pass = res.toolboxes.Statistics && res.toolboxes.Optimization && res.testgrid_found ...
    && all(cell2mat(struct2cell(res.functions_found))) && res.kelvin_cell_n_dof == 10728 ...
    && res.beam_n_dof == 2400;
disp(res);
if ~res.pass
    error('check_environment:failed', 'Environment check failed, see the output above.');
end
end
