function collect_results(resRoot)
%COLLECT_RESULTS Summarize all result files below resRoot into CSV tables in resRoot.
%   The smoke-test results (resRoot/smoke) are only included if resRoot is that folder.
%   runs_adaptive.csv      one row per adaptive run
%   iterations_adaptive.csv one row per outer iteration of every adaptive run
%   nonadaptive.csv        one row per (design, objective)
%   fd.csv                 one row per finite-difference optimization
%   failed.txt             jobs that ended with an error
if ~isfolder(resRoot)
    return
end
files = dir(fullfile(resRoot, '**', '*.mat'));
files = files(~endsWith({files.name}, '.tmp.mat'));
files = files(~startsWith({files.folder}, fullfile(resRoot, 'smoke')) | endsWith(resRoot, 'smoke'));
runs = {}; iters = {}; nonad = {}; fd = {};
for i = 1:numel(files)
    try
        L = load(fullfile(files(i).folder, files(i).name), 'res');
        res = L.res;
        m = res.meta;
        switch m.job_fn
            case 'run_adaptive'
                row = adaptive_row(res);
                rows = iteration_rows(res);
                runs{end+1, 1} = row; %#ok<AGROW>
                iters = [iters; rows]; %#ok<AGROW>
            case 'run_nonadaptive'
                nonad = [nonad; nonadaptive_rows(res)]; %#ok<AGROW>
            case 'run_fd_comparison'
                fd{end+1, 1} = fd_row(res); %#ok<AGROW>
        end
    catch ME
        warning('collect_results:skip', 'Skipping %s: %s', files(i).name, ME.message);
    end
end
write_rows(runs, fullfile(resRoot, 'runs_adaptive.csv'));
write_rows(iters, fullfile(resRoot, 'iterations_adaptive.csv'));
write_rows(nonad, fullfile(resRoot, 'nonadaptive.csv'));
write_rows(fd, fullfile(resRoot, 'fd.csv'));

errs = dir(fullfile(resRoot, '**', '*.error.txt'));
fid = fopen(fullfile(resRoot, 'failed.txt'), 'w');
for i = 1:numel(errs)
    fprintf(fid, '%s\n', fullfile(errs(i).folder, errs(i).name));
end
fclose(fid);
fprintf('collect_results: %d adaptive, %d non-adaptive, %d FD results; %d failed jobs (%s)\n', ...
    numel(runs), numel(nonad), numel(fd), numel(errs), resRoot);
end

function r = adaptive_row(res)
m = res.meta; f = res.final; t = res.time;
r = struct('group', m.job_group, 'id', m.job_id, 'model', m.job_args{1}.model.name, ...
    'objective', res.objective.id, 'seed', res.seed, ...
    'n_iterations', f.n_iterations, 'n_initial', f.n_initial, 'n_added', f.n_added, 'n_total', f.n_total, ...
    'r_final', f.r, 'poly_order_final', f.poly_order, 'p1_hat', f.p_hat(1), 'p2_hat', f.p_hat(2), ...
    'lnR_ts', f.lnR_ts, 'lnR_mean', f.lnR_mean, 'lnR_fom', f.lnR_fom, ...
    't_initial_sampling', t.initial_sampling, 't_sample_adding', sum(t.sample_adding), ...
    't_local_reduction', sum(t.local_reduction), 't_reprojection', sum(t.reprojection), ...
    't_training', sum(t.training), 't_optimization', sum(t.optimization), ...
    't_total', t.total, 't_logging', t.logging, 't_wall', t.total_wall, ...
    'n_moved', sum(res.samples.moved), 'mean_rel_err_grid', NaN, ...
    'git_commit', m.git_commit, 'host', m.host, 'matlab', m.matlab_version);
if isfield(res, 'testgrid')
    r.mean_rel_err_grid = mean(res.testgrid.meanRelErr);
end
end

function rows = iteration_rows(res)
m = res.meta; it = res.iterations; t = res.time;
rows = cell(numel(it), 1);
for k = 1:numel(it)
    r = struct('id', m.job_id, 'objective', res.objective.id, 'seed', res.seed, 'iteration', k, ...
        'n_samples', it(k).n_samples, 'r', it(k).r, 'poly_order', it(k).poly_order, ...
        'z0_1', it(k).z0(1), 'z0_2', it(k).z0(2), 'p1_hat', it(k).p_hat(1), 'p2_hat', it(k).p_hat(2), ...
        'lnR_ts', it(k).lnR_ts, 'lnR_mean', it(k).lnR_mean, 'eps', it(k).eps, ...
        'eps_step', field_or_nan(it(k), 'eps_step'), ...
        'n_entry_models', field_or_nan(it(k), 'n_entry_models'), ...
        'n_ts_skipped', field_or_nan(it(k), 'n_ts_skipped'), ...
        'fmincon_maxit', it(k).fmincon_maxit, 'fmincon_iterations', it(k).fmincon_iterations, ...
        'fmincon_funccount', it(k).fmincon_funccount, 'fmincon_exitflag', it(k).fmincon_exitflag, ...
        't_reprojection', t.reprojection(k), 't_training', t.training(k), 't_optimization', t.optimization(k), ...
        'minEigK_ts', NaN, 'minEigM_ts', NaN, 'minEigK_mean', NaN, 'minEigM_mean', NaN, ...
        'frac_indef_ts', NaN, 'frac_indef_mean', NaN, ...
        'next_sample_moved', NaN, 'next_sample_lnR_fom', NaN);
    if isfield(it, 'max_eig_ts') && ~isempty(it(k).max_eig_ts)
        % relative smallest eigenvalue; indefinite if below -1e-12 (round-off level)
        e = it(k).min_eig_ts ./ it(k).max_eig_ts;
        em = it(k).min_eig_mean ./ it(k).max_eig_mean;
        r.minEigK_ts = e(1,1); r.minEigM_ts = e(1,3);
        r.minEigK_mean = em(1,1); r.minEigM_mean = em(1,3);
        r.frac_indef_ts = mean(any(e(:, [1 3]) < -1e-12, 2));
        r.frac_indef_mean = mean(any(em(:, [1 3]) < -1e-12, 2));
    end
    idx = find(res.samples.origin == k, 1);
    if ~isempty(idx)
        r.next_sample_moved = res.samples.moved(idx);
        r.next_sample_lnR_fom = res.samples.lnR_fom(idx);
    end
    rows{k} = r;
end
end

function rows = nonadaptive_rows(res)
m = res.meta;
rows = cell(numel(res.optimization), 1);
for o = 1:numel(res.optimization)
    op = res.optimization(o);
    r = struct('id', m.job_id, 'design', res.design, 'n', res.n, 'seed', res.seed, ...
        'objective', op.objective_id, 'r', res.r, 'poly_order', res.poly_order, ...
        'p1_hat', op.p_hat(1), 'p2_hat', op.p_hat(2), 'lnR_mean', op.lnR_mean, 'lnR_fom', op.lnR_fom, ...
        't_initial_sampling', res.time.initial_sampling, 't_local_reduction', res.time.local_reduction, ...
        't_reprojection', res.time.reprojection, 't_training', res.time.training, ...
        't_build_total', res.time.build_total, 't_optimization', op.time, ...
        'fmincon_iterations', op.fmincon_iterations, 'fmincon_funccount', op.fmincon_funccount, ...
        'mean_rel_err_grid', NaN, 'git_commit', m.git_commit);
    if isfield(res, 'testgrid')
        r.mean_rel_err_grid = mean(res.testgrid.meanRelErr);
    end
    rows{o} = r;
end
end

function r = fd_row(res)
m = res.meta;
r = struct('id', m.job_id, 'level', res.level, 'objective', res.objective.id, 'seed', res.seed, ...
    'p1_hat', res.p_hat(1), 'p2_hat', res.p_hat(2), 'lnR', res.lnR, 'lnR_fom', res.lnR_fom, ...
    't_total', res.time.total, 'fmincon_iterations', res.fmincon_iterations, ...
    'fmincon_funccount', res.fmincon_funccount, 'fmincon_exitflag', res.fmincon_exitflag, ...
    'git_commit', m.git_commit, 'host', m.host);
end

function v = field_or_nan(s, name)
% Optional logging fields are missing in results written by earlier code versions.
if isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = NaN;
end
end

function write_rows(rows, file)
if isempty(rows)
    return
end
writetable(struct2table([rows{:}]'), file);
end
