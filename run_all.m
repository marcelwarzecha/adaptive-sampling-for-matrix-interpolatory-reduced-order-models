function run_all(queueName, varargin)
%RUN_ALL Run all jobs of a queue one after another; resumable.
%   RUN_ALL('smoke')                         quick check of the installation (~15-30 min)
%   RUN_ALL('example')                       one beam, one Kelvin cell and one non-adaptive run (~1 h)
%   RUN_ALL('beam')                          beam: reference grid and 51 adaptive runs
%   RUN_ALL('kelvin_cell')                   Kelvin cell: 41 adaptive runs
%   RUN_ALL('nonadaptive')                   Kelvin cell: non-adaptive designs (after 'kelvin_cell')
%   RUN_ALL('fd')                            Kelvin cell: optimization with finite differences
%   RUN_ALL('replay')                        Kelvin cell: intermediate pROMs (after 'kelvin_cell')
%   RUN_ALL('paper')                         all of the above except 'smoke' and 'example'
%   RUN_ALL(..., 'groups', {'kc_adaptive'})  only these job groups
%   RUN_ALL(..., 'workers', 4)               parallel pool size (default 12; 0 = serial,
%                                            e.g. without Parallel Computing Toolbox)
%   RUN_ALL(..., 'dry_run', true)            only list the jobs and their status
%   See config/build_queue.m for the jobs of every queue.
%
%   Every finished job writes results/<group>/<id>.mat, plus a .log file.
%   Jobs whose .mat file exists are skipped, so after an interruption
%   simply call RUN_ALL again. A failing job writes <id>.error.txt and the
%   queue continues; the job is retried on the next call.

if strcmp(queueName, 'paper')
    for q = {'beam', 'kelvin_cell', 'nonadaptive', 'fd', 'replay'}
        run_all(q{1}, varargin{:});
    end
    return
end

opts = struct('groups', {{}}, 'workers', [], 'dry_run', false);
for k = 1:2:numel(varargin)
    opts.(lower(varargin{k})) = varargin{k+1};
end

root = fileparts(mfilename('fullpath'));
setup_paths(root);
jobs = build_queue(queueName, root);
if ~isempty(opts.groups)
    jobs = jobs(ismember({jobs.group}, cellstr(opts.groups)));
end
resRoot = fullfile(root, 'results');
collectRoot = resRoot;                       % CSV summaries (smoke test: results/smoke)
if strcmp(queueName, 'smoke')
    collectRoot = fullfile(resRoot, 'smoke');
end

isDone = false(1, numel(jobs));
for i = 1:numel(jobs)
    isDone(i) = isfile(result_file(resRoot, jobs(i)));
end
fprintf('Queue "%s": %d jobs, %d done, %d open (estimated %.1f h).\n', queueName, numel(jobs), ...
    sum(isDone), sum(~isDone), sum([jobs(~isDone).est_min]) / 60);
if opts.dry_run
    for i = 1:numel(jobs)
        fprintf('  [%s] %-16s %s\n', ternary(isDone(i), 'x', ' '), jobs(i).group, jobs(i).id);
    end
    return
end
if all(isDone)
    collect_results(collectRoot);
    return
end

if isempty(opts.workers)
    opts.workers = jobs(1).args{1}.env.n_workers;
end
% Breakpoints or "pause on (caught) errors" would stall an unattended queue:
% switch them off and restore them when run_all returns.
dbState = dbstatus;
if ~isempty(dbState)
    fprintf('Debugger stops are switched off while the queue runs (restored afterwards).\n');
    dbclear all
    restoreDb = onCleanup(@() dbstop(dbState)); %#ok<NASGU>
end
ensure_pool(opts.workers);
setEnvironment('Diagnostic', 'verbosity', 3);    % SparseBayes diagnostics
setEnvironment('Diagnostic', 'fid', 1);

open = find(~isDone);
for n = 1:numel(open)
    job = jobs(open(n));
    out = result_file(resRoot, job);
    outDir = fileparts(out);
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    errFile = fullfile(outDir, [job.id '.error.txt']);
    if isfile(errFile)
        delete(errFile);
    end
    diary(fullfile(outDir, [job.id '.log']));
    fprintf('\n===== [%d/%d] %s / %s  (%s) =====\n', n, numel(open), job.group, job.id, char(datetime('now')));
    meta = run_metadata(root, job, opts.workers);
    t = tic;
    try
        res = feval(job.fn, job.args{:});
        meta.finished = char(datetime('now'));
        meta.job_wall_time = toc(t);
        res.meta = meta;
        save_result(out, res);
        fprintf('===== done in %.1f min =====\n', toc(t) / 60);
    catch ME
        fid = fopen(errFile, 'w');
        fprintf(fid, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        fclose(fid);
        fprintf(2, '===== FAILED after %.1f min: %s =====\n', toc(t) / 60, ME.message);
    end
    diary off
end

collect_results(collectRoot);
end

function f = result_file(resRoot, job)
f = fullfile(resRoot, job.group, [job.id '.mat']);
end

function ensure_pool(n)
if n == 0
    fprintf('Running without parallel pool (parfor loops execute serially).\n');
    return
end
if isempty(ver('parallel')) || ~license('test', 'Distrib_Computing_Toolbox')
    error('run_all:noParallel', ['Parallel Computing Toolbox not available. ' ...
        'Use run_all(..., ''workers'', 0) to run serially (fine for the smoke test, not for production timings).']);
end
pool = gcp('nocreate');
if ~isempty(pool) && pool.NumWorkers == n
    return
end
delete(pool);
try
    c = parcluster('Processes');    % R2022b and newer
catch
    c = parcluster('local');        % R2022a
end
if c.NumWorkers < n
    c.NumWorkers = n;
end
parpool(c, n);
end

function s = ternary(cond, a, b)
if cond
    s = a;
else
    s = b;
end
end
