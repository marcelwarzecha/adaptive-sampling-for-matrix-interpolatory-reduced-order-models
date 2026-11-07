function meta = run_metadata(root, job, n_workers)
%RUN_METADATA Provenance information stored with every result.
meta.job_group = job.group;
meta.job_id = job.id;
meta.job_fn = job.fn;
meta.job_args = job.args;
meta.started = char(datetime('now'));
meta.matlab_version = version;
meta.computer = computer;
meta.host = [getenv('COMPUTERNAME') getenv('HOSTNAME')];
meta.cpu = getenv('PROCESSOR_IDENTIFIER');
meta.n_workers = n_workers;
[st, out] = system(sprintf('git -C "%s" rev-parse HEAD', root));
meta.git_commit = strtrim(ternary(st == 0, out, 'unknown'));
[st, out] = system(sprintf('git -C "%s" status --porcelain -- code external src config', root));
meta.git_dirty = ternary(st == 0, ~isempty(strtrim(out)), NaN);
end

function s = ternary(cond, a, b)
if cond
    s = a;
else
    s = b;
end
end
