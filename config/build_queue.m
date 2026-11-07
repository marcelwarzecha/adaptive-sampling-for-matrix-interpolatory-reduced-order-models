function jobs = build_queue(queueName, root)
%BUILD_QUEUE List of jobs for run_all. Each job writes one result file, results/<group>/<id>.mat.
%   'smoke'       : quick check of the installation (~15-30 min), reduced settings
%   'example'     : one run of each kind (~1 h): the representative beam run (seed 30) and
%                   the beam reference grid, the representative Kelvin cell run (seed 30),
%                   and non-adaptive Kelvin cell pROMs from a Latin hypercube design with
%                   N = 32 (seed 1) and a 6x6 FFD; enough for Figs. 4, 5, 7, 8, 10, 12, 13
%   'beam'        : beam_reference  true beam objective on a 51x51 FOM grid
%                   beam_showcase   representative run, seed 30, surrogate surfaces logged
%                   beam_adaptive   adaptive runs, seeds 1-50
%   'kelvin_cell' : kc_showcase     representative run, seed 30, surrogate surfaces logged
%                   kc_adaptive     adaptive runs, 4 objectives x seeds 1-10
%   'nonadaptive' : kc_nonadaptive  square FFDs 5x5, 6x6, 7x7, 10x10; LHS with N = 30, 35,
%                                   40, 44 x seeds 1-10; LHS with N = 32 x seeds 1-10; and
%                                   for every adaptive run, the LHS with its final N and seed
%                                   (needs the results of 'kelvin_cell')
%   'fd'          : kc_fd           fmincon with finite differences on the FOM and on a
%                                   local ROM, 4 objectives x seeds 1-5
%   'replay'      : kc_replay       posterior-mean pROMs of every adaptive Kelvin cell run
%                                   after each iteration, evaluated near the true optimum
%                                   (needs the results of 'kelvin_cell')
%   'paper' (run_all only) runs 'beam', 'kelvin_cell', 'nonadaptive', 'fd' and 'replay'.

KC = config_for('KelvinCell', root);
BM = config_for('Beam', root);
jobs = struct('group', {}, 'id', {}, 'fn', {}, 'args', {}, 'est_min', {});

switch queueName
    case 'smoke'
        jobs = add(jobs, 'smoke', 'environment', 'check_environment', {KC}, 1);
        jobs = add(jobs, 'smoke', 'gradient_beam', 'check_gradient', {BM, 1}, 1);
        jobs = add(jobs, 'smoke', 'gradient_kc', 'check_gradient', {KC, 1}, 5);

        c = BM;
        c.loop.outerloop_minit = 2; c.loop.outerloop_maxit = 2; c.log.surface_points = 3;
        jobs = add(jobs, 'smoke', 'adaptive_beam_2it', 'run_adaptive', {c, 1, 1}, 1);
        jobs = add(jobs, 'smoke', 'reference_beam_5x5', 'run_reference_grid', {BM, 5}, 1);

        c = KC;
        c.loop.outerloop_minit = 2; c.loop.outerloop_maxit = 2;
        c.log.surface_points = 2; c.log.testgrid_stride = 26;
        jobs = add(jobs, 'smoke', 'adaptive_kc_2it', 'run_adaptive', {c, 1, 1}, 8);

        c = KC; c.log.testgrid_stride = 26;
        jobs = add(jobs, 'smoke', 'nonadaptive_kc_lhs9', 'run_nonadaptive', {c, 'LHS', 9, 1}, 5);

        c = KC; c.fd.MaxIterations = 1;
        jobs = add(jobs, 'smoke', 'fd_kc_fom_1it', 'run_fd_comparison', {c, 1, 'FOM', 1}, 3);
        jobs = add(jobs, 'smoke', 'fd_kc_rom_1it', 'run_fd_comparison', {c, 1, 'ROM', 1}, 5);

    case 'example'
        jobs = beam_reference(jobs, BM);
        jobs = beam_showcase(jobs, BM);
        jobs = kc_showcase(jobs, KC);
        jobs = lhs(jobs, KC, 32, 1);
        jobs = ffd(jobs, KC, 6);

    case 'beam'
        jobs = beam_reference(jobs, BM);
        jobs = beam_showcase(jobs, BM);
        for seed = 1:50
            jobs = add(jobs, 'beam_adaptive', sprintf('beam_f50-100_k1_seed%02d', seed), ...
                'run_adaptive', {BM, 1, seed}, 0.2);
        end

    case 'kelvin_cell'
        jobs = kc_showcase(jobs, KC);
        for seed = 1:10
            for o = 1:numel(KC.objectives)
                id = sprintf('kc_%s_seed%02d', KC.objectives(o).id, seed);
                jobs = add(jobs, 'kc_adaptive', id, 'run_adaptive', {KC, o, seed}, 19);
            end
        end

    case 'nonadaptive'
        for n = [5 6 7 10]
            jobs = ffd(jobs, KC, n);
        end
        pairs = zeros(0, 2);                     % LHS designs [N, seed]
        for seed = 1:10
            pairs = [pairs; [30 35 40 44 32]', seed * ones(5, 1)]; %#ok<AGROW>
        end
        % Equal budget: for every adaptive run, the LHS with its final N and seed. Runs
        % with equal N and seed share one design, which serves all objectives.
        missing = 0;
        for seed = 1:10
            for o = 1:numel(KC.objectives)
                file = adaptive_result(root, KC, o, seed);
                if isfile(file)
                    R = load(file, 'res');
                    pairs(end+1, :) = [R.res.final.n_total, seed]; %#ok<AGROW>
                else
                    missing = missing + 1;
                end
            end
        end
        if missing > 0
            warning('build_queue:equalBudget', ['%d adaptive Kelvin cell results are missing; ' ...
                'their equal-budget LHS designs are skipped. Run run_all(''kelvin_cell'') first.'], missing);
        end
        pairs = unique(pairs, 'rows');
        for i = 1:size(pairs, 1)
            jobs = lhs(jobs, KC, pairs(i, 1), pairs(i, 2));
        end

    case 'fd'
        for seed = 1:5
            for o = 1:numel(KC.objectives)
                for level = {'FOM', 'ROM'}
                    id = sprintf('kc_fd_%s_%s_seed%02d', level{1}, KC.objectives(o).id, seed);
                    jobs = add(jobs, 'kc_fd', id, 'run_fd_comparison', {KC, o, level{1}, seed}, 45);
                end
            end
        end

    case 'replay'
        for seed = 1:10
            for o = 1:numel(KC.objectives)
                file = adaptive_result(root, KC, o, seed);
                if ~isfile(file)
                    error('build_queue:replay', '%s not found. Run run_all(''kelvin_cell'') first.', file);
                end
                id = sprintf('kc_%s_seed%02d', KC.objectives(o).id, seed);
                jobs = add(jobs, 'kc_replay', id, 'replay_adaptive', {KC, file, 1}, 10);
            end
        end

    otherwise
        error('build_queue:unknown', ['Unknown queue "%s" (use ''smoke'', ''example'', ''beam'', ' ...
            '''kelvin_cell'', ''nonadaptive'', ''fd'', ''replay'' or ''paper'').'], queueName);
end
end

function cfg = config_for(modelName, root)
cfg = production_config(modelName);
cfg.paths.testgrid = fullfile(root, 'data', 'FFD_51x51_1000Hz.mat');
end

function file = adaptive_result(root, KC, o, seed)
file = fullfile(root, 'results', 'kc_adaptive', sprintf('kc_%s_seed%02d.mat', KC.objectives(o).id, seed));
end

function jobs = beam_reference(jobs, BM)
jobs = add(jobs, 'beam_reference', 'beam_reference_51x51', 'run_reference_grid', {BM, 51}, 10);
end

function jobs = beam_showcase(jobs, BM)
c = BM; c.log.surface_points = 20;
jobs = add(jobs, 'beam_showcase', 'beam_f50-100_k1_seed30', 'run_adaptive', {c, 1, 30}, 1);
end

function jobs = kc_showcase(jobs, KC)
c = KC; c.log.surface_points = 20;
jobs = add(jobs, 'kc_showcase', 'kc_f300-400_k1_seed30', 'run_adaptive', {c, 1, 30}, 20);
end

function jobs = ffd(jobs, KC, n)
jobs = add(jobs, 'kc_nonadaptive', sprintf('kc_ffd%dx%d', n, n), 'run_nonadaptive', {KC, 'FFD', n, 1}, n^2 / 4);
end

function jobs = lhs(jobs, KC, n, seed)
jobs = add(jobs, 'kc_nonadaptive', sprintf('kc_lhs%d_seed%02d', n, seed), ...
    'run_nonadaptive', {KC, 'LHS', n, seed}, 6.5);
end

function jobs = add(jobs, group, id, fn, args, est_min)
jobs(end+1) = struct('group', group, 'id', id, 'fn', fn, 'args', {args}, 'est_min', est_min);
end
