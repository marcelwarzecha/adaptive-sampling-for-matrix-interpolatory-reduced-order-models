function Te = load_testgrid(cfg)
%LOAD_TESTGRID Full-order reference transfer functions of the Kelvin cell on a 51x51 grid (cached).
%   Expects data/FFD_51x51_1000Hz.mat (S: 2601x2 [m], omegaHz: 1..1000 Hz,
%   y: 1x1000x2601 complex). Verified to match the current KelvinCell.m.
persistent cache
if isempty(cache)
    file = cfg.paths.testgrid;
    if ~isfile(file)
        error('load_testgrid:missing', 'Reference data not found: %s (see data/README.md).', file);
    end
    T = load(file, 'S', 'omegaHz', 'y');
    if size(T.S, 1) < size(T.S, 2)
        T.S = T.S';
    end
    cache.S = T.S;
    cache.f = T.omegaHz(:)';
    cache.y = reshape(T.y, size(T.y, 2), size(T.y, 3));     % n_f x n_points
end
Te = cache;
end
