function setup_paths(root)
%SETUP_PATHS Add the framework code, the bundled bm-mfem subset and the experiment code to the path.
if nargin < 1
    root = fileparts(mfilename('fullpath'));
end
addpath(genpath(fullfile(root, 'code', 'core')));
addpath(genpath(fullfile(root, 'external', 'bm-mfem')));
addpath(fullfile(root, 'src'));
addpath(fullfile(root, 'config'));
end
