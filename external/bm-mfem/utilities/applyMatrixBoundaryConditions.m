function [ matrix ] = applyMatrixBoundaryConditions( matrix, bc )
%APPLYMATRIXBOUNDARYCONDITIONS Removes bc lines and columns from a matrix
%   Detailed explanation goes here
%
%   Author: Dr. Quirin Aumann (https://qaumann.github.io/)
%   Part of bm-mfem, Chair of Structural Mechanics, Technical University
%   of Munich.
if ~bc; return; end
    matrix(bc, :) = [];
    matrix(:, bc) = [];

end

