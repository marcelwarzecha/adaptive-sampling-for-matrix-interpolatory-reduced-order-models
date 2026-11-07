function [ duplicates_array ] = findDuplicates( array )
%HASDUPLICATES Returns repeated values in an array
%   Detailed explanation goes here
%
%   Author: Dr. Quirin Aumann (https://qaumann.github.io/)
%   Part of bm-mfem, Chair of Structural Mechanics, Technical University
%   of Munich.
sorted_array = sort(array);
duplicates_indices = find(diff(sorted_array) == 0);
duplicates_array = sorted_array(duplicates_indices);

end

