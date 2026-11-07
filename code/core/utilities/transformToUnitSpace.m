function S_unit = transformToUnitSpace(S, Bounds)

%--% Input
% S: sample(s), one sample per row
% Bounds: bounds of the real parameter space, the first row contains the
% lower bounds of all parameters, the second row the upper bounds

%--% Output
% S_unit: sample(s) transformed to unit interval ([0,1])


%% transformation
% initialize S_unit
S_unit = zeros(size(S));
    for i = 1:size(S,1)
        for j = 1:size(S,2)
            
            S_unit(i,j) = (S(i,j) - Bounds(1,j)) / (Bounds(2,j) - Bounds(1,j));
            
        end
    end
end

