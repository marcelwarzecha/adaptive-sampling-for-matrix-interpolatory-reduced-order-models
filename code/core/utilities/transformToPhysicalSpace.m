function S = transformToPhysicalSpace(S_unit, Bounds)
%--% Input
% S: sample(s), one sample per row
% Bounds: bounds of the real parameter space, the first row contains the
% lower bounds of all parameters, the second row the upper bounds

%--% Output
% S_unit: sample(s) transformed to unit interval ([0,1])


%% transformation

% initialize S_unit
S = zeros(size(S_unit));

for i = 1:size(S_unit,1)
    for j = 1:size(S_unit,2)
        
        S(i,j) = Bounds(1,j)+ S_unit(i,j)* (Bounds(2,j)-Bounds(1,j));
        
    end
end
