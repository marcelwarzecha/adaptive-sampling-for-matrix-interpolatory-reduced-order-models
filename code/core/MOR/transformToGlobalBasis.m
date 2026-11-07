function [Kr_t, Dr_t, Mr_t, Fr_t, Gr_t] = transformToGlobalBasis(K, D, M, F, G, R)
%TRANSFORMTOGLOBALBASIS 
%   INPUT: 
%   - Kr, Dr, Mr, Fr, Gr: locally reduced operators
%   - R: Projection Matrix nDOF x ROM_order
%
%   OUTPUT:
%   - Kr_t, Dr_t, Mr_t, Fr_t, Gr_t: Transformed operators

    % get numSamples
    numSamples = length(K);

    Kr_t = cell(1, numSamples); Dr_t = cell(1, numSamples); Mr_t = cell(1, numSamples);
    Gr_t = cell(1, numSamples); Fr_t = cell(1, numSamples);
    
    for i = 1:numSamples % perform reduction with global basis
        Kr_t{i} = R'* K{i}*R;
        Dr_t{i} = R'* D{i}*R;
        Mr_t{i} = R'* M{i}*R;
        Fr_t{i} = R'*F{i};
        Gr_t{i} = G{i}*R;
    end
end

