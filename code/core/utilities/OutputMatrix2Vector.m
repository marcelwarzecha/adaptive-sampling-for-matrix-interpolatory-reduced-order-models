function Y = OutputMatrix2Vector(G)

% INPUTMATRIX2VECTOR resort the entries in the output matrix to a vector

%--% Input
% G: output matrix (number of outputs x r x samples)

%--% Output
% Y: Vector with entries of G

%% resort entries in G to required structure

% first index will be samples, second index r
% in case of multiple outputs, the next outputs will be appended to the right

%   Output 1   Output 2 
% [ . . . . . . . . . . ]   Sample 1
% [ . . . . . . . . . . ]   Sample 2
% [ . . . . . . . . . . ]   Sample 3

numSamples = size(G, 3);
r = size(G, 2);             % reduced order
n_o = size(G, 1);           % number of outputs

Y = zeros(numSamples, r*n_o);

for i = 1:numSamples    % iterate over samples
    
    cnt = 1;            % counter to fill Y while iterating over size of F

    for j = 1:n_o       % iterate over number of outputs
        for k = 1:r     % iterate over columns in G
            
            Y(i, cnt) = G(j,k,i);
            cnt = cnt + 1;
            
        end
    end
    
end
