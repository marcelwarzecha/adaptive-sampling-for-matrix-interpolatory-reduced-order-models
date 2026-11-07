function Y = InputMatrix2Vector(F)

% INPUTMATRIX2VECTOR resort the entries in the input matrix to a vector

%--% Input
% F: input matrix (r x number of inputs x samples)

%--% Output
% Y: Vector with entries of F

%% resort entries in F to required structure

% first indes will be samples, second index r
% in case of multiple inputs, the next inputs will be appended to the right

%    Input 1   Input 2 
% [ . . . . . . . . . . ]   Sample 1
% [ . . . . . . . . . . ]   Sample 2
% [ . . . . . . . . . . ]   Sample 3

numSamples = size(F, 3);
r = size(F, 1);             % reduced order
n_i = size(F, 2);           % number of inputs

Y = zeros(numSamples, r*n_i);

for i = 1:numSamples    % iterate over samples
    
    cnt = 1;            % counter to fill Y while iterating over size of F

    for j = 1:n_i       % iterate over number of inputs
        for k = 1:r     % iterate over rows in F
            
            Y(i, cnt) = F(k,j,i);
            cnt = cnt + 1;
            
        end
    end
    
end
