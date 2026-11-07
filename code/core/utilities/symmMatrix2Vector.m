function Y = symmMatrix2Vector(A)

% SYMMMATRIX2VECTOR resort the entries in the upper diagonal of A into a
% vector

%--% Input
% A: matrix of symmetric matrices, in each layer a symmetric matrix is
% stored

%--% Output
% Y: Vector with entries of A

%% resort entries in A to required structure

% entries in A are copied to Y in the following order:
% 1 - 2 - 4 - 7
%     |   |   |
%     3   5   8
%         |   |
%         6   9
%             |
%            10

n = size(A, 1);             % size of matrices in A

% column-major order of the upper triangle is exactly the order above
Y = reshape(A, n * n, []);
Y = Y(triu(true(n)), :).';
