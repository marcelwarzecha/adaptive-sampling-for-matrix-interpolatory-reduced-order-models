function [Kr, Dr, Mr, fr, gr, Vr] = SO_IRKA_SISO(K, D, M, f, g, s, tol, maxiter)
% SO_IRKA_SISO perform Model Order Reduction using the Second-order 
% iterative rational Krylov algorithm
%
% [Kr, Dr, Mr, fr, gr, Vr] = SO_IRKA_SISO(K,D,M,f,g,s,tol,maxiter) returns
% the reduced order matrices as well as the reduced basis from the full
% order matrices.
%
% Inputs: 
%   - K: stiffness matrix (n x n matrix)
%   - D: damping matrix (n x n matrix)
%   - M: mass matrix (n x n matrix)
%   - f: input vector (n x 1 vector)
%   - g: output vector (1 x n vector)
%   - s: vector of initial expansion frequencies closed under complex
%       conjugation (to get real reduced matrices)
%       example: s = 1i * linspace(-s_max, s_max, r)
%       s needs to be in radians per second (1 rad/s = 2 * pi * 1 Hz)
%   - tol : convergence tolerance of the expansion points
%   - maxiter : maximum number of iterations
%
% Outputs:
%   - Kr: reduced stiffness matrix
%   - Dr: reduced damping matrix
%   - Mr: reduced mass matrix
%   - fr: reduced force vector
%   - gr: reduced output vector
%   - Vr: reduced basis

%% sort complex conjugate pairs in s and initialize basis

s = s(:);
r = length(s);
s = cplxpair(s);
Vr = zeros(size(M,1),r);

%% initial basis computation

% since s only contains complex conjugate pairs, it is sufficient to
% iterate over half of tem
for ii = 1:r/2
    % dynamic stiffness matrix
    tmp1 = s(2*ii)^2*M + s(2*ii)*D + K;
    % complex solution
    x = tmp1\f;
    % split real and imaginary part in basis to get real reduced matrices
    Vr(:,2*ii-1) = real(x);
    Vr(:,2*ii) = imag(x);
end

% orthogonalize basis
[Vr, ~] = qr(Vr,0);

%% iterative basis computation

iter = 1;
err = 1;
while (iter <= maxiter) && (err > tol)

    s_old = s;
    
    % perform reduction
    Mr = Vr' * M * Vr;
    Dr = Vr' * D * Vr;
    Kr = Vr' * K * Vr;
    
    % solve quadratic eigenvalue problem
    lam = polyeig(Kr,Dr,Mr);
    
    % remove real and unstable eigenvalues
    lam(imag(lam) == 0) = [];
    lam(real(lam) > 0) = [];
    
    % sort for the eigenvalues next to the imaginary axis
    [~,order] = sort(abs(imag(lam)),'ascend');
    
    % choose expansion point as mirror images of eigenvalues
    s = -lam(order);
    s = s(1:r);

    % sort again
    s = cplxpair(s);
    
    % recompute basis
    Vr = zeros(size(M,1),r);
    
    % same as for the initial basis computation
    for ii = 1:r/2
        tmp1 = s(ii*2)^2*M + s(ii*2)*D + K;
        x = tmp1\f;
        Vr(:,2*ii-1) = real(x);
        Vr(:,2*ii) = imag(x);
    end
    
    % orthogonalize
    [Vr, ~] = qr(Vr,0);
    
    % check convergence
    err = norm(sort(s,'ComparisonMethod','abs') - ...
        sort(s_old,'ComparisonMethod','abs'))/norm(s_old);
    %fprintf('\terr = %e.\n',err)
    
    % increase iteration counter
    iter = iter + 1;
    
end

fprintf("IRKA: Final iteration %u, error: %e.\n", iter, err)
%% perform projection

Mr = Vr' * M * Vr;
Dr = Vr' * D * Vr;
Kr = Vr' * K * Vr;
fr = Vr' * f;
gr = g * Vr;

end

