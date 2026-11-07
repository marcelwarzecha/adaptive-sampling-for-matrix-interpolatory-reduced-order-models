function [R, ROM_order] = getGlobalBasis(Vr, SVD_retain_frac)
%GETGLOBALBASIS Summary of this function goes here
%   INPUT: 
%   - Vr: Cell containing reduced bases from IRKA nDOF x ROM_order x
%   numSamples
%   - SVD_retain_frac: Chosen fraction of total energy of the system that
%   should be considered (usually 0.99999)
%
%   OUTPUT: 
%   - R: Global reduction basis resulting from SVD nDOF x ROM_order (new
%   ROM Order)
%   - ROM_order: resulting from SVD_retain_frac

V_all = cell2mat(Vr);
    [U,Sigma,~]=svd(full(V_all),"econ");
       
    % Cut off SVD after SVD_retain_frac, % of information is retained
    svd_int = cumsum(diag(Sigma));
    svd_r = find((svd_int / max(svd_int)) > SVD_retain_frac, 1); 
    %ROM_order = min(100,svd_r);
    ROM_order = svd_r;
    %svd_r = 8;
    
    %Plot singular value distribution of SVD
    % figure
    % hold on
    % plot(diag(Sigma))
    % set(gca, 'YScale', 'log')
    % ylabel("SV Magnitude");
    % yyaxis right
    % plot(100 * svd_int / max(svd_int))
    % xlabel("SVD Terms");
    % ylabel("Retained Information [%]");
    
    R= U(:,1:ROM_order); % calculate global basis out of snapshots
end

