function [K, D, M, F, G] = assemble_fom(cfg, p)
%ASSEMBLE_FOM Full-order system matrices of the configured model at parameter point p.
switch cfg.model.name
    case 'KelvinCell'
        [K, D, M, F, G] = KelvinCell(p);
    case 'Beam'
        [K, D, M, F, G] = TimoshenkoBeam(cfg.model.L, p(1), p(2));
end
end
