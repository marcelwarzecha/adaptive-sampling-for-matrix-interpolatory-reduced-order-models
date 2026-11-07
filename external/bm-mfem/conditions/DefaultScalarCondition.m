classdef DefaultScalarCondition < Condition
    %DEFAULTSCALARCONDITION Summary of this class goes here
    %   Detailed explanation goes here
    %
    %   Author: Dr. Quirin Aumann (https://qaumann.github.io/)
    %   Part of bm-mfem, Chair of Structural Mechanics, Technical University
    %   of Munich.
    
    properties
    end
    
    methods
        function initialize(obj); end
        
        function dof = getDofList(obj)
            dof = Dof(Node(),[],[]);
            dof.setId(1);
        end
        
        function c = computeRHSContribution(obj)
            n = length(obj.getDofList());
            c = sparse(n,1);
        end
        
        function c = computeKContribution(obj)
            n = length(obj.getDofList());
            c = sparse(n,n);
        end
        
        function c = computeCContribution(obj)
            n = length(obj.getDofList());
            c = sparse(n,n);
        end
        
        function c = computeMContribution(obj)
            n = length(obj.getDofList());
            c = sparse(n,n);
        end
        
    end
end

