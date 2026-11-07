classdef RectangularTimoshenkoBeamElement3d2n < LinearElement
    %RECTANGULARTIMOSHENKOBEAMELEMENT3D2N 3D two-node Timoshenko beam
    %element with rectangular cross section and hard-coded system matrices.
    %
    %   Adapted from: H. Panzer, J. Hubele, R. Eid, B. Lohmann, "Generating
    %   a Parametric Finite Element Model of a 3D Cantilever Timoshenko Beam
    %   Using Matlab", Technical Reports on Automatic Control, Technical
    %   University of Munich, 2009.
    %   https://mediatum.ub.tum.de/doc/1072355/file.pdf/1000
    %
    %   Author: Sebastian Resch-Schopper
    %   (https://github.com/SebastianResch-Schopper)
    
    properties (Access = private)
        length0 %initial length
    end
    
    methods
        % constructor
        function obj = RectangularTimoshenkoBeamElement3d2n(id, nodeArray)
            
            requiredPropertyNames = cellstr(["IY", "IZ", "IT", ...
                "YOUNGS_MODULUS", "POISSON_RATIO", "CROSS_SECTION"]);
            
            if nargin == 0
                super_args = {};
            elseif nargin == 2
                if ~(length(nodeArray) == 2 && isa(nodeArray,'Node'))
                    error('problem with the nodes in element %d', id);
                end
                super_args = {id, nodeArray, requiredPropertyNames};
            end
            
            % call the super class constructor
            obj@LinearElement(super_args{:});
            obj.dofNames = cellstr(["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z", ...
                "ROTATION_X", "ROTATION_Y", "ROTATION_Z"]);
        end
        
        function initialize(obj)
            obj.localSystem = obj.computeLocalSystem();
            obj.length0 = computeLength(obj.nodeArray(1).getCoords, ...
                obj.nodeArray(2).getCoords);
        end
        
        function stiffnessMatrix = computeLocalStiffnessMatrix(obj)
           E = obj.getPropertyValue('YOUNGS_MODULUS');
           nu = obj.getPropertyValue('POISSON_RATIO');
           G = E/2/(1+nu); % G: Shear modulus [N/mˆ2]
           A = obj.getPropertyValue('CROSS_SECTION');
           Iy = obj.getPropertyValue('IY');
           Iz = obj.getPropertyValue('IZ');
           It = obj.getPropertyValue('IT');
           ASy = 5/6*A ; ASz = 5/6*A ; % effective area of shear
           L = obj.length0;
           L2 = L*L;
           L3 = L2*L;
           Py = 12*E*Iz/(G*ASy*L2) ; Pz = 12*E*Iy/(G*ASz*L2) ; % Phi

                      
           K11 = zeros(6,6);
           K11(1,1) = E*A/L ;
           K11(2,2) = 12*E*Iz/(L3*(1+Py)) ;
           K11(3,3) = 12*E*Iy/(L3*(1+Pz)) ;
           K11(4,4) = G*It/L ;
           K11(5,5) = (4+Pz)*E*Iy/(L*(1+Pz)) ;
           K11(6,6) = (4+Py)*E*Iz/(L*(1+Py)) ;
           K11(2,6) = 6*E*Iz/(L2*(1+Py)) ;
           K11(6,2) = K11(2,6) ;
           K11(3,5) = -6*E*Iy/(L2*(1+Pz)) ;
           K11(5,3) = K11(3,5) ;
           K22 = -K11 + 2*diag(diag(K11));
           K21 = K11 - 2*diag(diag(K11));
           K21(5,5) = (2-Pz)*E*Iy/(L*(1+Pz)) ;
           K21(6,6) = (2-Py)*E*Iz/(L*(1+Py)) ;
           K21(2,6) = -K21(6,2);
           K21(3,5) = -K21(5,3);
           stiffnessMatrix = sparse([K11, K21'; K21, K22]);
           
           tMat = obj.getTransformationMatrix;
           stiffnessMatrix = tMat' * stiffnessMatrix * tMat;
        end
        
        function massMatrix = computeLocalMassMatrix(obj)
            A = obj.getPropertyValue('CROSS_SECTION');
            L = obj.length0;
            L2 = L*L;
            rho = obj.tryGetPropertyValue('DENSITY');
            m = A * L * rho;
            Iy = obj.getPropertyValue('IY');
            Iz = obj.getPropertyValue('IZ');
            Ip = Iy + Iz;
            
            M11 = zeros(6,6);
            M11(1,1) = 1/3;
            M11(2,2) = 13/35 + 6*Iz/(5*A*L2);
            M11(3,3) = 13/35 + 6*Iy/(5*A*L2);
            M11(4,4) = Ip/(3*A);
            M11(5,5) = L2/105 + 2*Iy/(15*A);
            M11(6,6) = L2/105 + 2*Iz/(15*A);
            M11(6,2) = 11*L/210 + Iz/(10*A*L);
            M11(2,6) = M11(6,2) ;
            M11(5,3) = -11*L/210 - Iy/(10*A*L);
            M11(3,5) = M11(5,3) ;
            M22 = -M11 + 2*diag(diag(M11));
            M21 = zeros(6,6);
            M21(1,1) = 1/6;
            M21(2,2) = 9/70 - 6*Iz/(5*A*L2);
            M21(3,3) = 9/70 - 6*Iy/(5*A*L2);
            M21(4,4) = Ip/(6*A);
            M21(5,5) = -L2/140 - Iy/(30*A);
            M21(6,6) = -L2/140 - Iz/(30*A);
            M21(6,2) = -13*L/420 + Iz/(10*A*L);
            M21(2,6) = -M21(6,2);
            M21(5,3) = 13*L/420 - Iy/(10*A*L);
            M21(3,5) = -M21(5,3);
            massMatrix = sparse(m*[M11, M21'; M21, M22]);
            
            tMat = obj.getTransformationMatrix;
            massMatrix = tMat' * massMatrix * tMat;
            
        end
        
        function dampingMatrix = computeLocalDampingMatrix(obj)
            eProperties = obj.getProperties;
            if (eProperties.hasValue('RAYLEIGH_ALPHA')) ...
                    && (eProperties.hasValue('RAYLEIGH_BETA'))
                dampingMatrix = eProperties.getValue('RAYLEIGH_ALPHA') * obj.computeLocalMassMatrix ...
                    + eProperties.getValue('RAYLEIGH_BETA') * obj.computeLocalStiffnessMatrix;
            else
                dampingMatrix = sparse(12,12);
            end          
        end
        
        function forceVector = computeLocalForceVector(obj)
           forceVector = sparse(1,12);
%            disp = element.getValuesVector('end');
%            forceVector = element.computeLocalStiffnessMatrix() * disp';
%            forceVector = forceVector';
        end
        
        function dofs = getDofList(obj)
            dofs([1 7]) = obj.nodeArray.getDof('DISPLACEMENT_X');
            dofs([2 8]) = obj.nodeArray.getDof('DISPLACEMENT_Y'); 
            dofs([3 9]) = obj.nodeArray.getDof('DISPLACEMENT_Z');
            
            dofs([4 10]) = obj.nodeArray.getDof('ROTATION_X');
            dofs([5 11]) = obj.nodeArray.getDof('ROTATION_Y'); 
            dofs([6 12]) = obj.nodeArray.getDof('ROTATION_Z');
        end
        
        function vals = getValuesVector(obj, step)
            vals = zeros(1,12);
            
            vals([1 7]) = obj.nodeArray.getDofValue('DISPLACEMENT_X',step);
            vals([2 8]) = obj.nodeArray.getDofValue('DISPLACEMENT_Y',step);
            vals([3 9]) = obj.nodeArray.getDofValue('DISPLACEMENT_Z',step);
            
            vals([4 10]) = obj.nodeArray.getDofValue('ROTATION_X',step);
            vals([5 11]) = obj.nodeArray.getDofValue('ROTATION_Y',step);
            vals([6 12]) = obj.nodeArray.getDofValue('ROTATION_Z',step);
        end
        
        function vals = getFirstDerivativesVector(obj, step)
            vals = zeros(1,12);
            
            [~, vals([1 7]), ~] = obj.nodeArray.getDof('DISPLACEMENT_X').getAllValues(step);
            [~, vals([2 8]), ~] = obj.nodeArray.getDof('DISPLACEMENT_Y').getAllValues(step);
            [~, vals([3 9]), ~] = obj.nodeArray.getDof('DISPLACEMENT_Z').getAllValues(step);
            
            [~, vals([4 10]), ~] = obj.nodeArray.getDof('ROTATION_X').getAllValues(step);
            [~, vals([5 11]), ~] = obj.nodeArray.getDof('ROTATION_Y').getAllValues(step);
            [~, vals([6 12]), ~] = obj.nodeArray.getDof('ROTATION_Z').getAllValues(step);
        end
        
        function vals = getSecondDerivativesVector(obj, step)
            vals = zeros(1,12);            
            
            [~, ~, vals([1 7])] = obj.nodeArray.getDof('DISPLACEMENT_X').getAllValues(step);
            [~, ~, vals([2 8])] = obj.nodeArray.getDof('DISPLACEMENT_Y').getAllValues(step);
            [~, ~, vals([3 9])] = obj.nodeArray.getDof('DISPLACEMENT_Z').getAllValues(step);
            
            [~, ~, vals([4 10])] = obj.nodeArray.getDof('ROTATION_X').getAllValues(step);
            [~, ~, vals([5 11])] = obj.nodeArray.getDof('ROTATION_Y').getAllValues(step);
            [~, ~, vals([6 12])] = obj.nodeArray.getDof('ROTATION_Z').getAllValues(step);
        end
        
        function update(obj)
            obj.length0 = computeLength(obj.nodeArray(1).getCoords, ...
                    obj.nodeArray(2).getCoords);
        end
        
    end
    
end

