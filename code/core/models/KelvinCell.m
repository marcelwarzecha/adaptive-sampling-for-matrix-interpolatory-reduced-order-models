function [K,D,M,F,G,westDofs,eastDofs,bottomDofs,Coords] = KelvinCell(s)

%% lengths

lx = s(1);
ly = s(2); % 50e-3;
lz = 50e-3;

ax = 0.25*lx;
ay = 0.25*ly;
az = 0.25*lz;

bx = 2*ax;
by = 2*ay;
bz = 2*az;

%% corners

% bottom square
corner_B_N = [0, ay, -bz];
corner_B_E = [ax, 0, -bz];
corner_B_S = [0, -ay, -bz];
corner_B_W = [-ax, 0, -bz];

% north square
corner_N_B = [0, by, -az];
corner_N_E = [ax, by, 0];
corner_N_T = [0, by, az];
corner_N_W = [-ax, by, 0];

% east square
corner_E_B = [bx, 0, -az];
corner_E_N = [bx, ay, 0];
corner_E_T = [bx, 0, az];
corner_E_S = [bx, -ay, 0];

% south square
corner_S_B = [0, -by, -az];
corner_S_E = [ax, -by, 0];
corner_S_T = [0, -by, az];
corner_S_W = [-ax, -by, 0];

% west square
corner_W_B = [-bx, 0, -az];
corner_W_N = [-bx, ay, 0];
corner_W_T = [-bx, 0, az];
corner_W_S = [-bx, -ay, 0];

% top square
corner_T_N = [0, ay, bz];
corner_T_E = [ax, 0, bz];
corner_T_S = [0, -ay, bz];
corner_T_W = [-ax, 0, bz];

%% lines

lines = zeros(2,3,36);

% bottom square
lines(:,:,1) = [corner_B_N; corner_B_E];
lines(:,:,2) = [corner_B_E; corner_B_S];
lines(:,:,3) = [corner_B_S; corner_B_W];
lines(:,:,4) = [corner_B_W; corner_B_N];

% bottom square to middle squares
lines(:,:,5) = [corner_B_N; corner_N_B];
lines(:,:,6) = [corner_B_E; corner_E_B];
lines(:,:,7) = [corner_B_S; corner_S_B];
lines(:,:,8) = [corner_B_W; corner_W_B];

% north square
lines(:,:,9) = [corner_N_B; corner_N_E];
lines(:,:,10) = [corner_N_E; corner_N_T];
lines(:,:,11) = [corner_N_T; corner_N_W];
lines(:,:,12) = [corner_N_W; corner_N_B];

% east square
lines(:,:,13) = [corner_E_B; corner_E_S];
lines(:,:,14) = [corner_E_S; corner_E_T];
lines(:,:,15) = [corner_E_T; corner_E_N];
lines(:,:,16) = [corner_E_N; corner_E_B];

% south square
lines(:,:,17) = [corner_S_B; corner_S_W];
lines(:,:,18) = [corner_S_W; corner_S_T];
lines(:,:,19) = [corner_S_T; corner_S_E];
lines(:,:,20) = [corner_S_E; corner_S_B];

% west square
lines(:,:,21) = [corner_W_B; corner_W_N];
lines(:,:,22) = [corner_W_N; corner_W_T];
lines(:,:,23) = [corner_W_T; corner_W_S];
lines(:,:,24) = [corner_W_S; corner_W_B];

% connection of middle squares
lines(:,:,25) = [corner_N_E; corner_E_N];
lines(:,:,26) = [corner_E_S; corner_S_E];
lines(:,:,27) = [corner_S_W; corner_W_S];
lines(:,:,28) = [corner_W_N; corner_N_W];

% middle squares to top square
lines(:,:,29) = [corner_N_T; corner_T_N];
lines(:,:,30) = [corner_E_T; corner_T_E];
lines(:,:,31) = [corner_S_T; corner_T_S];
lines(:,:,32) = [corner_W_T; corner_T_W];

% top square
lines(:,:,33) = [corner_T_N; corner_T_E];
lines(:,:,34) = [corner_T_E; corner_T_S];
lines(:,:,35) = [corner_T_S; corner_T_W];
lines(:,:,36) = [corner_T_W; corner_T_N];

%% FEM model

% initialize FEM model
model = FemModel();

% number of elements
nEle = 50;

% element type
elementName = 'RectangularTimoshenkoBeamElement3d2n';

%% iterate over lines to generate coordinates

% initialize matrix of coordinates
Coords = zeros(36 * (nEle+1), 3);

% coordinate counter
i_co = 0;

for ln = 1:36
    
    % increase coordinate counter
    i_co = i_co + 1;
    
    for co = 1:3
        
        % compute coordinates of nodes along line
        Coords((ln-1) * (nEle+1) + 1:ln * (nEle+1), co) = ...
            linspace(lines(1,co,ln), lines(2,co,ln), nEle+1)';
        
    end
end

% make coordinates unique (delete double nodes at corners)
[Coords, ~, io] = unique(Coords, 'stable', 'rows');

%% create nodes

for nd = 1:size(Coords,1)
    
    model.addNewNode(nd, Coords(nd,1), Coords(nd,2), Coords(nd,3));
    
end

% add dofs
model.getAllNodes.addDof({'DISPLACEMENT_X', 'DISPLACEMENT_Y', 'DISPLACEMENT_Z', 'ROTATION_X', 'ROTATION_Y', 'ROTATION_Z'})

%% iterate over lines to create elements

% element counter
i_el = 0;

for ln = 1:36
    
    for el = 1:nEle
        
        nd1 = io((ln-1) * (nEle+1) + el);
        nd2 = io((ln-1) * (nEle+1) + el + 1);
        
        % increase element counter
        i_el = i_el + 1;
        
        % create element
        model.addNewElement(elementName, i_el, [nd1, nd2]);
        
    end
end

%% element properties

model.getAllElements.setPropertyValue('DENSITY', 1180)
model.getAllElements.setPropertyValue('YOUNGS_MODULUS', 4.35e9)
model.getAllElements.setPropertyValue('POISSON_RATIO', 0.3)

% truss radius
a = 1e-3;

A = a^2;
I_y = a^4/12;
I_z = a^4/12;
I_t = a^4/7;

model.getAllElements.setPropertyValue('CROSS_SECTION', A)
model.getAllElements.setPropertyValue('IY', I_y)
model.getAllElements.setPropertyValue('IZ', I_z)
model.getAllElements.setPropertyValue('IT', I_t)

%% initialize model

model.initialize();

%% get west, bottom and east nodes for input, support and output

allNodes = model.getAllNodes();

% get X, Y and Z coordinates
X = allNodes.getX();
Z = allNodes.getZ();

% initialize node lists
westNodes = [];
eastNodes = [];
bottomNodes = [];

for i = 1:length(X)
    
    if abs(-X(i) - bx) < 1e-12   % west nodes
        
        westNodes = [westNodes; i];
    end
    
    if abs(X(i) - bx) < 1e-12    % east nodes
        
        eastNodes = [eastNodes; i];
    end
    
    if abs(-Z(i) - bz) < 1e-12   % bottom nodes
        
        bottomNodes = [bottomNodes; i];
    end
end

%% get corresponding Dofs

% initialize
westDofs = zeros(length(westNodes * 6), 1);
eastDofs = zeros(length(eastNodes * 6), 1);
bottomDofs = zeros(length(bottomNodes * 6), 1);

for i = 1:length(westNodes) 
    westDofs( (i-1) * 6 + 1:i*6) = model.getNode(westNodes(i)).getDofArray.getId();
end

for i = 1:length(eastNodes) 
    eastDofs( (i-1) * 6 + 1:i*6) = model.getNode(eastNodes(i)).getDofArray.getId();
end

for i = 1:length(bottomNodes) 
    bottomDofs( (i-1) * 6 + 1:i*6) = model.getNode(bottomNodes(i)).getDofArray.getId();
end

%% assemble system matrices

K = SimpleAssembler.assembleGlobalStiffnessMatrix(model);
M = SimpleAssembler.assembleGlobalMassMatrix(model);

n = size(K,1);

%% input matrix

F = zeros(n, 1);
F(westDofs) = ones(length(westDofs), 1);
F = F./sum(F);

%% output matrix

G = zeros(1, n);
G(eastDofs) = ones(1, length(eastDofs));
G = G./sum(G);

%% support

b = length(bottomDofs);

K(:,bottomDofs) = zeros(n, b);
K(bottomDofs,:) = zeros(b, n);
K(bottomDofs,bottomDofs) = eye(b,b)*mean(diag(K));

M(:,bottomDofs) = zeros(n, b);
M(bottomDofs,:) = zeros(b, n);
M(bottomDofs,bottomDofs) = eye(b,b)*mean(diag(M));

D = 8e-6 * K + 8 * M;
 