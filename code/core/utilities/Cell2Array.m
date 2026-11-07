function varargout = Cell2Array(varargin)
%CELL2ARRAY 
%  INPUT: 
%   - varargin: Beliebig viele Cell-Strukturen
%
%  OUTPUT:
%   - varargout: Multidimensionale Arrays (entsprechend der Anzahl der Inputs)

    % Anzahl der Eingaben prüfen
    numInputs = nargin;
    varargout = cell(1, numInputs); % Initialisiere cell für Outputs
    
    % Für jede Eingabe Cell-Struktur in ein Array umwandeln
    for i = 1:numInputs
        A = varargin{i}; % Aktuelles Cell-Array
        nDOF = size(A{1},1);
        mDOF = size(A{1},2);
        numSamples = size(A,2);
        
        % Initialisiere das multidimensionale Array
        A_arr = zeros(nDOF, mDOF, numSamples);
        
        % Transformiere die Cell-Struktur
        for j = 1:numSamples
            A_arr(:,:,j) = A{j};
        end
        
        % Speichere das Ergebnis im Output
        varargout{i} = A_arr;
    end
end

