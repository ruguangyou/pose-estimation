function [ ] = matlab2opencv( variable, variableName, fileName, flag )
%MATLAB2OPENCV Save `variable` to yml/xml file 
% fileName: filename where the variable is stored
% flag: `a` for append, `w` for writing.
%   Detailed explanation goes here

[rows, cols] = size(variable);

% Beware of Matlab's linear indexing
variable = variable';

% Write mode as default
if ( ~exist('flag','var') )
    flag = 'w'; 
end

if ( ~exist(fileName,'file') || flag == 'w' )
    % New file or write mode specified 
    file = fopen( fileName, 'w');
    fprintf( file, '%%YAML:1.0\n');
else
    % Append mode
    file = fopen( fileName, 'a');
end

% Write variable header
fprintf( file, '%s: !!opencv-matrix\n', variableName);
fprintf( file, '    rows: %d\n', rows);
fprintf( file, '    cols: %d\n', cols);
fprintf( file, '    dt: d\n');
fprintf( file, '    data: [ ');

% Write variable data
for i=1:rows*cols
    fprintf( file, '%.9f', variable(i));
    if (i == rows*cols), break, end
    fprintf( file, ', ');
    % if mod(i,3) == 0
    %     fprintf( file, '\n        ');
    % end
end

fprintf( file, ']\n');

fclose(file);
end