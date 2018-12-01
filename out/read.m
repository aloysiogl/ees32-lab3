function out = read(name)
fileID = fopen(name, 'r');
formatSpec = '%lf';
out = fscanf(fileID,formatSpec);