function save_result(file, res) %#ok<INUSD>
%SAVE_RESULT Write res to file atomically (a crash never leaves a half-written result).
tmp = [file(1:end-4) '.tmp.mat'];
save(tmp, 'res', '-v7');
movefile(tmp, file, 'f');
end
