function [newSample, info] = distance_check(newSample, S, d_min, maxit, lb, ub)
%DISTANCE_CHECK Move a candidate away from its nearest sample if closer than d_min.
%   Paper Algorithm 3, line 10:
%   the candidate is placed at distance it_count*(d_min + 1e-6) from its
%   nearest neighbour along the connecting direction (random direction if
%   both coincide), clipped to the bounds, for at most maxit-1 attempts.
numParams = numel(newSample);
distances = sqrt(sum((S - newSample).^2, 2));
[min_dist, idx_closest] = min(distances);
info.min_dist_before = min_dist;
info.moved = false;
it_count = 1;

while (min_dist < d_min) && (maxit > it_count)
    direction = newSample - S(idx_closest, :);
    if norm(direction) == 0
        direction = rand(1, numParams) - 0.5;
    end
    direction = direction / norm(direction);
    newSample = S(idx_closest, :) + direction * (d_min + 1e-6) * it_count;

    for i = 1:numParams
        if newSample(i) < lb(i)
            newSample(i) = lb(i) + 1e-6;
        elseif newSample(i) > ub(i)
            newSample(i) = ub(i) - 1e-6;
        end
    end

    distances = sqrt(sum((S - newSample).^2, 2));
    [min_dist, idx_closest] = min(distances);
    it_count = it_count + 1;
    info.moved = true;
end
info.attempts = it_count - 1;
info.min_dist_after = min_dist;
end
