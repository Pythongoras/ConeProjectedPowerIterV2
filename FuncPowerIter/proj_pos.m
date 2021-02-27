% The projection function of the Positive Cone.
function v = proj_pos(v0)
v = zeros(length(v0), 1);
for i = 1:length(v0)
    v(i) = max([0, v0(i)]);
end
end