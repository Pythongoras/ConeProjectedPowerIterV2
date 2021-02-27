function v = mnt_cone_eigenvec_sparse01(p)
cardi = int32(10*log(p));
% cardi = 0.3*p;
v = [zeros(p-cardi,1); ones(cardi,1)];
v = v / sum(v.^2)^0.5;
% This shuffle is used for pos cone.
v = v(randperm(length(v)));
end


% test 
% mnt_cone_eigenvec1(9)