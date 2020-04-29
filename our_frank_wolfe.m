function  [opt_x, opt_f, fs,  runtime] ...
    = our_frank_wolfe(f, grad, param,max_iter);
% fs:  function value in each iteration
% opt_x: returned solution
% opt_f: returned fun. value
fs = [];
% m = param.m;
n = param.n;
%
% start x_0 = min_{x \in P} ||x||
%one = ones(n,1);
x1 = zeros(n, 1);
x = proj_polytope1(x1, param);
%
%
f_t = f(x, param);
fs = [fs f_t];
iter = 0;
tic;
% grad_t = zeros(n,1);
while iter <= max_iter
    %
    % calculate gradient
    %
    grad_t = grad(x, param); 
    vm = LMO_fw_variant(grad_t, param, x);   % 
    %
    %
    %gamma = min(gamma_cons, 1 -t);
    gamma = 1/(iter+1) ; 
    x = (1-gamma)*x + gamma*vm;
    %
    %t = t+ gamma;
    %
    f_t = f(x, param);
    fs = [fs f_t];
    iter = iter+1;

end
runtime = toc;
opt_x = x;
opt_f = fs(end);
end


function vm = LMO_fw_variant(grad_t, param, x)
% returned vm:  n*1
% x: current solution
% currently using 'interior-point'

% m = param.m;
n = param.n;
lb = param.lb;
ub = param.ub;
%ub_new = ub - x;
b = param.b;
A = param.A;
Aeq = param.Aeq; beq = param.beq;
opts= param.opts;
x0 = [];
% solve lp
vm = linprog(-grad_t,A,b,Aeq,beq,lb,ub,x0, opts);
[s1, s2] =  size(vm);
if n ~= s1 || 1 ~= s2
  vm = zeros(n, 1); % in case of returning NaN solution
end
end

function y  = proj_polytope1(x, param);

lb=param.lb;
ub=param.ub;
A = param.A;
b = param.b;
n = length(ub);

% formulate as QP 
H = eye(n);
h = -x;
opt_quad = optimoptions('quadprog','Display', 'off');
y = quadprog(H, h, A,b, [], [], lb, ub, [], opt_quad);
end