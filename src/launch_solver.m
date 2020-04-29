function [x_opt, opt_f, fs, runtime] = launch_solver(f, grad, param, method, max_iter);

switch method
    
    case 4, % nonmonotone_fw_variant
        [x_opt, opt_f, fs, runtime] ...
           = nonmonotone_fw_variant(f, grad, param,max_iter);
        
    case 2, % quadprogIP
        [x_opt, opt_f, fs, runtime] ...
            = quadprog_ip(f, grad, param,max_iter);
        
    case 5, % twophase FW
        [x_opt, opt_f, fs, runtime] ... 
            = twophase_fw(f, grad, param, max_iter);  

    case 1, %our algorithm frank-wolfe
        %
        [x_opt, opt_f, fs, runtime] ...
          = our_frank_wolfe(f, grad, param, max_iter);
    
    case 3, %our algorithm proj-gradient
        %
        [x_opt, opt_f, fs, runtime] ...
          = our_proj_grad(f, grad, param, max_iter);
        
end
end
