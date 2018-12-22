function X = dot_rep_large(A,d,mask,eps,iter_max,option) 

if option == 1
    display([''; 'd = ' num2str(d)]);
end

addpath('./PROPACK/');
addpath(genpath('./matlab_tools/'));

D = spconvert([[],[],[]; size(A,1),size(A,2),0]);
er = 1e3;
prevError = 1e3;

nz_loc = find(mask);

B = A+D;

check = er>eps;
iter = 0;

while(check)
    iter = iter+1; 
    [V,S] = laneig(B,d,'LM');

    S = max(S,0);    
    
    X = sqrt(S)*V'; 
    
    X(find(abs(X)<1e-7)) = 0; 
    [Ix,Jx,Vx]=find(X); 

    X_s = spconvert([Ix,Jx,Vx; size(X,1),size(X,2),0]);    
    
    C = ssmult(X_s',X_s);

    fA = spconvert([[],[],[]; size(A,1),size(A,2),0]);
    fA(nz_loc) = C(nz_loc);
    fA = fA-A;
    er = sqrt(sum(sum(fA.*fA)));

    if option ==1
          display(['processing iter' num2str(iter) ', error = '  num2str(er)]); 
    end
    B = C - fA;
    
    check = er > eps;
    if iter == iter_max || abs(prevError - er) < 1e-5
        check =0;
    end
    prevError = er;
end 

X = sparse(X);