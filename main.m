warning('off', 'all');
prefix = 'HR3897';
A = sparse(csvread(strcat(strcat('dataset/deezer_clean_data/top_5000_edges_subset/', prefix), 'Mat.csv')));
W = ones(size(A)) - A;
for d = [50, 100, 200, 400, 800]
    X = dot_rep_large(A, d, W, 1, 100, 1);
    csvwrite(strcat(strcat(strcat(prefix, 'd'), num2str(d)), 'X.csv'), full(X));
end