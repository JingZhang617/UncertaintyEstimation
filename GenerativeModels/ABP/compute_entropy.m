function output = compute_entropy(input)
output = -input.*log(min(input+1e-8,1));
end

