function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



temp1 = 0;
for i = 1:m
        temp1 = temp1 + (-1 * y(i) * log (sigmoid(X(i,:)*theta))) - (1 - y(i)) * log (1 - sigmoid(X(i,:)*theta));
end

temp2 = 0;
for i = 2:size(theta)(1)
	temp2 = temp2 + theta(i, 1)^2;
end

J = (temp1 / m) + ((temp2 * lambda) / (2*m))

grad = (X'*(sigmoid(X*theta) - y) / m) .+ (lambda/m).*theta(2:size(theta)(1),:)


% =============================================================

end
