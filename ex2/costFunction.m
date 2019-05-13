function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

temp = 0
for i = 1:m
	temp = temp + (-1 * y(i) * log (sigmoid(X(i,:)*theta))) - (1 - y(i)) * log (1 - sigmoid(X(i,:)*theta));
end
J = temp / m

%temp1 = theta(1,1)
%temp2 = theta(2,1)
%temp3 = theta(3,1)
%for i = 1:m
%	temp1 = temp1 + (sigmoid(X(i,:)*theta) - y(i)) * X(i, 1);
%	temp2 = temp2 + (sigmoid(X(i,:)*theta) - y(i)) * X(i, 2);
%	temp3 = temp3 + (sigmoid(X(i,:)*theta) - y(i)) * X(i, 3);
%end

%grad = [temp1/m; temp2/m; temp3/m]

grad = X'*(sigmoid(X*theta) - y) / m

% =============================================================

end
