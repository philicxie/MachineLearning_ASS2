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

for i = 1:m
    tem = -y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
    J = J + tem;
end
J = J/m;

theta_1 = 0;
theta_2 = 0;
theta_3 = 0;
for i = 1:m
       theta_1 = theta_1 + X(i,1)*(sigmoid(X(i,:)*theta)-y(i));
       theta_2 = theta_2 + X(i,2)*(sigmoid(X(i,:)*theta)-y(i));
       theta_3 = theta_3 + X(i,3)*(sigmoid(X(i,:)*theta)-y(i));
end
%theta_1 = sum(X(i,1)*(sigmoid(X(i,:)*theta)-y(i)))/m;
%theta_2 = sum(X(i,2)*(sigmoid(X(i,:)*theta)-y(i)))/m;
%theta_3 = sum(X(i,3)*(sigmoid(X(i,:)*theta)-y(i)))/m;

grad = [theta_1/m;theta_2/m;theta_3/m];







% =============================================================

end
