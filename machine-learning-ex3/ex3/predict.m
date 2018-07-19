function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ ones(size(X,1), 1) X]; % We add the ones column for the independent term

% Processing hidden layer
activationL2 = sigmoid(X * Theta1'); % This will be a 5000x25 vector (result of 5000x401*401x25)
% Adding ones column for the independent term of second layer.
activationL2 = [ ones(size(activationL2, 1), 1) activationL2 ];
activationL2 = activationL2'; % 26x5000 corresponding to the activation values of the second layer

% Processing output layer
output = sigmoid(Theta2 * activationL2); % 10x5000 vector (10x26 * 26x5000)
[ values, p ] = max(output, [], 1);
p = p';




% =========================================================================


end
