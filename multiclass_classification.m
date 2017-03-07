clear
clc

%% Generation of clusters

num_of_data = 90;
num_of_outputs = 3;

X = zeros(num_of_data,2);

X = randi(20,num_of_data/3,2);
X = [X;randi([40 60],num_of_data/3,2)];
temp = [randi(20,num_of_data/3,1) randi([40 60],num_of_data/3,1)];
X = [X;temp];
X = [ones(num_of_data,1) X];

y = zeros(num_of_data,num_of_outputs);
y_0 = zeros(num_of_data,1);
y_1 = zeros(num_of_data,1);
y_2 = zeros(num_of_data,1);

y_0(1:num_of_data/3) = 1;
y_1(num_of_data/3+1:num_of_data*2/3)=1;
y_2(num_of_data*2/3+1:end) = 1;

y = [y_0 y_1 y_2];


m = size(X,1);
iterations = 20000;
learning_rate = 0.005;

theta = ones(size(X,2),3);
hypothesis = (1./(1+exp(-X*theta)));

xmin = -20;
xmax = 100;
ymin = -20;
ymax = 100;

figure
axis([xmin, xmax, ymin, ymax])
hold on

scatter(X(1:num_of_data/3,2),X(1:num_of_data/3,3), 'x', 'b');
scatter(X(num_of_data/3+1:num_of_data*2/3,2),X(num_of_data/3+1:num_of_data*2/3,3), 'x','r');   
scatter(X(num_of_data*2/3:end,2),X(num_of_data*2/3:end,3), 'x','m');
     

y_intercept = -theta(1,:)./theta(3,:);
slope = -y_intercept./(-theta(1,:)./theta(2,:));
 
line1 = plot([xmin xmax],[y_intercept(1)+slope(1)*xmin y_intercept(1)+slope(1)*xmax]);
line2 = plot([xmin xmax],[y_intercept(2)+slope(2)*xmin y_intercept(2)+slope(2)*xmax]);
line3 = plot([xmin xmax],[y_intercept(3)+slope(3)*xmin y_intercept(3)+slope(3)*xmax]);

%% Classification
J_hist = zeros(iterations,3);

for j = 1:iterations
    
    %Cost function of logistic regression
    cost = - y.*log(hypothesis)-(1-y).*log(1-hypothesis);
    J_hist(j,:) = (1/m)*sum(cost);
 
    %Gradient descent of the parameters theta
    theta = theta - learning_rate*(1/m)*X'*(hypothesis-y);
    hypothesis = (1./(1+exp(-X*theta)));
    
     %if statement to update graph every 100 iterations of gradient descent
     if (mod(j, 100) == 0)
         
         x1 = [-theta(1,:)./theta(2,:); zeros(1,3)];
         x2 = [zeros(1,3); -theta(1,:)./theta(3,:)];
        
         y_intercept = -theta(1,:)./theta(3,:);
         slope = -y_intercept./(-theta(1,:)./theta(2,:));
         
         delete(line1)
         delete(line2)
         delete(line3)
         
         line1 = plot([xmin xmax],[y_intercept(1)+slope(1)*xmin y_intercept(1)+slope(1)*xmax]);
         line2 = plot([xmin xmax],[y_intercept(2)+slope(2)*xmin y_intercept(2)+slope(2)*xmax]);
         line3 = plot([xmin xmax],[y_intercept(3)+slope(3)*xmin y_intercept(3)+slope(3)*xmax]);
         
         %Pausing to observe classification lines on graph
         pause(0.01);
     
     end
end


%Plot of the cost function vs the number of iterations of gradient descent
figure
plot(1:iterations,J_hist)
xlabel('Number of iterations of gradient descent')
ylabel('J Cost function')


