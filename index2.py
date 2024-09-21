# Simple Linear Regression
import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0  # Slope (coefficient)
        self.b = 0  # Y-intercept (constant)

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        X: Feature variable (input)
        y: Target variable (output)
        """
        n = len(X)  # Number of data points

        for _ in range(self.epochs):
            y_pred = self.m * X + self.b  
            
        
            D_m = (-2/n) * sum(X * (y - y_pred))  
            D_b = (-2/n) * sum(y - y_pred)       
            
            
            self.m = self.m - self.learning_rate * D_m
            self.b = self.b - self.learning_rate * D_b

    def predict(self, X):
        return self.m * X + self.b

# Example:
if __name__ == "__main__":
    # Example dataset
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5, 7, 9, 11, 13], dtype=float)

    # Create and train the model
    model = SimpleLinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Predict values for input
    predictions = model.predict(X)
    
    print(f"Slope (m): {model.m}")
    print(f"Intercept (b): {model.b}")
    print(f"Predicted values: {predictions}")
    print(f"Actual values: {y}")
