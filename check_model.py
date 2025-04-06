import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Check if outputs directory exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')
    print("Created outputs directory")

# Check if grammar_scorer.pkl exists
if not os.path.exists('outputs/grammar_scorer.pkl'):
    print("Grammar scorer model not found. Creating a dummy model...")
    
    # Create a dummy model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Create dummy data for training
    X = np.random.rand(100, 8)  # 8 features
    y = np.random.rand(100) * 5  # Scores between 0 and 5
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    with open('outputs/grammar_scorer.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Created dummy grammar_scorer.pkl model")
else:
    print("Grammar scorer model found at outputs/grammar_scorer.pkl")

print("Setup complete. You can now run the application with 'python app.py'") 