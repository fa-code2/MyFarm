from model import CropYieldPredictor
import os
import sys

def main():
    print(" MyFarm - Crop Yield Prediction Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "data/crop_yield.csv"
    
    if not os.path.exists(dataset_path):
        print(f" Dataset not found at: {dataset_path}")
        print("\n To train the model:")
        print("1. Download your crop yield dataset from Kaggle")
        print("2. Place it in the 'data' folder as 'crop_yield.csv'")
        print("3. Run this script again: python train_model.py")
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = CropYieldPredictor()
        
        # Train the model
        print(" Starting model training...")
        metrics = predictor.train_model(dataset_path)
        
        # Save the model
        print(" Saving trained model...")
        predictor.save_model("crop_yield_model.pkl")
        
        print("\n Training completed successfully!")
        print("=" * 50)
        print(" Model Performance:")
        print(f"   • RMSE: {metrics['rmse']:.2f}")
        print(f"   • R² Score: {metrics['r2_score']:.3f}")
        
        # Show feature importance
        importance = predictor.get_feature_importance()
        print("\n Most Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
            print(f"   {i}. {feature}: {score:.3f}")
        
        print("\n Our model is ready!")
       
    except Exception as e:
        print(f" Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
