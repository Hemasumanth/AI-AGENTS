import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=1000):
    """Create synthetic dataset with realistic distributions using nullable types"""
    np.random.seed(42)
    
    # Initialize data with proper nullable types
    data = {
        'age': pd.Series(dtype='Int64'),  # Using pandas nullable integer type
        'income': np.random.lognormal(mean=4.5, sigma=0.3, size=n_samples).round(2),
        'education_level': np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            size=n_samples,
            p=[0.3, 0.4, 0.2, 0.1]
        ),
        'purchased': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }
    
    # Generate age data with proper null handling
    age_values = np.random.normal(40, 15, n_samples).astype(int)
    
    # Introduce missing values using pandas.NA for integer nulls
    mask = np.random.rand(n_samples) < 0.1
    age_values = age_values.astype(object)  # Convert to object to hold NA
    age_values[mask] = pd.NA
    
    data['age'] = pd.array(age_values, dtype=pd.Int64Dtype())
    
    # Create DataFrame with explicit dtype specification
    df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
    
    # Convert to proper nullable types
    df = df.astype({
        'age': 'Int64',
        'income': 'float64',
        'education_level': 'category',
        'purchased': 'int64'
    })
    
    return df

# Generate and validate data
try:
    df = generate_synthetic_data()
    print("Data generated successfully with dtypes:")
    print(df.dtypes)
    print("\nSample data with null values:")
    print(df.head(10))
    
    # Save to Excel
    excel_filename = "synthetic_data.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"\nData saved to {excel_filename}")
    
except Exception as e:
    print(f"Error generating data: {str(e)}")