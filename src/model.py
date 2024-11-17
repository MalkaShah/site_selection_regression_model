import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import folium
from folium import plugins
import joblib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

class SiteSelectionModel:
    def __init__(self):
        """Initialize the Site Selection Model"""
        self._setup_logging()
        self._setup_directories()
        
        # Initialize components
        self.scaler = MinMaxScaler()
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Initialize data containers
        self.site_data = None
        self.demographic_data = None
        self.enhanced_data = None
        
        # Define scoring weights
        self.weights = {
            'traffic_score': 0.3,
            'population_density': 0.2,
            'competition_score': 0.15,
            'accessibility_score': 0.2,
            'income_level': 0.15
        }
        
        self.logger.info("Site Selection Model initialized successfully")

    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'site_selection_{datetime.now().strftime("%Y%m%d")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_directories(self):
        """Create necessary project directories"""
        directories = ['data/raw', 'data/processed', 'output', 'logs', 'models']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_data(self, sites_file, demographics_file):
        """Load site and demographic data"""
        try:
            # Load sites data
            with open(sites_file, 'r') as f:
                sites_data = json.load(f)
            
            # Convert GeoJSON to DataFrame
            sites_list = []
            for feature in sites_data['features']:
                site = feature['properties']
                site['latitude'] = feature['geometry']['coordinates'][1]
                site['longitude'] = feature['geometry']['coordinates'][0]
                sites_list.append(site)
            
            self.site_data = pd.DataFrame(sites_list)
            
            # Load demographics
            self.demographic_data = pd.read_csv(demographics_file)
            
            self.logger.info("Data loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False

    def preprocess_data(self):
        """Preprocess and normalize data"""
        try:
            if self.site_data is None:
                raise ValueError("No data loaded")
            
            # Remove missing values
            self.site_data = self.site_data.dropna()
            
            # Normalize numerical columns
            numerical_cols = ['population_density', 'avg_income', 'traffic_volume']
            for col in numerical_cols:
                if col in self.site_data.columns:
                    self.site_data[f'{col}_normalized'] = self.scaler.fit_transform(
                        self.site_data[[col]]
                    )
            
            self.logger.info("Data preprocessing completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return False

    def calculate_site_scores(self):
        """Calculate comprehensive site scores"""
        try:
            self.site_data['score'] = (
                self.weights['traffic_score'] * self.site_data['traffic_volume_normalized'] +
                self.weights['population_density'] * self.site_data['population_density_normalized'] +
                self.weights['income_level'] * self.site_data['avg_income_normalized']
            )
            
            self.logger.info("Site scores calculated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error calculating scores: {str(e)}")
            return False

    def train_ml_model(self):
        """Train machine learning model for site prediction"""
        try:
            # Prepare features and target
            features = ['population_density_normalized', 
                       'avg_income_normalized', 
                       'traffic_volume_normalized']
            
            X = self.site_data[features]
            y = self.site_data['score']
            
            # Train model
            self.ml_model.fit(X, y)
            
            # Save model
            joblib.dump(self.ml_model, 'models/site_predictor.joblib')
            
            self.logger.info("ML model trained and saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML model: {str(e)}")
            return False

    def create_visualizations(self):
        """Create various visualizations"""
        try:
            # Create base map
            center_lat = self.site_data['latitude'].mean()
            center_lon = self.site_data['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
            
            # Add markers
            for _, row in self.site_data.iterrows():
                popup_content = f"""
                    <b>City:</b> {row['city']}<br>
                    <b>Score:</b> {row['score']:.2f}<br>
                    <b>Population Density:</b> {row['population_density']}<br>
                    <b>Average Income:</b> {row['avg_income']}<br>
                    <b>Traffic Volume:</b> {row['traffic_volume']}
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=10,
                    popup=folium.Popup(popup_content, max_width=300),
                    color='red',
                    fill=True
                ).add_to(m)
            
            # Add heatmap layer
            heat_data = [[row['latitude'], row['longitude'], row['score']] 
                        for _, row in self.site_data.iterrows()]
            plugins.HeatMap(heat_data).add_to(m)
            
            # Save maps
            m.save('output/site_analysis.html')
            
            self.logger.info("Visualizations created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return False

    def export_results(self):
        """Export analysis results"""
        try:
            if self.site_data is not None:
                self.site_data.to_csv('output/site_analysis_results.csv', index=False)
                self.logger.info("Results exported successfully")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            return False

def create_sample_data():
    """Create sample data for testing"""
    try:
        # Create sample sites data
        sample_sites = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "city": "Kuala Lumpur",
                        "population_density": 7500,
                        "avg_income": 45000,
                        "traffic_volume": 8500
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [101.6869, 3.1390]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {
                        "city": "Bangkok",
                        "population_density": 8200,
                        "avg_income": 42000,
                        "traffic_volume": 9000
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [100.5018, 13.7563]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {
                        "city": "Jakarta",
                        "population_density": 9100,
                        "avg_income": 38000,
                        "traffic_volume": 8800
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [106.8456, -6.2088]
                    }
                }
            ]
        }
        
        # Save GeoJSON data
        with open('data/raw/sample_sites.geojson', 'w') as f:
            json.dump(sample_sites, f)
        
        # Create sample demographics data
        demographics_data = {
            'city': ['Kuala Lumpur', 'Bangkok', 'Jakarta'],
            'population': [1800000, 8280000, 10500000],
            'median_income': [45000, 42000, 38000],
            'avg_age': [32, 34, 30],
            'households': [450000, 2900000, 3100000]
        }
        
        # Save demographics data
        pd.DataFrame(demographics_data).to_csv('data/raw/demographics.csv', index=False)
        
        return True
        
    except Exception as e:
        print(f"Error creating sample data: {str(e)}")
        return False

def main():
    """Main execution function"""
    try:
        print("\nStarting Site Selection Model Analysis...\n")
        
        # Phase 1: Initial Setup
        print("Phase 1: Initial Setup")
        print("-" * 50)
        
        # Create sample data if needed
        print("Creating sample data...")
        if create_sample_data():
            print("✓ Sample data created successfully")
        else:
            raise Exception("Failed to create sample data")
        
        # Initialize model
        print("\nInitializing model...")
        model = SiteSelectionModel()
        print("✓ Model initialized successfully")
        
        # Phase 2: Data Processing
        print("\nPhase 2: Data Processing")
        print("-" * 50)
        
        # Load data
        print("Loading data...")
        if not model.load_data('data/raw/sample_sites.geojson', 'data/raw/demographics.csv'):
            raise Exception("Failed to load data")
        print("✓ Data loaded successfully")
        
        # Preprocess data
        print("\nPreprocessing data...")
        if not model.preprocess_data():
            raise Exception("Failed to preprocess data")
        print("✓ Data preprocessed successfully")
        
        # Calculate scores
        print("\nCalculating site scores...")
        if not model.calculate_site_scores():
            raise Exception("Failed to calculate scores")
        print("✓ Site scores calculated successfully")
        
        # Phase 3: Advanced Analysis
        print("\nPhase 3: Advanced Analysis")
        print("-" * 50)
        
        # Train ML model
        print("Training machine learning model...")
        if not model.train_ml_model():
            raise Exception("Failed to train ML model")
        print("✓ ML model trained successfully")
        
        # Create visualizations
        print("\nCreating visualizations...")
        if not model.create_visualizations():
            raise Exception("Failed to create visualizations")
        print("✓ Visualizations created successfully")
        
        # Export results
        print("\nExporting final results...")
        if not model.export_results():
            raise Exception("Failed to export results")
        print("✓ Results exported successfully")
        
        print("\n✅ All phases completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()