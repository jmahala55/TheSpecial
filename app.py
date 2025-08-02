from flask import Flask, render_template, request
from google.cloud import bigquery
import os
import xgboost as xgb
import pandas as pd
import numpy as np

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'harvard-baseball-13fab221b2d4.json'

# Initialize BigQuery client
try:
    client = bigquery.Client()
    print("BigQuery client initialized successfully")
except Exception as e:
    print(f"Error initializing BigQuery client: {e}")
    client = None

# Load XGBoost models
def load_xgboost_models():
    """Load the 3 XGBoost models"""
    try:
        xgb_model = xgb.Booster({'nthread': 4})
        xgb_model.load_model('FASTBALL.json')

        xgb_model1 = xgb.Booster({'nthread': 4})
        xgb_model1.load_model('BB.json')

        xgb_model3 = xgb.Booster({'nthread': 4})
        xgb_model3.load_model('SOFT.json')

        model_params = {
            'Fastball': {'model': xgb_model, 'mean_xRV': 0.005168468691408634, 'std_xRV': 0.013001615181565285},
            'Breaking Balls': {'model': xgb_model1, 'mean_xRV': -0.005895022302865982, 'std_xRV': 0.010344094596803188},
            'Offspeed': {'model': xgb_model3, 'mean_xRV': -0.0011278651654720306, 'std_xRV': 0.010134916752576828}
        }
        
        print("XGBoost models loaded successfully")
        return model_params
    except Exception as e:
        print(f"Error loading XGBoost models: {e}")
        return None

model_params = load_xgboost_models()

def categorize_pitch_type(tagged_pitch_type):
    """Convert TaggedPitchType to model categories"""
    pitch_lower = tagged_pitch_type.lower()
    
    if any(word in pitch_lower for word in ['fastball', 'four-seam', '4-seam', 'twoseam', 'two-seam', 'sinker']):
        return 'Fastball'
    elif any(word in pitch_lower for word in ['slider', 'curveball', 'curve', 'cutter', 'sweeper', 'slurve']):
        return 'Breaking Balls'
    elif any(word in pitch_lower for word in ['changeup', 'change-up', 'changup', 'splitter', 'split-finger', 'knuckleball']):
        return 'Offspeed'
    else:
        return 'Fastball'

def safe_float(value, default=0.0):
    """Safely convert value to float, handling None values"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_in_zone(plate_side, plate_height):
    """Calculate if pitch is in strike zone"""
    if plate_side is None or plate_height is None:
        return 0
    
    strike_zone = {
        'xmin': -9.97/12,
        'xmax': 9.97/12,
        'ymin': 18.00/12,
        'ymax': 40.53/12
    }
    
    flipped_plate_side = -1 * safe_float(plate_side)
    plate_height_float = safe_float(plate_height)
    
    return 1 if (strike_zone['xmin'] <= flipped_plate_side <= strike_zone['xmax'] and
                 strike_zone['ymin'] <= plate_height_float <= strike_zone['ymax']) else 0

def predict_stuffplus_scenario(base_pitch_data, fastball_velo, model_category, modifications=None, batter_side='right'):
    """Use XGBoost model to predict Stuff+ for a specific scenario with modifications"""
    if not model_params or model_category not in model_params:
        return None
    
    try:
        model_info = model_params[model_category]
        model_to_use = model_info['model']
        mean_xRV = model_info['mean_xRV']
        std_xRV = model_info['std_xRV']
        
        # Start with base pitch characteristics
        input_data = {
            "RelSpeed": safe_float(base_pitch_data.get('RelSpeed'), 90.0),
            "InducedVertBreak": safe_float(base_pitch_data.get('InducedVertBreak'), 10.0),
            "HorzBreak": safe_float(base_pitch_data.get('HorzBreak'), 0.0),
            "RelHeight": safe_float(base_pitch_data.get('RelHeight'), 6.0),
            "RelSide": safe_float(base_pitch_data.get('RelSide'), 2.0),
            "throws_r": 1 if base_pitch_data.get('PitcherThrows', 'Right') == 'Right' else 0,
            "bats_r": 1 if batter_side == 'right' else 0,
            "year": 2024,
            "in_zone": calculate_in_zone(base_pitch_data.get('PlateLocSide'), base_pitch_data.get('PlateLocHeight')),
            "SEC_Game": 0,
            "Extension": safe_float(base_pitch_data.get('Extension'), 6.5)
        }
        
        # Apply modifications if provided
        if modifications:
            for key, value in modifications.items():
                if key in input_data:
                    input_data[key] = value
        
        # Calculate velo_diff for Breaking Balls and Offspeed after modifications
        if model_category in ['Breaking Balls', 'Offspeed']:
            input_data['velo_diff'] = abs(fastball_velo - input_data['RelSpeed'])
        
        # Create DataFrame and predict using XGBoost
        input_df = pd.DataFrame([input_data])
        dmatrix = xgb.DMatrix(input_df)
        predicted_xRV = float(model_to_use.predict(dmatrix)[0])
        
        # Calculate StuffPlus score using your exact formula
        standardized_xRV = -(predicted_xRV - mean_xRV) / std_xRV
        A = 34
        stuffplus_score = standardized_xRV * A + 100
        
        return round(stuffplus_score, 1)
        
    except Exception as e:
        print(f"Error predicting scenario Stuff+: {e}")
        return None

def get_realistic_improvement_ranges():
    """Get realistic improvement ranges based on what's actually achievable"""
    # These are more realistic ranges for what a pitcher can actually improve
    return {
        'velo_improvement': 2.0,      # +2 mph is realistic with training
        'ivb_improvement': 2.5,       # +2.5 inches IVB is achievable 
        'hb_improvement': 3.0,        # +3 inches HB is reasonable
        'rel_side_improvement': 0.3,  # 0.3 feet = ~3.6 inches is realistic
        'rel_height_improvement': 0.2, # 0.2 feet = ~2.4 inches is doable
        'extension_improvement': 0.15  # 0.15 feet = ~1.8 inches extension gain
    }

def test_improvements_with_xgboost(avg_pitch_data, fastball_velo, model_category, current_stuffplus):
    """Test various improvements using XGBoost models to see actual impact on Stuff+"""
    if current_stuffplus is None:
        return []
    
    improvement_ranges = get_realistic_improvement_ranges()
    recommendations = []
    
    # Get pitcher handedness for release side direction guidance
    pitcher_throws = avg_pitch_data.get('PitcherThrows', 'Right')
    
    # Define improvement scenarios to test with XGBoost
    improvement_scenarios = []
    
    # Different options to achieve the best version of your pitch
    improvement_scenarios.extend([
        {
            'name': 'Less Armside Run',
            'modifications': {'HorzBreak': avg_pitch_data['HorzBreak'] - improvement_ranges['hb_improvement']},
            'description': 'Less armside run (HB)'
        },
        {
            'name': 'Added Velocity with Carry',
            'modifications': {
                'RelSpeed': avg_pitch_data['RelSpeed'] + (improvement_ranges['velo_improvement'] * 0.7),
                'InducedVertBreak': avg_pitch_data['InducedVertBreak'] + (improvement_ranges['ivb_improvement'] * 0.7)
            },
            'description': 'Added velocity with a little bit more carry/ride (IVB)'
        },
        {
            'name': 'Added Carry',
            'modifications': {'InducedVertBreak': avg_pitch_data['InducedVertBreak'] + improvement_ranges['ivb_improvement']},
            'description': 'Added ride/carry (IVB)'
        },
        {
            'name': 'Add Velocity',
            'modifications': {'RelSpeed': avg_pitch_data['RelSpeed'] + improvement_ranges['velo_improvement']},
            'description': 'Add velocity'
        },
            {
        'name': 'More Armside Run',
        'modifications': {'HorzBreak': avg_pitch_data['HorzBreak'] + improvement_ranges['hb_improvement']},
        'description': 'More armside run (HB)'
    },
    {
        'name': 'Added Sink',
        'modifications': {'InducedVertBreak': avg_pitch_data['InducedVertBreak'] - improvement_ranges['ivb_improvement']},
        'description': 'Reduce ride to get more sink'
    }
])
    
    # Add pitch-type specific scenarios for Fastball
    if model_category == 'Fastball':
        improvement_scenarios.extend([
            {
                'name': 'Lower Release Height',
                'modifications': {'RelHeight': avg_pitch_data['RelHeight'] - improvement_ranges['rel_height_improvement']},
                'description': 'Drop your release point a little lower.'
            },
            {
                'name': 'Higher Release Height',
                'modifications': {'RelHeight': avg_pitch_data['RelHeight'] + improvement_ranges['rel_height_improvement']},
                'description': 'Raise your release point a little higher.'
            },
            {
                'name': 'High Velocity + Release Point',
                'modifications': {
                    'RelSpeed': avg_pitch_data['RelSpeed'] + improvement_ranges['velo_improvement'],
                    'RelSide': avg_pitch_data['RelSide'] + (improvement_ranges['rel_side_improvement'] * 0.5)
                },
                'description': 'Throw faster and shift your release toward 3rd base.'
            }
        ])
    if model_category in ['Fastball', 'Breaking Balls']:
        shift_value = improvement_ranges['rel_side_improvement']
        direction = 1 if pitcher_throws == 'Right' else -1
        improvement_scenarios.append({
            'name': 'Shifted Release Side',
            'modifications': {'RelSide': avg_pitch_data['RelSide'] + (direction * shift_value)},
            'description': 'Shifted release toward arm side'
    })
    
    elif model_category == 'Offspeed':
        current_velo_diff = abs(fastball_velo - avg_pitch_data['RelSpeed'])
        improvement_scenarios.extend([
            {
                'name': 'More Velocity Separation',
                'modifications': {'RelSpeed': fastball_velo - (current_velo_diff + 2)},
                'description': 'Slow it down a lot more than your fastball.'
            }
        ])
    
    # Test each scenario using XGBoost models
    print(f"Testing {len(improvement_scenarios)} improvement scenarios for {model_category}...")
    
    for scenario in improvement_scenarios:
        # Use XGBoost to predict new Stuff+ with modifications
        new_stuffplus = predict_stuffplus_scenario(
            avg_pitch_data, 
            fastball_velo, 
            model_category, 
            scenario['modifications']
        )
        
        if new_stuffplus is not None:
            gain = new_stuffplus - current_stuffplus
            
            # Only include improvements that actually help
            if gain > 0.3:  # At least 0.3 Stuff+ improvement
                
                # Add release point guidance based on pitcher handedness and pitch type
                description = scenario['description']
                if 'release' in scenario['name'].lower() and pitcher_throws:
                    if model_category == 'Breaking Balls':
                        if pitcher_throws == 'Right':
                            description += " (try toward 1st base for breaking balls)"
                        else:
                            description += " (try toward 3rd base for breaking balls)"
                    elif model_category == 'Fastball':
                        description += " (try both directions to mix it up)"
                
                recommendations.append({
                    'improvement': scenario['name'],
                    'description': description,
                    'modifications': scenario['modifications'],
                    'current_stuffplus': current_stuffplus,
                    'projected_stuffplus': new_stuffplus,
                    'gain': round(gain, 1),
                    'tested_with_xgboost': True
                })
                
                print(f"  {scenario['name']}: {current_stuffplus:.1f} → {new_stuffplus:.1f} (+{gain:.1f})")
    
    # Sort by potential gain and return top recommendations
    recommendations.sort(key=lambda x: x['gain'], reverse=True)
    return recommendations[:5]  # Top 5 recommendations

def predict_stuffplus(pitch_data, fastball_velo, model_category, batter_side='right'):
    """Calculate average Stuff+ from multiple pitches"""
    if not model_params or model_category not in model_params:
        return None
    
    try:
        predictions = []
        
        for pitch in pitch_data:
            # Use the scenario function with no modifications
            prediction = predict_stuffplus_scenario(pitch, fastball_velo, model_category, None, batter_side)
            if prediction is not None:
                predictions.append(prediction)
        
        return predictions if predictions else None
        
    except Exception as e:
        print(f"Error predicting Stuff+ for {model_category}: {e}")
        return None

@app.route('/')
def index():
    if not client:
        return "BigQuery client not initialized"

    selected_pitcher = request.args.get('pitcher', default=None)
    
    try:
        query = """
        SELECT 
            Pitcher,
            TaggedPitchType,
            PitcherThrows,
            RelSpeed,
            SpinRate,
            InducedVertBreak,
            HorzBreak,
            RelHeight,
            RelSide,
            Extension,
            PlateLocSide,
            PlateLocHeight
        FROM `V1PBR.Test`
        WHERE Pitcher IS NOT NULL 
        AND TaggedPitchType IS NOT NULL
        AND RelSpeed IS NOT NULL
        AND SpinRate IS NOT NULL
        AND InducedVertBreak IS NOT NULL
        AND HorzBreak IS NOT NULL
        ORDER BY Pitcher, TaggedPitchType
        """



# Execute query and parse into list of dicts
        result = client.query(query)
        raw_data = [dict(row) for row in result]

# Collect unique pitcher names for the dropdown
        unique_pitchers = sorted(set(row['Pitcher'] for row in raw_data))

# Filter by selected pitcher, if applicable
        if selected_pitcher:
            raw_data = [row for row in raw_data if row['Pitcher'] == selected_pitcher]
        
        # Group by pitcher and pitch type
        pitcher_data = {}
        pitcher_fastball_velos = {}
        
        for row in raw_data:
            pitcher = row['Pitcher']
            pitch_type = row['TaggedPitchType']
            key = (pitcher, pitch_type)
            
            if key not in pitcher_data:
                pitcher_data[key] = []
            pitcher_data[key].append(row)
            
            # Track fastball velocities
            if categorize_pitch_type(pitch_type) == 'Fastball':
                if pitcher not in pitcher_fastball_velos:
                    pitcher_fastball_velos[pitcher] = []
                pitcher_fastball_velos[pitcher].append(safe_float(row['RelSpeed'], 90.0))
        
        # Calculate average fastball velocity for each pitcher
        avg_fastball_velos = {}
        for pitcher, velos in pitcher_fastball_velos.items():
            avg_fastball_velos[pitcher] = np.mean(velos) if velos else 92.0
        
        # Create table data with XGBoost-tested recommendations
        table_data = []
        
        for (pitcher, pitch_type), pitches in pitcher_data.items():
            if len(pitches) < 3:  # Skip if less than 3 pitches
                continue
            
            model_category = categorize_pitch_type(pitch_type)
            fastball_velo = avg_fastball_velos.get(pitcher, 92.0)
            
            print(f"Processing: {pitcher} - {pitch_type} ({model_category})")
            
            # Calculate average pitch characteristics
            avg_pitch_data = {
                'RelSpeed': np.mean([safe_float(p.get('RelSpeed')) for p in pitches]),
                'InducedVertBreak': np.mean([safe_float(p.get('InducedVertBreak')) for p in pitches]),
                'HorzBreak': np.mean([safe_float(p.get('HorzBreak')) for p in pitches]),
                'RelHeight': np.mean([safe_float(p.get('RelHeight')) for p in pitches]),
                'RelSide': np.mean([safe_float(p.get('RelSide')) for p in pitches]),
                'Extension': np.mean([safe_float(p.get('Extension')) for p in pitches]),
                'PitcherThrows': pitches[0].get('PitcherThrows', 'Right'),
                'PlateLocSide': np.mean([safe_float(p.get('PlateLocSide')) for p in pitches]),
                'PlateLocHeight': np.mean([safe_float(p.get('PlateLocHeight')) for p in pitches])
            }
            
            base_data = {
                'pitcher': pitcher,
                'pitch_type': pitch_type,
                'count': len(pitches),
                'velocity': round(avg_pitch_data['RelSpeed'], 1),
                'spin_rate': round(np.mean([safe_float(p.get('SpinRate')) for p in pitches]), 0),
                'ivb': round(avg_pitch_data['InducedVertBreak'], 1),
                'hb': round(avg_pitch_data['HorzBreak'], 1),
            }
            
            # Get current Stuff+ using XGBoost
            current_stuffplus = None
            if model_params:
                predictions = predict_stuffplus(pitches, fastball_velo, model_category, 'right')
                if predictions:
                    current_stuffplus = round(np.mean(predictions), 1)
                    base_data['stuffplus'] = current_stuffplus
                else:
                    base_data['stuffplus'] = 'N/A'
            else:
                base_data['stuffplus'] = 'No Model'
            
            # Generate XGBoost-based improvement recommendations
            if current_stuffplus is not None:
                recommendations = test_improvements_with_xgboost(
                    avg_pitch_data, fastball_velo, model_category, current_stuffplus
                )
                base_data['recommendations'] = recommendations
                print(f"  Generated {len(recommendations)} recommendations")
            else:
                base_data['recommendations'] = []
            
            table_data.append(base_data)
        
        # Sort by Stuff+ (highest first)
        table_data.sort(key=lambda x: x['stuffplus'] if isinstance(x['stuffplus'], (int, float)) else -999, reverse=True)
        

        return render_template('improvements_table.html',
                            data=table_data,
                            models_loaded=model_params is not None,
                            unique_pitchers=unique_pitchers,
                            selected_pitcher=selected_pitcher
)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print("Starting XGBoost-based improvement recommendation server...")
    print("Available at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)