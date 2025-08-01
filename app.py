from flask import Flask, render_template, jsonify, request
from google.cloud import bigquery
import os
import json
from datetime import datetime
import io
import concurrent.futures
from functools import lru_cache
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import atexit

# NEW IMPORTS FOR WEASYPRINT
import weasyprint
from jinja2 import Template
# IMPORT THE NEW EMAIL SERVICE
from email_service import (
    EmailConfig, 
    BulkEmailProcessor, 
    ProductionEmailSender,
    test_email_connection,
    estimate_send_time
)

# Add this to the top of your app.py file with other imports
import subprocess
import sys
import time

def run_stuffplus_analysis():
    """Run the Stuffplus analysis script once before generating PDFs"""
    try:
        print("Running Stuffplus analysis to update database...")
        
        # Run the stuffplus_analysis.py script
        result = subprocess.run([
            sys.executable, 'stuffplus_analysis.py'
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✓ Stuffplus analysis completed successfully")
            if result.stdout:
                print("Analysis output:", result.stdout[-500:])  # Show last 500 chars
            return True
        else:
            print(f"✗ Stuffplus analysis failed with return code: {result.returncode}")
            if result.stderr:
                print("Error output:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Stuffplus analysis timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"✗ Error running Stuffplus analysis: {str(e)}")
        return False

def run_stuffplus_analysis():
    """Run the Stuffplus analysis script once before generating PDFs"""
    try:
        print("Running Stuffplus analysis to update database...")
        
        # Run the stuffplus_analysis.py script with better error capture
        result = subprocess.run([
            sys.executable, 'stuffplus_analysis.py'
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        print(f"Stuffplus analysis return code: {result.returncode}")
        
        if result.stdout:
            print("=== STDOUT ===")
            print(result.stdout)
            print("=== END STDOUT ===")
        
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
            print("=== END STDERR ===")
        
        if result.returncode == 0:
            print("✓ Stuffplus analysis completed successfully")
            return True
        else:
            print(f"✗ Stuffplus analysis failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Stuffplus analysis timed out after 10 minutes")
        return False
    except FileNotFoundError:
        print("✗ stuffplus_analysis.py file not found")
        return False
    except Exception as e:
        print(f"✗ Error running Stuffplus analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


app = Flask(__name__)

email_config = EmailConfig()

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'harvard-baseball-13fab221b2d4.json'

# Initialize BigQuery client
try:
    client = bigquery.Client()
    print("BigQuery client initialized successfully")
except Exception as e:
    print(f"Error initializing BigQuery client: {e}")
    client = None



import json
import os
from datetime import datetime

# Simple file-based storage for email status (we'll upgrade to database later)
EMAIL_STATUS_FILE = 'email_status.json'

def load_email_status():
    """Load email status from file"""
    try:
        if os.path.exists(EMAIL_STATUS_FILE):
            with open(EMAIL_STATUS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading email status: {e}")
        return {}

def save_email_status(status_data):
    """Save email status to file"""
    try:
        with open(EMAIL_STATUS_FILE, 'w') as f:
            json.dump(status_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving email status: {e}")
        return False

def update_email_status(prospect_name, date, status, email=None):
    """Update email status for a prospect"""
    status_data = load_email_status()
    
    # Create date key if it doesn't exist
    if date not in status_data:
        status_data[date] = {}
    
    # Update prospect status
    status_data[date][prospect_name] = {
        'status': status,  # 'pending', 'sent', 'failed'
        'email': email,
        'last_sent': datetime.now().isoformat() if status == 'sent' else None,
        'updated_at': datetime.now().isoformat()
    }
    
    save_email_status(status_data)
    return True

# 2. Add this NEW API endpoint to get email status
@app.route('/api/email-status')
def get_email_status():
    """Get email status for all prospects on a specific date"""
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter required'}), 400
    
    try:
        status_data = load_email_status()
        date_status = status_data.get(selected_date, {})
        
        return jsonify({
            'status': date_status,
            'date': selected_date,
            'total_prospects': len(date_status)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validate-matches')
def validate_matches():
    """Validate which prospects have matching pitch data"""
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter required'}), 400
    
    try:
        print(f"Validating matches for date: {selected_date}")
        
        # Get all prospects for the date from Info table
        prospects_query = """
        SELECT DISTINCT Prospect as name
        FROM `V1PBRInfo.Info`
        WHERE Prospect IS NOT NULL
        ORDER BY Prospect
        """
        
        prospects_result = client.query(prospects_query)
        prospect_names = set([row.name for row in prospects_result])
        print(f"Found {len(prospect_names)} prospects in Info table")
        
        # Get all pitchers with data for the selected date from Test table
        pitchers_query = """
        SELECT DISTINCT Pitcher
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher IS NOT NULL
        ORDER BY Pitcher
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
            ]
        )
        
        pitchers_result = client.query(pitchers_query, job_config=job_config)
        pitcher_names = set([row.Pitcher for row in pitchers_result])
        print(f"Found {len(pitcher_names)} pitchers with data for {selected_date}")
        
        # Compare and categorize
        matched = list(prospect_names.intersection(pitcher_names))
        no_match = list(prospect_names - pitcher_names)
        extra_pitchers = list(pitcher_names - prospect_names)  # Pitchers with data but not in prospects
        
        print(f"Matched: {len(matched)}, No match: {len(no_match)}, Extra pitchers: {len(extra_pitchers)}")
        
        return jsonify({
            'matched': matched,
            'no_match': no_match,
            'extra_pitchers': extra_pitchers,
            'total_prospects': len(prospect_names),
            'total_pitchers': len(pitcher_names),
            'match_rate': len(matched) / len(prospect_names) * 100 if prospect_names else 0
        })
        
    except Exception as e:
        print(f"Error validating matches: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ADD: At the top with other imports
pdf_cache = {}  # Simple in-memory PDF cache
college_data_cache = {}  # Cache for college comparison data


# ADD: Batch college data loading
def preload_college_data_batch(comparison_levels, pitch_types, pitcher_throws_set):
    """Pre-load all college data combinations to avoid repeated queries"""
    global college_data_cache
    
    print(f"Pre-loading college data for {len(comparison_levels)} levels...")
    
    for level in comparison_levels:
        for pitch_type in pitch_types:
            for pitcher_throws in pitcher_throws_set:
                cache_key = f"{pitch_type}_{level}_{pitcher_throws}"
                
                if cache_key not in college_data_cache:
                    college_data_cache[cache_key] = {
                        'averages': get_college_averages(pitch_type, level, pitcher_throws),
                        'percentiles': get_college_percentile_data(pitch_type, level, pitcher_throws),
                        'max_velo': get_college_max_velocity_percentile_data(pitch_type, level, pitcher_throws)
                    }
    
    print(f"Pre-loaded {len(college_data_cache)} college data combinations")



# ADD: Cache-aware college data functions
def get_college_averages_cached(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get college averages from cache if available"""
    cache_key = f"{pitch_type}_{comparison_level}_{pitcher_throws}"
    
    if cache_key in college_data_cache:
        return college_data_cache[cache_key]['averages']
    else:
        # Fallback to original function
        return get_college_averages(pitch_type, comparison_level, pitcher_throws)

def get_college_percentile_data_cached(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get percentile data from cache if available"""
    cache_key = f"{pitch_type}_{comparison_level}_{pitcher_throws}"
    
    if cache_key in college_data_cache:
        return college_data_cache[cache_key]['percentiles']
    else:
        return get_college_percentile_data(pitch_type, comparison_level, pitcher_throws)

def get_college_max_velocity_percentile_data_cached(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get max velocity data from cache if available"""
    cache_key = f"{pitch_type}_{comparison_level}_{pitcher_throws}"
    
    if cache_key in college_data_cache:
        return college_data_cache[cache_key]['max_velo']
    else:
        return get_college_max_velocity_percentile_data(pitch_type, comparison_level, pitcher_throws)

@app.route('/api/send-bulk-emails-production', methods=['POST'])
def send_bulk_emails_production():
    """PRODUCTION: Ultra-fast bulk email sending for 300+ reports"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    try:
        start_time = time.time()
        data = request.get_json()
        selected_date = data.get('date')
        prospect_names = data.get('prospect_names', [])  # Handle both formats
        
        if not selected_date:
            return jsonify({'error': 'Date is required'}), 400
        
        if not prospect_names:
            return jsonify({'error': 'No prospects specified'}), 400
        
        print(f"=== PRODUCTION BULK EMAIL START: {selected_date} ===")
        print(f"Processing {len(prospect_names)} prospects")
        
        # Test email connection first
        if not test_email_connection(email_config):
            return jsonify({'error': 'Email connection test failed'}), 500
        
        # Step 1: Run Stuffplus analysis
        print("Step 1: Running Stuffplus analysis...")
        stuffplus_success = run_stuffplus_analysis()
        if not stuffplus_success:
            return jsonify({'error': 'Stuffplus analysis failed'}), 500
        
        # Step 2: Get all data
        print("Step 2: Loading pitcher and prospect data...")
        pitcher_data_map = get_all_pitcher_data_batch(selected_date)
        if not pitcher_data_map:
            return jsonify({'error': 'No pitcher data found'}), 400
        
        # Get prospects that match our criteria
        prospects_query = """
        SELECT Event, Prospect, Email, Type, Comp
        FROM `V1PBRInfo.Info`
        WHERE Prospect IS NOT NULL
        AND Type = 'Pitching'
        AND Email IS NOT NULL
        ORDER BY Prospect
        """
        
        prospects_result = client.query(prospects_query)
        
        # Step 3: Pre-generate all PDFs for matching prospects
        print("Step 3: Pre-generating PDFs...")
        email_tasks = []
        
        pdf_generation_start = time.time()
        
        # Filter to only requested prospects that have data
        requested_prospects = []
        for row in prospects_result:
            if row.Prospect in prospect_names and row.Prospect in pitcher_data_map:
                requested_prospects.append(row)
        
        print(f"Found {len(requested_prospects)} prospects with matching data")
        
        # Use ThreadPoolExecutor for PDF generation
        with ThreadPoolExecutor(max_workers=4) as pdf_executor:
            pdf_futures = {}
            
            for row in requested_prospects:
                # Submit PDF generation task
                future = pdf_executor.submit(
                    generate_pitcher_pdf,
                    row.Prospect,
                    pitcher_data_map[row.Prospect],
                    selected_date,
                    row.Comp or 'D1'
                )
                pdf_futures[future] = {
                    'pitcher_name': row.Prospect,
                    'email': row.Email,
                    'pitch_data': pitcher_data_map[row.Prospect],
                    'date': selected_date,
                    'comparison_level': row.Comp or 'D1'
                }
            
            # Collect PDF results
            for future in as_completed(pdf_futures):
                task_info = pdf_futures[future]
                try:
                    pdf_data = future.result()
                    if pdf_data:
                        task_info['pdf_data'] = pdf_data
                        email_tasks.append(task_info)
                    else:
                        print(f"Failed to generate PDF for {task_info['pitcher_name']}")
                except Exception as e:
                    print(f"PDF generation error for {task_info['pitcher_name']}: {e}")
        
        pdf_generation_time = time.time() - pdf_generation_start
        print(f"PDF generation completed in {pdf_generation_time:.1f}s for {len(email_tasks)} reports")
        
        if not email_tasks:
            return jsonify({'error': 'No PDFs could be generated'}), 400
        
        # Step 4: Send emails using production email service
        print(f"Step 4: Sending {len(email_tasks)} emails...")
        
        # Estimate time
        estimated_time = estimate_send_time(len(email_tasks), email_config)
        print(f"Estimated send time: {estimated_time}")
        
        processor = BulkEmailProcessor(email_config, max_workers=8)
        results = processor.process_bulk_emails(email_tasks)
        
        # Calculate final stats
        sent_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - sent_count
        elapsed_time = time.time() - start_time
        rate = (sent_count / elapsed_time * 60) if elapsed_time > 0 else 0
        
        print(f"=== PRODUCTION BULK EMAIL COMPLETE ===")
        print(f"Total: {len(email_tasks)}, Sent: {sent_count}, Failed: {failed_count}")
        print(f"Time: {elapsed_time:.1f}s, Rate: {rate:.1f} emails/min")
        
        # Update email status for all sent emails
        for result in results:
            status = 'sent' if result['success'] else 'failed'
            update_email_status(
                result['pitcher_name'], 
                selected_date, 
                status, 
                result['email']
            )
        
        return jsonify({
            'success': True,
            'queued_count': len(email_tasks),  # For compatibility with frontend
            'task_ids': ['bulk_email_task'],   # For compatibility with frontend
            'summary': {
                'total_emails': len(email_tasks),
                'sent_successfully': sent_count,
                'failed': failed_count,
                'elapsed_time_seconds': elapsed_time,
                'emails_per_minute': rate,
                'pdf_generation_time': pdf_generation_time,
                'estimated_vs_actual': f"Estimated: {estimated_time}, Actual: {elapsed_time:.1f}s"
            },
            'results': results[:10],  # First 10 for preview
            'full_results_available': len(results) > 10
        })
        
    except Exception as e:
        print(f"Production bulk email error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-individual-email-fast', methods=['POST'])
def send_individual_email_fast():
    """Fast individual email using production email service"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    try:
        data = request.get_json()
        selected_date = data.get('date')
        pitcher_name = data.get('pitcher_name')
        pitcher_email = data.get('pitcher_email')
        
        if not selected_date or not pitcher_name or not pitcher_email:
            return jsonify({'error': 'Date, pitcher name, and email are required'}), 400
        
        print(f"Sending individual email to {pitcher_name} at {pitcher_email}")
        
        # Update status to 'sending'
        update_email_status(pitcher_name, selected_date, 'sending', pitcher_email)
        
        # Get pitcher data
        pitcher_data_query = """
        SELECT *
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher = @pitcher
        ORDER BY PitchNo
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
                bigquery.ScalarQueryParameter("pitcher", "STRING", pitcher_name),
            ]
        )
        
        pitcher_result = client.query(pitcher_data_query, job_config=job_config)
        pitch_data = [dict(row) for row in pitcher_result]
        
        if not pitch_data:
            update_email_status(pitcher_name, selected_date, 'failed', pitcher_email)
            return jsonify({'error': f'No pitch data found for {pitcher_name}'}), 400
        
        # Generate PDF
        comparison_level = get_pitcher_competition_level(pitcher_name)
        pdf_data = generate_pitcher_pdf(pitcher_name, pitch_data, selected_date, comparison_level)
        
        if not pdf_data:
            update_email_status(pitcher_name, selected_date, 'failed', pitcher_email)
            return jsonify({'error': 'Failed to generate PDF'}), 500
        
        # Send email using production service
        email_sender = ProductionEmailSender(email_config)
        success = email_sender.send_email(
            pitcher_name, 
            pitcher_email, 
            pdf_data, 
            selected_date, 
            comparison_level
        )
        email_sender.close()
        
        # Update status
        status = 'sent' if success else 'failed'
        update_email_status(pitcher_name, selected_date, status, pitcher_email)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Email sent successfully to {pitcher_name}',
                'pitcher_name': pitcher_name,
                'email': pitcher_email,
                'comparison_level': comparison_level
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to send email to {pitcher_name}'
            })
    
    except Exception as e:
        if 'pitcher_name' in locals():
            update_email_status(pitcher_name, selected_date, 'failed', pitcher_email)
        
        print(f"Error in individual email: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/email-system-status', methods=['GET'])
def email_system_status():
    """Get email system health and configuration"""
    try:
        # Test connection
        connection_healthy = test_email_connection(email_config)
        
        # Get configuration info (without sensitive data)
        config_info = {
            'host': email_config.get('host'),
            'port': email_config.get('port'),
            'connection_pool_size': email_config.get('connection_pool_size'),
            'batch_size': email_config.get('batch_size'),
            'max_retries': email_config.get('max_retries'),
            'daily_limit': email_config.get('daily_limit')
        }
        
        return jsonify({
            'status': 'healthy' if connection_healthy else 'unhealthy',
            'connection_test': 'passed' if connection_healthy else 'failed',
            'configuration': config_info,
            'estimated_time_300_emails': estimate_send_time(300, email_config),
            'ready_for_production': connection_healthy
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'ready_for_production': False
        }), 500






# STEP 3: ALSO ADD DUPLICATE DETECTION ENDPOINT (simple version for now)

@app.route('/api/detect-duplicates')
def detect_duplicates():
    """Detect potential duplicate prospect names"""
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter required'}), 400
    
    try:
        print(f"Detecting duplicates for date: {selected_date}")
        
        # Get all prospects for the date
        prospects_query = """
        SELECT Prospect as name, Email, Event, Type, Comp
        FROM `V1PBRInfo.Info`
        WHERE Prospect IS NOT NULL
        ORDER BY Prospect
        """
        
        prospects_result = client.query(prospects_query)
        prospects = []
        for row in prospects_result:
            prospects.append({
                'name': row.name,
                'email': row.Email,
                'event': row.Event,
                'type': row.Type,
                'comp': row.Comp
            })
        
        # Simple duplicate detection (exact matches after normalization)
        name_groups = {}
        
        for prospect in prospects:
            # Normalize name for comparison
            normalized_name = normalize_name(prospect['name'])
            
            if normalized_name not in name_groups:
                name_groups[normalized_name] = []
            name_groups[normalized_name].append(prospect)
        
        # Find groups with more than one prospect
        duplicate_groups = []
        total_duplicates = 0
        
        for normalized_name, group in name_groups.items():
            if len(group) > 1:
                duplicate_groups.append({
                    'normalized_name': normalized_name,
                    'prospects': group,
                    'count': len(group)
                })
                total_duplicates += len(group)
        
        print(f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} total duplicates")
        
        return jsonify({
            'duplicate_groups': duplicate_groups,
            'total_duplicates': total_duplicates,
            'total_prospects': len(prospects)
        })
        
    except Exception as e:
        print(f"Error detecting duplicates: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def normalize_name(name):
    """Normalize name for comparison"""
    if not name:
        return ""
    
    import re
    # Convert to lowercase, remove punctuation, normalize spacing
    normalized = re.sub(r'[^\w\s]', '', name.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Handle common name variations
    # "Smith, John" -> "john smith"
    if ',' in name:
        parts = [p.strip() for p in name.split(',')]
        if len(parts) == 2:
            normalized = f"{parts[1]} {parts[0]}".lower()
    
    return normalized

# Add this new function to load and process the conference data
import pandas as pd

def load_conference_data():
    """Load conference data from CSV and create team mappings"""
    try:
        # Load the CSV file
        df = pd.read_csv('Teams.csv')
        
        # Create a mapping of conference names to team abbreviations
        conference_teams = {}
        
        # Group by conference and collect all team abbreviations
        for _, row in df.iterrows():
            conference = row['Conference']
            conf_ab = row['ConfAB']
            team_ab = row['TeamAB']
            
            # Skip rows without team data (summary rows)
            if pd.isna(team_ab):
                continue
                
            # Add teams to both full conference name and abbreviation
            if conference not in conference_teams:
                conference_teams[conference] = []
            if conf_ab not in conference_teams:
                conference_teams[conf_ab] = []
                
            if team_ab not in conference_teams[conference]:
                conference_teams[conference].append(team_ab)
            if team_ab not in conference_teams[conf_ab]:
                conference_teams[conf_ab].append(team_ab)
        
        return conference_teams
        
    except Exception as e:
        print(f"Error loading conference data: {str(e)}")
        return {}

# Global variable to store conference data
conference_teams_map = load_conference_data()

def get_team_filter_for_conference(comparison_level):
    """Get team filter string for a given conference"""
    global conference_teams_map
    
    # Check if it's a conference we have data for
    if comparison_level in conference_teams_map:
        teams = conference_teams_map[comparison_level]
        if teams:
            # Create SQL IN clause with proper escaping
            team_list = "', '".join(teams)
            return f"PitcherTeam IN ('{team_list}')"
    
    # Fall back to standard level filtering for D1, D2, D3
    if comparison_level in ['D1', 'D2', 'D3']:
        return f"Level = '{comparison_level}'"
    
    # Default to D1 if nothing matches
    return "Level = 'D1'"


@lru_cache(maxsize=200)
def get_college_averages(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get college baseball averages for comparison, filtered by pitcher handedness"""
    try:
        # Get the appropriate filter based on conference or level
        level_filter = get_team_filter_for_conference(comparison_level)
        
        query = f"""
        SELECT 
            AVG(RelSpeed) as avg_velocity,
            AVG(SpinRate) as avg_spin_rate,
            AVG(InducedVertBreak) as avg_ivb,
            AVG(HorzBreak) as avg_hb,
            AVG(RelSide) as avg_rel_side,
            AVG(RelHeight) as avg_rel_height,
            AVG(Extension) as avg_extension,
            COUNT(*) as pitch_count
        FROM `NCAABaseball.2025Final`
        WHERE TaggedPitchType = @pitch_type
        AND {level_filter}
        AND PitcherThrows = @pitcher_throws
        AND RelSpeed IS NOT NULL
        AND SpinRate IS NOT NULL
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitch_type", "STRING", pitch_type),
                bigquery.ScalarQueryParameter("pitcher_throws", "STRING", pitcher_throws),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        row = list(result)[0] if result else None
        
        if row and row.pitch_count > 0:
            return {
                'avg_velocity': float(row.avg_velocity) if row.avg_velocity else None,
                'avg_spin_rate': float(row.avg_spin_rate) if row.avg_spin_rate else None,
                'avg_ivb': float(row.avg_ivb) if row.avg_ivb else None,
                'avg_hb': float(row.avg_hb) if row.avg_hb else None,
                'avg_rel_side': float(row.avg_rel_side) if row.avg_rel_side else None,
                'avg_rel_height': float(row.avg_rel_height) if row.avg_rel_height else None,
                'avg_extension': float(row.avg_extension) if row.avg_extension else None,
                'pitch_count': int(row.pitch_count)
            }
        return None
        
    except Exception as e:
        print(f"Error getting college averages for {pitch_type} ({pitcher_throws}): {str(e)}")
        return None

@lru_cache(maxsize=200)
def get_college_percentile_data(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get college baseball data for percentile calculations"""
    try:
        # Get the appropriate filter based on conference or level
        level_filter = get_team_filter_for_conference(comparison_level)
        
        query = f"""
        SELECT 
            RelSpeed,
            SpinRate,
            InducedVertBreak,
            HorzBreak,
            RelSide,
            RelHeight,
            Extension
        FROM `NCAABaseball.2025Final`
        WHERE TaggedPitchType = @pitch_type
        AND {level_filter}
        AND PitcherThrows = @pitcher_throws
        AND RelSpeed IS NOT NULL
        AND SpinRate IS NOT NULL
        AND InducedVertBreak IS NOT NULL
        AND HorzBreak IS NOT NULL
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitch_type", "STRING", pitch_type),
                bigquery.ScalarQueryParameter("pitcher_throws", "STRING", pitcher_throws),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        
        data = {
            'velocity': [],
            'spin_rate': [],
            'ivb': [],
            'hb': [],
            'rel_side': [],
            'rel_height': [],
            'extension': []
        }
        
        for row in result:
            if row.RelSpeed is not None:
                data['velocity'].append(float(row.RelSpeed))
            if row.SpinRate is not None:
                data['spin_rate'].append(float(row.SpinRate))
            if row.InducedVertBreak is not None:
                data['ivb'].append(float(row.InducedVertBreak))
            if row.HorzBreak is not None:
                data['hb'].append(float(row.HorzBreak))
            if row.RelSide is not None:
                data['rel_side'].append(float(row.RelSide))
            if row.RelHeight is not None:
                data['rel_height'].append(float(row.RelHeight))
            if row.Extension is not None:
                data['extension'].append(float(row.Extension))
        
        return data if any(len(values) > 0 for values in data.values()) else None
        
    except Exception as e:
        print(f"Error getting college percentile data for {pitch_type} ({pitcher_throws}): {str(e)}")
        return None

@lru_cache(maxsize=200)
def get_college_max_velocity_percentile_data(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get college baseball MAX velocity data for percentile calculations"""
    try:
        # Get the appropriate filter based on conference or level
        level_filter = get_team_filter_for_conference(comparison_level)
        
        query = f"""
        SELECT 
            MAX(RelSpeed) as max_velo
        FROM `NCAABaseball.2025Final`
        WHERE TaggedPitchType = @pitch_type
        AND {level_filter}
        AND PitcherThrows = @pitcher_throws
        AND RelSpeed IS NOT NULL
        GROUP BY Pitcher
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitch_type", "STRING", pitch_type),
                bigquery.ScalarQueryParameter("pitcher_throws", "STRING", pitcher_throws),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        
        max_velocities = []
        for row in result:
            if row.max_velo is not None:
                max_velocities.append(float(row.max_velo))
        
        return max_velocities if len(max_velocities) > 0 else None
        
    except Exception as e:
        print(f"Error getting college max velocity percentile data for {pitch_type} ({pitcher_throws}): {str(e)}")
        return None

def calculate_percentile_rank(player_value, college_data_list, metric_name=None, pitch_type=None, pitcher_throws=None):
    """Calculate what percentile the player's value falls into compared to college population"""
    if player_value is None or not college_data_list or len(college_data_list) == 0:
        return None
    
    sorted_college_data = sorted(college_data_list)
    total_count = len(sorted_college_data)
    values_below = sum(1 for value in sorted_college_data if value < player_value)
    raw_percentile = (values_below / total_count) * 100
    
    # Determine if this percentile represents "better" performance using existing logic
    is_better = False
    if metric_name and pitch_type:
        if metric_name == 'hb':
            is_better = is_horizontal_break_better(1, pitch_type, pitcher_throws)
        elif metric_name == 'ivb':
            is_better = is_ivb_better(1, pitch_type)
        elif metric_name == 'velocity':
            is_better = is_velocity_better(1, pitch_type)
        elif metric_name == 'spin_rate':
            is_better = is_spin_rate_better(1, pitch_type)
        else:
            is_better = True
            
        if not is_better:
            raw_percentile = 100 - raw_percentile
    
    final_percentile = round(raw_percentile, 1)
    
    # Cap percentiles: 0% becomes 1%, 100% becomes 99%
    if final_percentile <= 0:
        final_percentile = 1.0
    elif final_percentile >= 100:
        final_percentile = 99.0
    
    return {
        'percentile': final_percentile,
        'better': final_percentile >= 50,
        'total_pitchers': total_count
    }

def calculate_difference_from_average_with_percentile(pitcher_value, college_data, metric_name=None, pitch_type=None, pitcher_throws=None):
    """Calculate percentile instead of difference but maintain existing structure for compatibility"""
    if pitcher_value is None or not college_data:
        return None
    
    percentile_result = calculate_percentile_rank(
        pitcher_value, 
        college_data, 
        metric_name=metric_name, 
        pitch_type=pitch_type, 
        pitcher_throws=pitcher_throws
    )
    
    if not percentile_result:
        return None
    
    return {
        'difference': percentile_result['percentile'],
        'better': percentile_result['better'],
        'absolute_diff': abs(percentile_result['percentile'] - 50)
    }

def calculate_percentile(value, comparison_value, metric_name=None, pitch_type=None, pitcher_throws=None):
    """Calculate how the pitcher's value compares to college average with proper pitch-specific logic"""
    if value is None or comparison_value is None:
        return None
    
    difference = value - comparison_value
    percentage_diff = (difference / comparison_value) * 100
    
    # Determine if the difference is "better" based on metric and pitch type
    if metric_name == 'hb' and pitch_type and pitcher_throws:
        better = is_horizontal_break_better(difference, pitch_type, pitcher_throws)
    elif metric_name == 'ivb' and pitch_type:
        better = is_ivb_better(difference, pitch_type)
    elif metric_name == 'velocity' and pitch_type:
        better = is_velocity_better(difference, pitch_type)
    else:
        # For all other metrics, more is generally better
        better = difference > 0
    
    return {
        'difference': difference,
        'percentage_diff': percentage_diff,
        'better': better
    }

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/dates')
def get_dates():
    """API endpoint to get all available dates"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    try:
        query = """
        SELECT DISTINCT Date
        FROM `V1PBR.Test`
        WHERE Date IS NOT NULL
        ORDER BY Date
        """
        
        result = client.query(query)
        dates = []
        for row in result:
            # Convert date to string format that matches what's stored
            date_val = row.Date
            if hasattr(date_val, 'strftime'):
                # If it's a datetime object, format it
                dates.append(date_val.strftime('%Y-%m-%d'))
            else:
                # If it's already a string, use as-is
                dates.append(str(date_val))
        
        return jsonify({'dates': dates})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pitchers')
def get_pitchers():
    """API endpoint to get unique pitchers for a specific date"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is required'}), 400
    
    try:
        # First, let's check what the actual date format is in the table
        debug_query = """
        SELECT DISTINCT Date, TYPEOF(Date) as date_type
        FROM `V1PBR.Test`
        WHERE Date IS NOT NULL
        LIMIT 5
        """
        
        debug_result = client.query(debug_query)
        print("Debug - Date formats in table:")
        for row in debug_result:
            print(f"Date: {row.Date}, Type: {row.date_type}")
        
        # Try different date matching approaches
        query = """
        SELECT DISTINCT Pitcher
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher IS NOT NULL
        ORDER BY Pitcher
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        pitchers = [row.Pitcher for row in result]
        
        return jsonify({'pitchers': pitchers})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pitcher-details')
def get_pitcher_details():
    """API endpoint to get detailed pitch data for a specific pitcher and date"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    selected_date = request.args.get('date')
    pitcher_name = request.args.get('pitcher')
    
    if not selected_date or not pitcher_name:
        return jsonify({'error': 'Date and pitcher parameters are required'}), 400
    
    try:
        query = """
        SELECT *
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher = @pitcher
        ORDER BY PitchNo
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
                bigquery.ScalarQueryParameter("pitcher", "STRING", pitcher_name),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        
        # Convert to list of dictionaries
        pitch_data = []
        for row in result:
            pitch_data.append(dict(row))
        
        return jsonify({'pitch_data': pitch_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/all-prospects')
def get_all_prospects():
    """API endpoint to get ALL prospects from Info table for any event on the selected date"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter required'}), 400
    
    try:
        # Get ALL prospects from Info table
        # We'll filter by finding events that have pitch data on the selected date
        # First, get all events that have pitch data for this date
        events_query = """
        SELECT DISTINCT Event
        FROM `V1PBR.Test` t
        JOIN `V1PBRInfo.Info` i ON t.Pitcher = i.Prospect
        WHERE CAST(t.Date AS STRING) = @date
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
            ]
        )
        
        events_result = client.query(events_query, job_config=job_config)
        events_for_date = [row.Event for row in events_result]
        
        if not events_for_date:
            # If no events found, return empty list
            return jsonify({'prospects': []})
        
        # Now get ALL prospects for those events (not just ones with pitch data)
        events_list = "', '".join(events_for_date)
        prospects_query = f"""
        SELECT Event, Prospect, Email, Type, Comp
        FROM `V1PBRInfo.Info`
        WHERE Prospect IS NOT NULL
        AND Type = 'Pitching'
        AND Event IN ('{events_list}')
        ORDER BY Prospect
        """
        
        prospects_result = client.query(prospects_query)
        all_prospects = []
        
        for row in prospects_result:
            all_prospects.append({
                'name': row.Prospect,
                'email': row.Email,
                'type': row.Type,
                'event': row.Event,
                'comp': row.Comp or 'D1'
            })
        
        print(f"Found {len(all_prospects)} total prospects for events on {selected_date}")
        return jsonify({'prospects': all_prospects})
    
    except Exception as e:
        print(f"Error in get_all_prospects: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint to get general dataset statistics"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    try:
        # Get total record count
        count_query = "SELECT COUNT(*) as total FROM `V1PBR.Test`"
        count_result = client.query(count_query)
        total_records = list(count_result)[0].total
        
        # Get date range
        date_range_query = """
        SELECT 
            MIN(CAST(Date AS STRING)) as earliest_date,
            MAX(CAST(Date AS STRING)) as latest_date,
            COUNT(DISTINCT CAST(Date AS STRING)) as unique_dates,
            COUNT(DISTINCT Pitcher) as unique_pitchers
        FROM `V1PBR.Test`
        WHERE Date IS NOT NULL
        """
        
        date_result = client.query(date_range_query)
        date_info = list(date_result)[0]
        
        # Get matching analysis between Test and Info tables
        # Get all pitchers from Test table
        test_pitchers_query = """
        SELECT DISTINCT Pitcher
        FROM `V1PBR.Test`
        WHERE Pitcher IS NOT NULL
        ORDER BY Pitcher
        """
        
        test_result = client.query(test_pitchers_query)
        test_pitchers = set([row.Pitcher for row in test_result])
        
        # Get all prospects from Info table
        info_prospects_query = """
        SELECT Event, Prospect, Email, Type
        FROM `V1PBRInfo.Info`
        ORDER BY Prospect
        """
        
        info_result = client.query(info_prospects_query)
        info_prospects = []
        info_prospect_names = set()
        
        for row in info_result:
            info_prospects.append({
                'name': row.Prospect,
                'email': row.Email,
                'type': row.Type,
                'event': row.Event
            })
            info_prospect_names.add(row.Prospect)
        
        # Find matches and mismatches
        matched_names = test_pitchers.intersection(info_prospect_names)
        test_only = test_pitchers - info_prospect_names  # In Test but not in Info
        info_only = info_prospect_names - test_pitchers  # In Info but not in Test
        
        # Get email info for matched prospects
        matched_with_email = 0
        matched_without_email = 0
        
        for prospect in info_prospects:
            if prospect['name'] in matched_names:
                if prospect['email']:
                    matched_with_email += 1
                else:
                    matched_without_email += 1
        
        return jsonify({
            'total_records': total_records,
            'earliest_date': date_info.earliest_date,
            'latest_date': date_info.latest_date,
            'unique_dates': date_info.unique_dates,
            'unique_pitchers': date_info.unique_pitchers,
            'matching_stats': {
                'total_in_info': len(info_prospect_names),
                'total_in_test': len(test_pitchers),
                'matched_names': len(matched_names),
                'matched_with_email': matched_with_email,
                'matched_without_email': matched_without_email,
                'in_test_only': len(test_only),
                'in_info_only': len(info_only),
                'test_only_names': list(test_only),
                'info_only_names': list(info_only),
                'matched_names_list': list(matched_names)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pitcher-summary')
def get_pitcher_summary():
    """API endpoint to get pitcher summary with automatic comparison level"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    selected_date = request.args.get('date')
    pitcher_name = request.args.get('pitcher')
    
    if not selected_date or not pitcher_name:
        return jsonify({'error': 'Date and pitcher parameters are required'}), 400
    
    try:
        # Get pitcher's detailed data
        query = """
        SELECT *
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher = @pitcher
        ORDER BY PitchNo
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
                bigquery.ScalarQueryParameter("pitcher", "STRING", pitcher_name),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        pitch_data = [dict(row) for row in result]
        
        if not pitch_data:
            return jsonify({'error': 'No pitch data found'}), 404
        
        # Get comparison level from Info table
        comparison_level = get_pitcher_competition_level(pitcher_name)
        
        # Determine pitcher handedness
        pitcher_throws = 'Right'
        for pitch in pitch_data:
            if pitch.get('PitcherThrows'):
                pitcher_throws = pitch.get('PitcherThrows')
                break
        
        # Generate multi-level comparisons using the pitcher's competition level
        multi_level_stats = get_multi_level_comparisons(pitch_data, pitcher_throws)
        
        # Generate movement plot
        movement_plot_svg = generate_movement_plot_svg(pitch_data, comparison_level=comparison_level)
        pitch_location_plot_svg = generate_pitch_location_plot_svg(pitch_data)
        
        # Calculate zone rates
        zone_rate_data = calculate_zone_rates(pitch_data)
        
        return jsonify({
            'pitch_data': pitch_data,
            'multi_level_stats': multi_level_stats,
            'movement_plot_svg': movement_plot_svg,
            'pitch_location_plot_svg': pitch_location_plot_svg,
            'zone_rate_data': zone_rate_data,  # Add this new data
            'comparison_level': comparison_level,
            'pitcher_throws': pitcher_throws
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/matched-prospects')
def get_matched_prospects():
    """API endpoint to get prospects that have both pitch data and email info"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is required'}), 400
    
    try:
        # Get pitchers for the selected date
        pitchers_query = """
        SELECT DISTINCT Pitcher
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher IS NOT NULL
        ORDER BY Pitcher
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
            ]
        )
        
        pitchers_result = client.query(pitchers_query, job_config=job_config)
        pitchers_from_test = [row.Pitcher for row in pitchers_result]
        
        # Get prospect info from Info table
        prospects_query = """
        SELECT Event, Prospect, Email, Type, Comp
        FROM `V1PBRInfo.Info`
        WHERE Prospect IS NOT NULL
        ORDER BY Prospect
        """
        
        prospects_result = client.query(prospects_query)
        matched_prospects = []
        
        for row in prospects_result:
            if row.Prospect in pitchers_from_test:
                matched_prospects.append({
                    'name': row.Prospect,
                    'email': row.Email,
                    'type': row.Type,
                    'event': row.Event,
                    'comp': row.Comp or 'D1'
                })
        
        return jsonify({'prospects': matched_prospects})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lru_cache(maxsize=200)
def get_college_zone_rates(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get college baseball zone rates for comparison"""
    try:
        # Get the appropriate filter based on conference or level
        level_filter = get_team_filter_for_conference(comparison_level)
        
        # Strike zone boundaries (same as your calculate_zone_rates function)
        strike_zone_query = f"""
        WITH zone_calculations AS (
            SELECT 
                CASE 
                    WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                    AND (-9.97/12) <= (-1 * PlateLocSide) AND (-1 * PlateLocSide) <= (9.97/12)
                    AND (18.00/12) <= PlateLocHeight AND PlateLocHeight <= (40.53/12)
                    THEN 1 
                    ELSE 0 
                END as in_zone
            FROM `NCAABaseball.2025Final`
            WHERE TaggedPitchType = @pitch_type
            AND {level_filter}
            AND PitcherThrows = @pitcher_throws
            AND PlateLocSide IS NOT NULL
            AND PlateLocHeight IS NOT NULL
        )
        SELECT 
            AVG(in_zone) * 100 as avg_zone_rate,
            COUNT(*) as pitch_count
        FROM zone_calculations
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitch_type", "STRING", pitch_type),
                bigquery.ScalarQueryParameter("pitcher_throws", "STRING", pitcher_throws),
            ]
        )
        
        result = client.query(strike_zone_query, job_config=job_config)
        row = list(result)[0] if result else None
        
        if row and row.pitch_count > 0:
            return {
                'avg_zone_rate': float(row.avg_zone_rate) if row.avg_zone_rate else None,
                'pitch_count': int(row.pitch_count)
            }
        return None
        
    except Exception as e:
        print(f"Error getting college zone rates for {pitch_type} ({pitcher_throws}): {str(e)}")
        return None


@lru_cache(maxsize=200)
def get_overall_college_zone_rate(comparison_level='D1', pitcher_throws='Right'):
    """Get overall college baseball zone rate across all pitch types"""
    try:
        # Get the appropriate filter based on conference or level
        level_filter = get_team_filter_for_conference(comparison_level)
        
        # Overall zone rate query
        overall_zone_query = f"""
        WITH zone_calculations AS (
            SELECT 
                CASE 
                    WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                    AND (-9.97/12) <= (-1 * PlateLocSide) AND (-1 * PlateLocSide) <= (9.97/12)
                    AND (18.00/12) <= PlateLocHeight AND PlateLocHeight <= (40.53/12)
                    THEN 1 
                    ELSE 0 
                END as in_zone
            FROM `NCAABaseball.2025Final`
            WHERE {level_filter}
            AND PitcherThrows = @pitcher_throws
            AND PlateLocSide IS NOT NULL
            AND PlateLocHeight IS NOT NULL
        )
        SELECT 
            AVG(in_zone) * 100 as avg_zone_rate,
            COUNT(*) as pitch_count
        FROM zone_calculations
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitcher_throws", "STRING", pitcher_throws),
            ]
        )
        
        result = client.query(overall_zone_query, job_config=job_config)
        row = list(result)[0] if result else None
        
        if row and row.pitch_count > 0:
            return float(row.avg_zone_rate) if row.avg_zone_rate else None
        return None
        
    except Exception as e:
        print(f"Error getting overall college zone rate ({pitcher_throws}): {str(e)}")
        return None

def calculate_zone_rates(pitch_data, comparison_level='D1', pitcher_throws='Right'):
    """Calculate zone rates for each pitch type and overall, with college comparisons"""
    try:
        # Strike zone boundaries (in feet, same as used in the plot)
        strike_zone = {
            'xmin': -9.97/12,  # Convert inches to feet
            'xmax': 9.97/12,
            'ymin': 18.00/12,
            'ymax': 40.53/12
        }
        
        def is_in_zone(plate_side, plate_height):
            """Check if a pitch is in the strike zone"""
            if plate_side is None or plate_height is None:
                return False
            
            # Flip plate_side for batter's perspective (same as in the plot)
            flipped_plate_side = -1 * float(plate_side)
            plate_height_float = float(plate_height)
            
            return (strike_zone['xmin'] <= flipped_plate_side <= strike_zone['xmax'] and
                    strike_zone['ymin'] <= plate_height_float <= strike_zone['ymax'])
        
        # Group pitches by type and calculate zone rates
        pitch_type_data = {}
        total_pitches = 0
        total_in_zone = 0
        
        for pitch in pitch_data:
            pitch_type = pitch.get('TaggedPitchType', 'Unknown')
            plate_side = pitch.get('PlateLocSide')
            plate_height = pitch.get('PlateLocHeight')
            
            # Skip pitches without location data
            if plate_side is None or plate_height is None:
                continue
                
            total_pitches += 1
            in_zone = is_in_zone(plate_side, plate_height)
            
            if in_zone:
                total_in_zone += 1
            
            if pitch_type not in pitch_type_data:
                pitch_type_data[pitch_type] = {
                    'total': 0,
                    'in_zone': 0
                }
            
            pitch_type_data[pitch_type]['total'] += 1
            if in_zone:
                pitch_type_data[pitch_type]['in_zone'] += 1
        
        # Calculate zone rate percentages for each pitch type WITH college comparisons
        zone_rates = {}
        for pitch_type, data in pitch_type_data.items():
            if data['total'] > 0:
                player_zone_rate = (data['in_zone'] / data['total']) * 100
                
                # Get college zone rate for this pitch type
                college_zone_data = get_college_zone_rates(pitch_type, comparison_level, pitcher_throws)
                college_zone_rate = college_zone_data['avg_zone_rate'] if college_zone_data else None
                
                # Calculate comparison
                zone_comparison = None
                if college_zone_rate is not None:
                    difference = player_zone_rate - college_zone_rate
                    zone_comparison = {
                        'difference': difference,
                        'better': difference > 0  # Higher zone rate is generally better
                    }
                
                zone_rates[pitch_type] = {
                    'zone_rate': player_zone_rate,
                    'in_zone': data['in_zone'],
                    'total': data['total'],
                    'college_zone_rate': college_zone_rate,
                    'zone_comparison': zone_comparison
                }
        
        # Calculate overall zone rate
        overall_zone_rate = (total_in_zone / total_pitches * 100) if total_pitches > 0 else 0
        
        # Get overall college zone rate
        overall_college_zone_rate = get_overall_college_zone_rate(comparison_level, pitcher_throws)
        
        # Calculate overall comparison
        overall_zone_comparison = None
        if overall_college_zone_rate is not None:
            overall_difference = overall_zone_rate - overall_college_zone_rate
            overall_zone_comparison = {
                'difference': overall_difference,
                'better': overall_difference > 0
            }
        
        return {
            'pitch_type_zone_rates': zone_rates,
            'overall_zone_rate': overall_zone_rate,
            'overall_college_zone_rate': overall_college_zone_rate,
            'overall_zone_comparison': overall_zone_comparison,
            'total_pitches_with_location': total_pitches,
            'total_in_zone': total_in_zone
        }
        
    except Exception as e:
        print(f"Error calculating zone rates with college comparison: {str(e)}")
        return None

def generate_pitch_location_plot_svg(pitch_data, width=700, height=600):

    try:
        # Group pitches by type
        pitch_types = {}
        for pitch in pitch_data:
            pitch_type = pitch.get('TaggedPitchType')
            if pitch_type and pitch.get('PlateLocSide') is not None and pitch.get('PlateLocHeight') is not None:
                if pitch_type not in pitch_types:
                    pitch_types[pitch_type] = []
                pitch_types[pitch_type].append({
                    'plate_side': -1 * float(pitch.get('PlateLocSide', 0)),  # Flip for batter's perspective
                    'plate_height': float(pitch.get('PlateLocHeight', 0))
                })
        
        if not pitch_types:
            return None
        
        # Define colors for pitch types (same as movement plot)
        colors = {
            'ChangeUp': '#059669', 'Curveball': '#1D4ED8', 'Cutter': '#BE185D',
            'Fastball': '#DC2626', 'Knuckleball': '#9333EA', 'Sinker': '#EA580C',
            'Slider': '#7C3AED', 'Splitter': '#0891B2', 'Sweeper': '#F59E0B',
            'Four-Seam': '#DC2626', '4-Seam': '#DC2626', 'Two-Seam': '#EA580C',
            'TwoSeam': '#EA580C', 'Changeup': '#059669', 'Change-up': '#059669',
            'Curve': '#1D4ED8', 'Cut Fastball': '#BE185D', 'Split-Finger': '#0891B2'
        }
        
        # Set up plot dimensions - wider with space for 3D effect
        margin_left = 60
        margin_right = 150  # Space for legend and 3D effect
        margin_top = 60
        margin_bottom = 60
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        
        # Define coordinate ranges (feet)
        x_min, x_max = -4, 4
        y_min, y_max = -2, 6
        
        # Scale functions
        def scale_x(x):
            return margin_left + (x - x_min) / (x_max - x_min) * plot_width
        
        def scale_y(y):
            return margin_top + plot_height - (y - y_min) / (y_max - y_min) * plot_height
        
        # Strike zone dimensions (feet)
        strike_zone = {
            'xmin': -9.97/12,  # Convert inches to feet
            'xmax': 9.97/12,
            'ymin': 18.00/12,
            'ymax': 40.53/12
        }
        
        # Larger strike zone (shadow zone)
        larger_strike_zone = {
            'xmin': strike_zone['xmin'] - 2.00/12,
            'xmax': strike_zone['xmax'] + 2.00/12,
            'ymin': strike_zone['ymin'] - 1.47/12,
            'ymax': strike_zone['ymax'] + 1.47/12
        }
        
        # Home plate coordinates - create 3D effect with multiple layers (from old Python code)
        # Base layer (bottom)
        home_plate_base = [
            (-0.7, -0.1),
            (0.7, -0.1),
            (0.7, 0.2),
            (0, 0.5),
            (-0.7, 0.2),
            (-0.7, -0.1)
        ]
        
        # Top layer (slightly offset for 3D effect)
        home_plate_lifted = [
            (-0.7, 0.0),
            (0.7, 0.0),
            (0.7, 0.3),
            (0, 0.6),
            (-0.7, 0.3),
            (-0.7, 0.0)
        ]
        
        # Start SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '<style>',
            '.strike-zone { stroke: black; stroke-width: 2; fill: none; }',
            '.shadow-zone { stroke: black; stroke-width: 1; fill: none; stroke-dasharray: 3,3; }',
            '.home-plate-base { stroke: black; stroke-width: 1; fill: #f0f0f0; }',
            '.home-plate-top { stroke: black; stroke-width: 2; fill: white; }',
            '.plate-connector { stroke: black; stroke-width: 1; }',
            '.batter-box { stroke: black; stroke-width: 3; fill: none; }',
            '.axis-text { font-family: Arial, sans-serif; font-size: 10px; fill: black; }',
            '.plot-title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #1a1a1a; text-anchor: start; }',
            '.plot-subtitle { font-family: Arial, sans-serif; font-size: 12px; fill: #666666; text-anchor: start; font-style: italic; }',
            '.legend-text { font-family: Arial, sans-serif; font-size: 9px; fill: black; }',
            '</style>',
            '</defs>',
            
            # White background
            f'<rect width="{width}" height="{height}" fill="white"/>',
        ]
        
        # Title and subtitle
        svg_parts.extend([
            f'<text x="{margin_left}" y="25" class="plot-title">Pitch Location</text>',
            f'<text x="{margin_left}" y="40" class="plot-subtitle">Pitcher\'s Perspective</text>'
        ])
        
        # Draw larger strike zone (shadow zone)
        larger_left = scale_x(larger_strike_zone['xmin'])
        larger_right = scale_x(larger_strike_zone['xmax'])
        larger_bottom = scale_y(larger_strike_zone['ymin'])
        larger_top = scale_y(larger_strike_zone['ymax'])
        
        svg_parts.append(f'<rect x="{larger_left}" y="{larger_top}" width="{larger_right - larger_left}" height="{larger_bottom - larger_top}" class="shadow-zone"/>')
        
        # Draw main strike zone
        zone_left = scale_x(strike_zone['xmin'])
        zone_right = scale_x(strike_zone['xmax'])
        zone_bottom = scale_y(strike_zone['ymin'])
        zone_top = scale_y(strike_zone['ymax'])
        
        svg_parts.append(f'<rect x="{zone_left}" y="{zone_top}" width="{zone_right - zone_left}" height="{zone_bottom - zone_top}" class="strike-zone"/>')
        
        # Draw strike zone grid lines (thirds)
        # Horizontal lines
        third_height = (strike_zone['ymax'] - strike_zone['ymin']) / 3
        for i in [1, 2]:
            y_pos = scale_y(strike_zone['ymin'] + i * third_height)
            svg_parts.append(f'<line x1="{zone_left}" y1="{y_pos}" x2="{zone_right}" y2="{y_pos}" stroke="black" stroke-width="1"/>')
        
        # Vertical lines
        third_width = (strike_zone['xmax'] - strike_zone['xmin']) / 3
        for i in [1, 2]:
            x_pos = scale_x(strike_zone['xmin'] + i * third_width)
            svg_parts.append(f'<line x1="{x_pos}" y1="{zone_top}" x2="{x_pos}" y2="{zone_bottom}" stroke="black" stroke-width="1"/>')
        
        # Draw home plate with 3D effect (like R code)
        # Draw base plate (bottom layer)
        base_points = []
        for x, y in home_plate_base:
            base_points.append(f"{scale_x(x)},{scale_y(y)}")
        svg_parts.append(f'<polygon points="{" ".join(base_points)}" class="home-plate-base"/>')
        
        # Draw lifted plate (top layer)
        lifted_points = []
        for x, y in home_plate_lifted:
            lifted_points.append(f"{scale_x(x)},{scale_y(y)}")
        svg_parts.append(f'<polygon points="{" ".join(lifted_points)}" class="home-plate-top"/>')
        
        # Draw connecting lines for 3D effect (like the R code segments)
        for i in range(len(home_plate_base)):
            x1, y1 = home_plate_base[i]
            x2, y2 = home_plate_lifted[i]
            svg_parts.append(f'<line x1="{scale_x(x1)}" y1="{scale_y(y1)}" x2="{scale_x(x2)}" y2="{scale_y(y2)}" class="plate-connector"/>')
        
        # Draw batter's boxes with 3D effect (like R code)
        # Right batter's box
        right_box_back = 1.1
        right_box_front = 0.85
        right_box_outside = 2.5
        box_top = 0.3
        box_bottom = -1.0
        
        # Right box 3D lines (matching R code segments exactly)
        svg_parts.extend([
            # Right box angled front line (matches R: x = 1.1, y = -1, xend = .92, yend = 0.3)
            f'<line x1="{scale_x(1.1)}" y1="{scale_y(-1)}" x2="{scale_x(0.92)}" y2="{scale_y(0.3)}" stroke="black" stroke-width="5"/>',
            # Right box horizontal top line (matches R: x = .85, y = 0.3, xend = 2.5, yend = 0.3)
            f'<line x1="{scale_x(0.85)}" y1="{scale_y(0.3)}" x2="{scale_x(2.5)}" y2="{scale_y(0.3)}" stroke="black" stroke-width="5"/>',
        ])
        
        # Left batter's box (mirror of right)
        svg_parts.extend([
            # Left box angled front line (matches R: x = -1.1, y = -1, xend = -0.92, yend = 0.3)
            f'<line x1="{scale_x(-1.1)}" y1="{scale_y(-1)}" x2="{scale_x(-0.92)}" y2="{scale_y(0.3)}" stroke="black" stroke-width="5"/>',
            # Left box horizontal top line (matches R: x = -0.85, y = 0.3, xend = -2.5, yend = 0.3)
            f'<line x1="{scale_x(-0.85)}" y1="{scale_y(0.3)}" x2="{scale_x(-2.5)}" y2="{scale_y(0.3)}" stroke="black" stroke-width="5"/>',
        ])
        
        # Plot pitch locations
        for pitch_type, pitches in pitch_types.items():
            color = colors.get(pitch_type, '#666666')
            
            for pitch in pitches:
                x_pos = scale_x(pitch['plate_side'])
                y_pos = scale_y(pitch['plate_height'])
                
                # Simple circles for all pitches
                svg_parts.append(f'<circle cx="{x_pos}" cy="{y_pos}" r="4" fill="{color}" fill-opacity="0.7" stroke="white" stroke-width="1"/>')
        
        # Legend (positioned more to the left)
        legend_x = margin_left + plot_width - 100  # Moved 80px to the left
        legend_y = margin_top + 50
        
        # Pitch type legend
        svg_parts.append(f'<text x="{legend_x}" y="{legend_y}" class="legend-text" style="font-weight: bold;">Pitch Types:</text>')
        current_y = legend_y + 15
        
        for pitch_type in pitch_types.keys():
            color = colors.get(pitch_type, '#666666')
            svg_parts.extend([
                f'<circle cx="{legend_x + 5}" cy="{current_y}" r="3" fill="{color}"/>',
                f'<text x="{legend_x + 15}" y="{current_y + 3}" class="legend-text">{pitch_type}</text>'
            ])
            current_y += 15
        
        # Add axis labels
        # X-axis label
        x_center = margin_left + plot_width/2
        svg_parts.append(f'<text x="{x_center}" y="{height - 20}" class="axis-text" text-anchor="middle" style="font-weight: bold;">Plate Location - Side (ft)</text>')
        
        # Y-axis label
        y_center = margin_top + plot_height/2
        svg_parts.append(f'<text x="20" y="{y_center}" class="axis-text" text-anchor="middle" style="font-weight: bold;" transform="rotate(-90, 20, {y_center})">Plate Location - Height (ft)</text>')
        
        # Add tick marks and labels
        # X-axis ticks
        for x in [-3, -2, -1, 0, 1, 2, 3]:
            x_pos = scale_x(x)
            svg_parts.append(f'<line x1="{x_pos}" y1="{margin_top + plot_height}" x2="{x_pos}" y2="{margin_top + plot_height + 5}" stroke="black" stroke-width="1"/>')
            svg_parts.append(f'<text x="{x_pos}" y="{margin_top + plot_height + 18}" class="axis-text" text-anchor="middle">{x}</text>')
        
        # Y-axis ticks
        for y in [0, 1, 2, 3, 4, 5]:
            y_pos = scale_y(y)
            svg_parts.append(f'<line x1="{margin_left - 5}" y1="{y_pos}" x2="{margin_left}" y2="{y_pos}" stroke="black" stroke-width="1"/>')
            svg_parts.append(f'<text x="{margin_left - 10}" y="{y_pos + 3}" class="axis-text" text-anchor="end">{y}</text>')
        
        # Close SVG
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
        
    except Exception as e:
        print(f"Error generating pitch location plot SVG: {str(e)}")
        return None

@lru_cache(maxsize=50)
def get_college_release_point_averages(comparison_level='D1', pitcher_throws='Right'):
    """Get college baseball release point averages for comparison, filtered by pitcher handedness"""
    try:
        # Get the appropriate filter based on conference or level
        level_filter = get_team_filter_for_conference(comparison_level)
        
        query = f"""
        SELECT 
            AVG(RelSide) as avg_rel_side,
            AVG(RelHeight) as avg_rel_height,
            COUNT(*) as pitch_count
        FROM `NCAABaseball.2025Final`
        WHERE {level_filter}
        AND PitcherThrows = @pitcher_throws
        AND RelSide IS NOT NULL
        AND RelHeight IS NOT NULL
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitcher_throws", "STRING", pitcher_throws),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        row = list(result)[0] if result else None
        
        if row and row.pitch_count > 0:
            return {
                'avg_rel_side': float(row.avg_rel_side) if row.avg_rel_side else None,
                'avg_rel_height': float(row.avg_rel_height) if row.avg_rel_height else None,
                'pitch_count': int(row.pitch_count)
            }
        return None
        
    except Exception as e:
        print(f"Error getting college release point averages ({pitcher_throws}): {str(e)}")
        return None

def get_movement_description(pitch_type, pitcher_throws, hb, ivb, rel_height):
    """Generate movement descriptions based on pitch characteristics"""
    
    # Map your pitch types to the R code abbreviations
    pitch_mapping = {
        'Fastball': 'FF', 'Sinker': 'SINK', 'Slider': 'SL',
        'Curveball': 'CB', 'Cutter': 'CT', 'ChangeUp': 'CH', 'Splitter': 'SP'
    }
    
    pitch_code = pitch_mapping.get(pitch_type, '')
    if not pitch_code:
        return ""
    
    ivb_desc = ""
    hb_desc = ""
    
    # IVB (Vertical) Descriptions for Fastballs and Sinkers
    if pitch_code in ['FF', 'SINK']:
        if rel_height > 6.1:
            if ivb > 20: ivb_desc = "Plus Ride"
            elif ivb >= 17: ivb_desc = "Ride"
            elif ivb >= 13: ivb_desc = "Straight"
            elif ivb >= 8: ivb_desc = "Sink"
            else: ivb_desc = "Plus Sink"
        elif 5.5 <= rel_height <= 6.1:
            if ivb >= 20: ivb_desc = "Plus Ride"
            elif ivb >= 17: ivb_desc = "Ride"
            elif ivb >= 12: ivb_desc = "Straight"
            elif ivb >= 7: ivb_desc = "Sink"
            else: ivb_desc = "Plus Sink"
        elif 4.4 <= rel_height < 5.5:
            if ivb >= 17: ivb_desc = "Plus Ride"
            elif ivb >= 13: ivb_desc = "Ride"
            elif ivb >= 10: ivb_desc = "Straight"
            elif ivb >= 7: ivb_desc = "Sink"
            else: ivb_desc = "Plus Sink"
        else:  # rel_height < 4.4
            if ivb >= 15: ivb_desc = "Plus Ride"
            elif ivb >= 10: ivb_desc = "Ride"
            elif ivb >= 8: ivb_desc = "Straight"
            elif ivb >= 5: ivb_desc = "Sink"
            else: ivb_desc = "Plus Sink"
        
        # HB (Horizontal) Descriptions for Fastballs and Sinkers
        if pitcher_throws == 'Right':
            if hb < 0: hb_desc = "With Elite Cut"
            elif 12 < hb <= 15: hb_desc = "Armside Run"
            elif hb > 15: hb_desc = "Armside +Run"
        else:  # Left-handed
            if hb > 0: hb_desc = "With Elite Cut"
            elif -15 <= hb < -12: hb_desc = "Armside Run"
            elif hb < -15: hb_desc = "Armside +Run"
    
    # Breaking Ball Descriptions
    elif pitch_code in ['SL', 'CB', 'CT']:
        if pitcher_throws == 'Right':
            if -30 <= hb <= -12 and -5 <= ivb <= 20:
                ivb_desc = "SWEEPER"
            elif -30 <= hb <= -10 and -30 <= ivb < -5:
                ivb_desc = "Slurve"
            elif -12 < hb <= -4 and -5 <= ivb <= 30:
                ivb_desc = "Traditional Slider"
            elif -10 < hb <= -4 and -30 <= ivb <= -5:
                ivb_desc = "Slurve"
            elif -4 < hb <= 0 and -5 <= ivb <= 4:
                ivb_desc = "Gyro"
            elif hb > -4 and ivb >= 4:
                ivb_desc = "Cutter"
            elif -4 < hb <= 0 and ivb <= -5:
                ivb_desc = "12/6"
        else:  # Left-handed
            if 12 <= hb <= 30 and -5 <= ivb <= 20:
                ivb_desc = "SWEEPER"
            elif 10 <= hb <= 30 and -30 <= ivb < -5:
                ivb_desc = "Slurve"
            elif 4 <= hb < 12 and -5 <= ivb <= 30:
                ivb_desc = "Traditional Slider"
            elif 4 <= hb <= 10 and -30 <= ivb <= -5:
                ivb_desc = "Slurve"
            elif 0 <= hb <= 4 and -5 <= ivb <= 4:
                ivb_desc = "Gyro"
            elif hb <= 4 and ivb >= 4:
                ivb_desc = "Cutter"
            elif 0 <= hb <= 4 and ivb <= -5:
                ivb_desc = "12/6"
    
    # Changeup/Splitter Descriptions
    elif pitch_code in ['CH', 'SP']:
        if pitcher_throws == 'Right':
            if 0 <= hb <= 13 and ivb <= 5:
                ivb_desc = "+Depth"
            elif 0 <= hb <= 13 and 5 <= ivb <= 10:
                ivb_desc = "Depth"
            elif 0 <= hb <= 17 and ivb >= 10:
                ivb_desc = "Fastball Shape"
            elif 13 <= hb <= 17 and 5 <= ivb <= 10:
                ivb_desc = "Depth with fade"
            elif 13 <= hb <= 17 and ivb <= 5:
                ivb_desc = "+Depth with Fade"
            elif hb >= 17 and ivb <= 5:
                ivb_desc = "+Depth with +Fade"
            elif hb >= 17 and 5 <= ivb <= 10:
                ivb_desc = "Depth with +fade"
            elif hb >= 17 and ivb >= 10:
                ivb_desc = "+Fade"
        else:  # Left-handed
            if -13 <= hb <= 0 and ivb <= 5:
                ivb_desc = "+Depth"
            elif -13 <= hb <= 0 and 5 <= ivb <= 10:
                ivb_desc = "Depth"
            elif -17 <= hb <= 0 and ivb >= 10:
                ivb_desc = "Fastball Shape"
            elif -17 <= hb <= -13 and 5 <= ivb <= 10:
                ivb_desc = "Depth with fade"
            elif -17 <= hb <= -13 and ivb <= 5:
                ivb_desc = "+Depth with Fade"
            elif hb <= -17 and ivb <= 5:
                ivb_desc = "+Depth with +Fade"
            elif hb <= -17 and 5 <= ivb <= 10:
                ivb_desc = "Depth with +fade"
            elif hb <= -17 and ivb >= 10:
                ivb_desc = "+Fade"
    
    # Clean up descriptions - remove "With" if it's standalone
    if hb_desc.startswith("With ") and ivb_desc == "":
        hb_desc = hb_desc[5:]  # Remove "With "
    elif hb_desc.startswith("With ") and ivb_desc == "Straight":
        hb_desc = hb_desc[5:]  # Remove "With "
        ivb_desc = ""  # Remove "Straight"
    
    # Combine descriptions
    if ivb_desc and hb_desc:
        return f"{ivb_desc}, {hb_desc}"
    elif ivb_desc:
        return ivb_desc
    elif hb_desc:
        return hb_desc
    else:
        return ""

def generate_movement_plot_svg(pitch_data, width=1000, height=500, comparison_level='D1'):
    """Generate SVG for both movement plot (left) and release plot (right) with movement descriptions and release point legend"""
    try:
        # Group pitches by type
        pitch_types = {}
        pitcher_throws = 'Right'  # Default to right-handed
        
        for pitch in pitch_data:
            pitch_type = pitch.get('TaggedPitchType')
            
            # Get pitcher handedness from the data
            if pitch.get('PitcherThrows'):
                pitcher_throws = pitch.get('PitcherThrows')
            
            if pitch_type and pitch.get('HorzBreak') is not None and pitch.get('InducedVertBreak') is not None:
                if pitch_type not in pitch_types:
                    pitch_types[pitch_type] = []
                pitch_types[pitch_type].append({
                    'hb': float(pitch.get('HorzBreak', 0)),
                    'ivb': float(pitch.get('InducedVertBreak', 0)),
                    'rel_side': float(pitch.get('RelSide', 0)) if pitch.get('RelSide') is not None else None,
                    'rel_height': float(pitch.get('RelHeight', 5.5)) if pitch.get('RelHeight') is not None else None
                })
        
        if not pitch_types:
            return None
        
        # Define colors for pitch types
        colors = {
            'ChangeUp': '#059669', 'Curveball': '#1D4ED8', 'Cutter': '#BE185D',
            'Fastball': '#DC2626', 'Knuckleball': '#9333EA', 'Sinker': '#EA580C',
            'Slider': '#7C3AED', 'Splitter': '#0891B2', 'Sweeper': '#F59E0B',
            'Four-Seam': '#DC2626', '4-Seam': '#DC2626', 'Two-Seam': '#EA580C',
            'TwoSeam': '#EA580C', 'Changeup': '#059669', 'Change-up': '#059669',
            'Curve': '#1D4ED8', 'Cut Fastball': '#BE185D', 'Split-Finger': '#0891B2'
        }
        
        # Function to calculate 95% confidence ellipse (only for movement plot)
        def calculate_confidence_ellipse(x_values, y_values, confidence=0.95):
            if len(x_values) < 3:
                return []
            
            import math
            
            x_mean = sum(x_values) / len(x_values)
            y_mean = sum(y_values) / len(y_values)
            
            x_diff = [x - x_mean for x in x_values]
            y_diff = [y - y_mean for y in y_values]
            n = len(x_values)
            
            cov_xx = sum(x * x for x in x_diff) / (n - 1)
            cov_xy = sum(x * y for x, y in zip(x_diff, y_diff)) / (n - 1)
            cov_yy = sum(y * y for y in y_diff) / (n - 1)
            
            trace = cov_xx + cov_yy
            det = cov_xx * cov_yy - cov_xy * cov_xy
            
            if det <= 0:
                return []
            
            lambda1 = (trace + math.sqrt(trace * trace - 4 * det)) / 2
            lambda2 = (trace - math.sqrt(trace * trace - 4 * det)) / 2
            
            if lambda1 <= 0 or lambda2 <= 0:
                return []
            
            scale = math.sqrt(5.991)  # 95% confidence for 2D
            a = scale * math.sqrt(lambda1)
            b = scale * math.sqrt(lambda2)
            
            if abs(cov_xy) < 1e-10:
                theta = 0 if cov_xx >= cov_yy else math.pi / 2
            else:
                theta = math.atan2(2 * cov_xy, cov_xx - cov_yy) / 2
            
            points = []
            for t in [i * 0.1 for i in range(int(2 * math.pi / 0.1) + 1)]:
                x = a * math.cos(t) * math.cos(theta) - b * math.sin(t) * math.sin(theta) + x_mean
                y = a * math.cos(t) * math.sin(theta) + b * math.sin(t) * math.cos(theta) + y_mean
                points.append((x, y))
            
            return points
        
        # Set up plot dimensions - two side-by-side plots
        margin_left = 60
        margin_right = 40
        margin_top = 40
        margin_bottom = 40
        center_gap = 40
        
        plot_width = (width - margin_left - margin_right - center_gap) // 2
        plot_height = height - margin_top - margin_bottom
        
        # Movement plot (left)
        mov_x_start = margin_left
        mov_y_start = margin_top
        
        # Release plot (right)
        rel_x_start = margin_left + plot_width + center_gap
        rel_y_start = margin_top
        
        # Scale functions for movement plot
        mov_x_min, mov_x_max = -30, 30
        mov_y_min, mov_y_max = -30, 30
        
        def scale_mov_x(x):
            return mov_x_start + (x - mov_x_min) / (mov_x_max - mov_x_min) * plot_width
        
        def scale_mov_y(y):
            return mov_y_start + plot_height - (y - mov_y_min) / (mov_y_max - mov_y_min) * plot_height
        
        # Scale functions for release plot
        rel_x_min, rel_x_max = -5, 5
        rel_y_min, rel_y_max = 0, 8
        
        def scale_rel_x(x):
            return rel_x_start + (x - rel_x_min) / (rel_x_max - rel_x_min) * plot_width
        
        def scale_rel_y(y):
            return rel_y_start + plot_height - (y - rel_y_min) / (rel_y_max - rel_y_min) * plot_height
        
        # Calculate movement descriptions for each pitch type
        pitch_descriptions = {}
        for pitch_type, pitches in pitch_types.items():
            # Calculate average movement for description
            avg_hb = sum(p['hb'] for p in pitches) / len(pitches)
            avg_ivb = sum(p['ivb'] for p in pitches) / len(pitches)
            avg_rel_height = sum(p['rel_height'] for p in pitches if p['rel_height'] is not None) / len([p for p in pitches if p['rel_height'] is not None]) if any(p['rel_height'] is not None for p in pitches) else 5.5
            
            description = get_movement_description(pitch_type, pitcher_throws, avg_hb, avg_ivb, avg_rel_height)
            pitch_descriptions[pitch_type] = description
        
        # Calculate positions for axis labels
        mov_center_x = mov_x_start + plot_width/2
        mov_bottom_y = height - 10
        mov_left_y = mov_y_start + plot_height/2
        
        rel_center_x = rel_x_start + plot_width/2
        rel_right_x = width - 15
        rel_center_y = rel_y_start + plot_height/2
        
        # Start SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '<style>',
            '.axis-line { stroke: #990000; stroke-width: 2; }',
            '.grid-line { stroke: rgba(0,0,0,0.2); stroke-width: 1; }',
            '.axis-text { font-family: Arial, sans-serif; font-size: 10px; fill: black; }',
            '.axis-title { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; fill: black; }',
            '.legend-text { font-family: Arial, sans-serif; font-size: 9px; fill: black; }',
            '.plot-title { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; fill: #1a1a1a; text-anchor: start; }',
            '.plot-subtitle { font-family: Arial, sans-serif; font-size: 10px; fill: #666666; text-anchor: start; font-style: italic; }',
            '.plot-border { stroke: black; stroke-width: 2; fill: none; }',
            '.confidence-ellipse { fill: none; stroke-width: 1.5; stroke-opacity: 0.7; }',
            '.home-plate { stroke: #990000; stroke-width: 2; fill: white; }',
            '.release-text { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; fill: #990000; text-anchor: middle; }',
            '.movement-desc { font-family: Arial, sans-serif; font-size: 10px; font-weight: bold; text-anchor: middle; }',
            '.release-legend { font-family: Arial, sans-serif; font-size: 10px; fill: black; }',
            '.release-legend-box { fill: rgba(255,255,255,0.9); stroke: #ccc; stroke-width: 1; }',
            '</style>',
            '</defs>',
            
            # White background
            f'<rect width="{width}" height="{height}" fill="white"/>',
        ]
        
        # === MOVEMENT PLOT (LEFT) ===
        
        # Movement plot titles
        svg_parts.append(f'<text x="{mov_x_start}" y="20" class="plot-title">Movement Profile</text>')
        svg_parts.append(f'<text x="{mov_x_start}" y="32" class="plot-subtitle">Pitcher\'s Perspective</text>')
        
        # Movement plot grid and axes
        for x in range(-25, 31, 5):
            x_pos = scale_mov_x(x)
            line_class = 'axis-line' if x == 0 else 'grid-line'
            svg_parts.append(f'<line x1="{x_pos}" y1="{mov_y_start}" x2="{x_pos}" y2="{mov_y_start + plot_height}" class="{line_class}"/>')
            if x != 0:
                svg_parts.append(f'<text x="{x_pos}" y="{mov_y_start + plot_height + 15}" class="axis-text" text-anchor="middle">{x}</text>')
        
        for y in range(-25, 31, 5):
            y_pos = scale_mov_y(y)
            line_class = 'axis-line' if y == 0 else 'grid-line'
            svg_parts.append(f'<line x1="{mov_x_start}" y1="{y_pos}" x2="{mov_x_start + plot_width}" y2="{y_pos}" class="{line_class}"/>')
            if y != 0:
                svg_parts.append(f'<text x="{mov_x_start - 10}" y="{y_pos + 3}" class="axis-text" text-anchor="end">{y}</text>')
        
        svg_parts.append(f'<rect x="{mov_x_start}" y="{mov_y_start}" width="{plot_width}" height="{plot_height}" class="plot-border"/>')
        
        # Movement plot axis labels
        svg_parts.extend([
            f'<text x="{mov_center_x}" y="{mov_bottom_y}" class="axis-title" text-anchor="middle">Horizontal Break (in)</text>',
            f'<text x="20" y="{mov_left_y}" class="axis-title" text-anchor="middle" transform="rotate(-90, 20, {mov_left_y})">Induced Vertical Break (in)</text>'
        ])
        
        # === RELEASE PLOT (RIGHT) ===
        
        # Release plot titles
        svg_parts.append(f'<text x="{rel_x_start}" y="20" class="plot-title">Release Point</text>')
        svg_parts.append(f'<text x="{rel_x_start}" y="32" class="plot-subtitle">Pitcher\'s Perspective</text>')
        
        # Release plot grid
        for x in range(-4, 5, 1):
            x_pos = scale_rel_x(x)
            line_class = 'axis-line' if x == 0 else 'grid-line'
            svg_parts.append(f'<line x1="{x_pos}" y1="{rel_y_start}" x2="{x_pos}" y2="{rel_y_start + plot_height}" class="{line_class}"/>')
            if x != 0:
                svg_parts.append(f'<text x="{x_pos}" y="{rel_y_start + plot_height + 15}" class="axis-text" text-anchor="middle">{x}</text>')
        
        for y in range(1, 8, 1):
            y_pos = scale_rel_y(y)
            svg_parts.append(f'<line x1="{rel_x_start}" y1="{y_pos}" x2="{rel_x_start + plot_width}" y2="{y_pos}" class="grid-line"/>')
            svg_parts.append(f'<text x="{rel_x_start - 10}" y="{y_pos + 3}" class="axis-text" text-anchor="end">{y}</text>')
        
        # Release plot border
        svg_parts.append(f'<rect x="{rel_x_start}" y="{rel_y_start}" width="{plot_width}" height="{plot_height}" class="plot-border"/>')
        
        # Release plot axis labels
        svg_parts.extend([
            f'<text x="{rel_center_x}" y="{mov_bottom_y}" class="axis-title" text-anchor="middle">Release Side (ft)</text>',
            f'<text x="{rel_right_x}" y="{rel_center_y}" class="axis-title" text-anchor="middle" transform="rotate(-90, {rel_right_x}, {rel_center_y})">Release Height (ft)</text>'
        ])
        
        # Add LHP/RHP labels to release plot
        svg_parts.extend([
            f'<text x="{scale_rel_x(-4)}" y="{scale_rel_y(7.5)}" class="release-text">LHP</text>',
            f'<text x="{scale_rel_x(4)}" y="{scale_rel_y(7.5)}" class="release-text">RHP</text>'
        ])
        
        # Add home plate to release plot
        plate_left = scale_rel_x(-1.0)
        plate_right = scale_rel_x(1.0)
        plate_top = scale_rel_y(1.0)
        plate_bottom = scale_rel_y(0.7)
        svg_parts.append(f'<rect x="{plate_left}" y="{plate_top}" width="{plate_right - plate_left}" height="{plate_bottom - plate_top}" class="home-plate"/>')
        
        # === DYNAMIC COLLEGE AVERAGE RELEASE POINT ===
        
        # Get dynamic release point averages based on league and handedness
        college_release_averages = get_college_release_point_averages(comparison_level, pitcher_throws)
        
        if college_release_averages and college_release_averages['avg_rel_side'] and college_release_averages['avg_rel_height']:
            # Use actual college averages
            avg_rel_side = college_release_averages['avg_rel_side']
            avg_rel_height = college_release_averages['avg_rel_height']
            
            # Add the dynamic average release point circle
            avg_x = scale_rel_x(avg_rel_side)
            avg_y = scale_rel_y(avg_rel_height)
            svg_parts.append(f'<circle cx="{avg_x}" cy="{avg_y}" r="6" fill="white" stroke="black" stroke-width="2"/>')
            
            print(f"Using {comparison_level} {pitcher_throws} release averages: Side={avg_rel_side:.2f}, Height={avg_rel_height:.2f}")
        else:
            # Fallback to original hardcoded values if query fails
            avg_rel_side = -1.7 if pitcher_throws == 'Left' else 1.66
            avg_rel_height = 5.7
            avg_x = scale_rel_x(avg_rel_side)
            avg_y = scale_rel_y(avg_rel_height)
            svg_parts.append(f'<circle cx="{avg_x}" cy="{avg_y}" r="6" fill="white" stroke="black" stroke-width="2"/>')
            
            print(f"Using fallback release averages: Side={avg_rel_side:.2f}, Height={avg_rel_height:.2f}")
        
        # === RELEASE POINT LEGEND ===
        
        # Create handedness abbreviation
        handedness_abbr = 'RHP' if pitcher_throws == 'Right' else 'LHP'
        college_text = f"{comparison_level} {handedness_abbr} Average"
        
        # Simple check: if text is too long, wrap it
        if len(college_text) > 14:  # Rough character limit for box width
            # Wrap the college text
            legend_box_height = 50
            svg_parts.append(f'<rect x="{rel_x_start + plot_width - 115}" y="{rel_y_start + 45}" width="110" height="{legend_box_height}" class="release-legend-box" rx="3"/>')
            
            # College average (open circle) - wrapped
            svg_parts.append(f'<circle cx="{rel_x_start + plot_width - 107}" cy="{rel_y_start + 57}" r="4" fill="white" stroke="black" stroke-width="1.5"/>')
            svg_parts.append(f'<text x="{rel_x_start + plot_width - 97}" y="{rel_y_start + 55}" class="release-legend" style="font-size: 9px;">{comparison_level} {handedness_abbr}</text>')
            svg_parts.append(f'<text x="{rel_x_start + plot_width - 97}" y="{rel_y_start + 65}" class="release-legend" style="font-size: 9px;">Average</text>')
            
            # Individual pitches
            first_pitch_color = list(colors.values())[0] if colors else '#666666'
            svg_parts.append(f'<circle cx="{rel_x_start + plot_width - 107}" cy="{rel_y_start + 77}" r="3" fill="{first_pitch_color}"/>')
            svg_parts.append(f'<text x="{rel_x_start + plot_width - 97}" y="{rel_y_start + 81}" class="release-legend" style="font-size: 9px;">Individual Pitches</text>')
        else:
            # Single line text
            legend_box_height = 35
            svg_parts.append(f'<rect x="{rel_x_start + plot_width - 115}" y="{rel_y_start + 45}" width="110" height="{legend_box_height}" class="release-legend-box" rx="3"/>')
            
            # College average (open circle) - single line
            svg_parts.append(f'<circle cx="{rel_x_start + plot_width - 107}" cy="{rel_y_start + 55}" r="4" fill="white" stroke="black" stroke-width="1.5"/>')
            svg_parts.append(f'<text x="{rel_x_start + plot_width - 97}" y="{rel_y_start + 59}" class="release-legend">{college_text}</text>')
            
            # Individual pitches
            first_pitch_color = list(colors.values())[0] if colors else '#666666'
            svg_parts.append(f'<circle cx="{rel_x_start + plot_width - 107}" cy="{rel_y_start + 70}" r="3" fill="{first_pitch_color}"/>')
            svg_parts.append(f'<text x="{rel_x_start + plot_width - 97}" y="{rel_y_start + 74}" class="release-legend">Individual Pitches</text>')
        
        # === PLOT DATA FOR BOTH CHARTS ===
        
        for pitch_type, pitches in pitch_types.items():
            color = colors.get(pitch_type, '#666666')
            
            # Extract coordinates
            hb_values = [p['hb'] for p in pitches]
            ivb_values = [p['ivb'] for p in pitches]
            rel_side_values = [p['rel_side'] for p in pitches if p['rel_side'] is not None]
            rel_height_values = [p['rel_height'] for p in pitches if p['rel_height'] is not None]
            
            # === MOVEMENT PLOT DATA ===
            
            # Draw 95% confidence ellipse
            if len(pitches) >= 3:
                ellipse_points = calculate_confidence_ellipse(hb_values, ivb_values)
                if ellipse_points:
                    path_data = []
                    for i, (x, y) in enumerate(ellipse_points):
                        x_pos = scale_mov_x(x)
                        y_pos = scale_mov_y(y)
                        if i == 0:
                            path_data.append(f'M {x_pos} {y_pos}')
                        else:
                            path_data.append(f'L {x_pos} {y_pos}')
                    path_data.append('Z')
                    svg_parts.append(f'<path d="{" ".join(path_data)}" class="confidence-ellipse" stroke="{color}"/>')
            
            # Individual movement points
            for pitch in pitches:
                x_pos = scale_mov_x(pitch['hb'])
                y_pos = scale_mov_y(pitch['ivb'])
                svg_parts.append(f'<circle cx="{x_pos}" cy="{y_pos}" r="2.5" fill="{color}" fill-opacity="0.6" stroke="rgba(255,255,255,0.4)" stroke-width="0.5"/>')
            
            # Average movement point
            if pitches:
                avg_hb = sum(p['hb'] for p in pitches) / len(pitches)
                avg_ivb = sum(p['ivb'] for p in pitches) / len(pitches)
                avg_x = scale_mov_x(avg_hb)
                avg_y = scale_mov_y(avg_ivb)
                svg_parts.append(f'<circle cx="{avg_x}" cy="{avg_y}" r="5" fill="{color}" stroke="rgba(0,0,0,0.8)" stroke-width="2"/>')
                
                # ADD MOVEMENT DESCRIPTION LABEL
                description = pitch_descriptions.get(pitch_type, '')
                if description:
                    label_x = avg_x
                    label_y = avg_y - 15  # Position above the average point
                    svg_parts.append(f'<text x="{label_x}" y="{label_y}" class="movement-desc" fill="{color}">{description}</text>')
            
            # === RELEASE PLOT DATA ===
            
            # Release individual points (only if we have release data)
            if rel_side_values and rel_height_values:
                for pitch in pitches:
                    if pitch['rel_side'] is not None and pitch['rel_height'] is not None:
                        x_pos = scale_rel_x(pitch['rel_side'])
                        y_pos = scale_rel_y(pitch['rel_height'])
                        svg_parts.append(f'<circle cx="{x_pos}" cy="{y_pos}" r="2.5" fill="{color}" fill-opacity="0.6" stroke="rgba(255,255,255,0.4)" stroke-width="0.5"/>')
        
        # Movement plot legend with descriptions (positioned to avoid release legend)
        if pitcher_throws == 'Left':
            legend_x = mov_x_start + 20
        else:
            legend_x = mov_x_start + plot_width - 180  # More space for longer text
        
        legend_y_start = mov_y_start + plot_height - 20
        current_legend_y = legend_y_start
        
        for pitch_type in pitch_types.keys():
            color = colors.get(pitch_type, '#666666')
            description = pitch_descriptions.get(pitch_type, '')
            
            # Create legend text with description
            legend_text = pitch_type
            if description:
                legend_text += f" - {description}"
            
            svg_parts.extend([
                f'<circle cx="{legend_x}" cy="{current_legend_y}" r="3" fill="{color}"/>',
                f'<text x="{legend_x + 10}" y="{current_legend_y + 3}" class="legend-text">{legend_text}</text>'
            ])
            current_legend_y -= 15
        
        # Close SVG
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
        
    except Exception as e:
        print(f"Error generating movement plot SVG: {str(e)}")
        return None

import json
import os
from datetime import datetime

def log_name_change(old_name, new_name, email, event, success=True, error_message=None):
    """Log name change to audit file"""
    try:
        audit_file = 'name_changes_audit.json'
        
        # Load existing audit log
        if os.path.exists(audit_file):
            with open(audit_file, 'r') as f:
                audit_log = json.load(f)
        else:
            audit_log = []
        
        # Create audit entry
        audit_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'old_name': old_name,
            'new_name': new_name,
            'email': email,
            'event': event,
            'success': success,
            'error_message': error_message
        }
        
        # Add to log
        audit_log.append(audit_entry)
        
        # Save back to file
        with open(audit_file, 'w') as f:
            json.dump(audit_log, f, indent=2)
        
        print(f"✓ Logged: {old_name} -> {new_name} ({'SUCCESS' if success else 'FAILED'})")
        return True
        
    except Exception as e:
        print(f"Error logging audit: {str(e)}")
        return False

@app.route('/api/replace-name', methods=['POST'])
def replace_name():
    """Replace a prospect name in the Info table with audit logging"""
    if not client:
        return jsonify({'error': 'BigQuery client not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['old_name', 'new_name', 'email', 'event']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields: old_name, new_name, email, event'}), 400
        
        old_name = data['old_name']
        new_name = data['new_name']
        email = data['email']
        event = data['event']
        
        print(f"Attempting to replace '{old_name}' with '{new_name}' for email {email}")
        
        try:
            # First, verify the new name exists in the Test table
            verify_query = """
            SELECT COUNT(*) as count
            FROM `V1PBR.Test`
            WHERE Pitcher = @new_name
            """
            
            verify_job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("new_name", "STRING", new_name),
                ]
            )
            
            verify_result = client.query(verify_query, job_config=verify_job_config)
            verify_row = list(verify_result)[0]
            
            if verify_row.count == 0:
                # Log failed attempt
                log_name_change(old_name, new_name, email, event, 
                              success=False, error_message=f'Name "{new_name}" not found in Test data')
                return jsonify({'error': f'Name "{new_name}" not found in Test data'}), 400
            
            # Update the Info table
            update_query = """
            UPDATE `V1PBRInfo.Info`
            SET Prospect = @new_name
            WHERE Prospect = @old_name
            AND Email = @email
            AND Event = @event
            """
            
            update_job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("old_name", "STRING", old_name),
                    bigquery.ScalarQueryParameter("new_name", "STRING", new_name),
                    bigquery.ScalarQueryParameter("email", "STRING", email),
                    bigquery.ScalarQueryParameter("event", "STRING", event),
                ]
            )
            
            update_job = client.query(update_query, job_config=update_job_config)
            update_job.result()  # Wait for the job to complete
            
            print(f"Updated {update_job.num_dml_affected_rows} rows")
            
            if update_job.num_dml_affected_rows == 0:
                # Log failed attempt
                log_name_change(old_name, new_name, email, event, 
                              success=False, error_message='No matching records found to update')
                return jsonify({'error': 'No matching records found to update'}), 404
            
            # Log successful change
            log_name_change(old_name, new_name, email, event, success=True)
            
            print(f"✓ Successfully replaced '{old_name}' with '{new_name}' for {email} in event {event}")
            
            return jsonify({
                'success': True,
                'message': f'Successfully updated {update_job.num_dml_affected_rows} record(s)',
                'old_name': old_name,
                'new_name': new_name,
                'rows_affected': update_job.num_dml_affected_rows
            })
            
        except Exception as db_error:
            # Log database error
            log_name_change(old_name, new_name, email, event, 
                          success=False, error_message=str(db_error))
            raise db_error
        
    except Exception as e:
        print(f"Error replacing name: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/audit-log')
def get_audit_log():
    """Get the audit log of name changes"""
    try:
        audit_file = 'name_changes_audit.json'
        
        if not os.path.exists(audit_file):
            return jsonify({'audit_log': []})
        
        with open(audit_file, 'r') as f:
            audit_log = json.load(f)
        
        # Show newest first
        audit_log = sorted(audit_log, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({'audit_log': audit_log})
        
    except Exception as e:
        print(f"Error getting audit log: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Add this new function to get college max velocity averages
@lru_cache(maxsize=200)
def get_college_max_velocity_averages(pitch_type, comparison_level='D1', pitcher_throws='Right'):
    """Get college baseball MAX velocity averages for comparison, filtered by pitcher handedness"""
    try:
        # Get the appropriate filter based on conference or level
        level_filter = get_team_filter_for_conference(comparison_level)
        
        query = f"""
        SELECT 
            AVG(max_velo) as avg_max_velocity,
            COUNT(DISTINCT Pitcher) as pitcher_count
        FROM (
            SELECT 
                Pitcher,
                MAX(RelSpeed) as max_velo
            FROM `NCAABaseball.2025Final`
            WHERE TaggedPitchType = @pitch_type
            AND {level_filter}
            AND PitcherThrows = @pitcher_throws
            AND RelSpeed IS NOT NULL
            GROUP BY Pitcher
        )
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitch_type", "STRING", pitch_type),
                bigquery.ScalarQueryParameter("pitcher_throws", "STRING", pitcher_throws),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        row = list(result)[0] if result else None
        
        if row and row.pitcher_count > 0:
            return {
                'avg_max_velocity': float(row.avg_max_velocity) if row.avg_max_velocity else None,
                'pitcher_count': int(row.pitcher_count)
            }
        return None
        
    except Exception as e:
        print(f"Error getting college max velocity averages for {pitch_type} ({pitcher_throws}): {str(e)}")
        return None

def get_available_conferences():
    """Get list of available conferences from the CSV data"""
    global conference_teams_map
    
    # Filter out standard levels (D1, D2, D3) and return just conferences
    conferences = []
    for conf in conference_teams_map.keys():
        if conf not in ['D1', 'D2', 'D3']:
            conferences.append(conf)
    
    return sorted(conferences)

# Replace the calculate_percentile_rank function with this new function:

def is_horizontal_break_better(difference, pitch_type, pitcher_throws):
    """Determine if horizontal break difference is better based on pitch type and handedness"""
    
    # Normalize pitch type names for comparison
    pitch_type_lower = pitch_type.lower()
    
    # Define pitch categories based on expected break patterns
    # These should break away from arm side (glove side break)
    breaking_balls = ['curveball', 'curve', 'slider', 'cutter', 'cut fastball', 'sweeper']
    
    # These should have arm-side run
    fastballs_and_offspeed = [
        'fastball', 'four-seam', '4-seam', 'fourseam', 'four seam',
        'sinker', 'two-seam', '2-seam', 'twoseam', 'two seam',
        'changeup', 'change-up', 'change up', 'changup',
        'splitter', 'split-finger', 'splitfinger', 'split finger',
        'knuckleball', 'knuckle ball'
    ]
    
    # Check if pitch type matches any variation
    is_breaking_ball = any(pattern in pitch_type_lower for pattern in breaking_balls)
    is_fastball_or_offspeed = any(pattern in pitch_type_lower for pattern in fastballs_and_offspeed)
    
    if pitcher_throws == 'Right':
        if is_breaking_ball:
            # RHP breaking balls should go negative (toward 1B) - more negative is better
            return difference < 0
        elif is_fastball_or_offspeed:
            # RHP fastballs/offspeed should go positive (toward 3B) - more positive is better
            return difference > 0
    elif pitcher_throws == 'Left':
        if is_breaking_ball:
            # LHP breaking balls should go positive (toward 3B) - more positive is better
            return difference > 0
        elif is_fastball_or_offspeed:
            # LHP fastballs/offspeed should go negative (toward 1B) - more negative is better
            return difference < 0
    
    # Default case: if pitch type doesn't match known categories, assume more is better
    return difference > 0


def is_ivb_better(difference, pitch_type):
    """Determine if IVB difference is better based on pitch type"""
    
    # Normalize pitch type names for comparison
    pitch_type_lower = pitch_type.lower()
    
    # Define pitch categories where negative IVB is better (breaking balls and offspeed)
    negative_ivb_pitches = [
        'curveball', 'curve', 
        'changeup', 'change-up', 'change up', 'changup',
        'splitter', 'split-finger', 'splitfinger', 'split finger',
        'knuckleball', 'knuckle ball',
        'sinker', 'two-seam', '2-seam', 'twoseam', 'two seam'  # Added sinker variations
    ]
    
    # Check for matches (using 'in' to catch variations)
    is_negative_ivb_pitch = any(pattern in pitch_type_lower for pattern in negative_ivb_pitches)
    
    if is_negative_ivb_pitch:
        # For these pitches, more negative IVB is better (more drop/sink)
        return difference < 0
    else:
        # For fastballs, cutters, sliders, sweepers - more positive IVB is better (more carry/rise)
        return difference > 0

def is_velocity_better(difference, pitch_type):
    """Determine if velocity difference is better based on pitch type"""
    
    # Normalize pitch type names for comparison
    pitch_type_lower = pitch_type.lower()
    
    # Define pitch categories where lower velocity is better (offspeed pitches)
    lower_velo_pitches = [
        'changeup', 'change-up', 'change up', 'changup',
        'splitter', 'split-finger', 'splitfinger', 'split finger',
        'knuckleball', 'knuckle ball'
    ]
    
    # Check for matches
    is_lower_velo_pitch = any(pattern in pitch_type_lower for pattern in lower_velo_pitches)
    
    if is_lower_velo_pitch:
        # For changeups, splitters, and knuckleballs, lower velocity is better
        return difference < 0
    else:
        # For all other pitches, higher velocity is better
        return difference > 0

def is_spin_rate_better(difference, pitch_type):
    """Determine if spin rate difference is better based on pitch type"""
    
    # Normalize pitch type names for comparison
    pitch_type_lower = pitch_type.lower()
    
    # Define pitch categories where lower spin rate is better
    lower_spin_pitches = [
        'splitter', 'split-finger', 'splitfinger', 'split finger',
        'knuckleball', 'knuckle ball'  # knuckleballs also benefit from lower spin
    ]
    
    # Check for matches
    is_lower_spin_pitch = any(pattern in pitch_type_lower for pattern in lower_spin_pitches)
    
    if is_lower_spin_pitch:
        # For splitters and knuckleballs, lower spin rate is better
        return difference < 0
    else:
        # For all other pitches, higher spin rate is better
        return difference > 0

# Update the calculate_difference_from_average function to use spin rate logic
def calculate_difference_from_average(pitcher_value, college_average, metric_name=None, pitch_type=None, pitcher_throws=None):
    """Calculate the difference between pitcher's value and college average"""
    if pitcher_value is None or college_average is None:
        return None
    
    difference = pitcher_value - college_average
    
    # Determine if the difference is "better" based on metric and pitch type
    if metric_name == 'hb' and pitch_type and pitcher_throws:
        better = is_horizontal_break_better(difference, pitch_type, pitcher_throws)
    elif metric_name == 'ivb' and pitch_type:
        better = is_ivb_better(difference, pitch_type)
    elif metric_name == 'velocity' and pitch_type:
        better = is_velocity_better(difference, pitch_type)
    elif metric_name == 'spin_rate' and pitch_type:  # Add this line
        better = is_spin_rate_better(difference, pitch_type)
    else:
        # For all other metrics, more is generally better
        better = difference > 0
    
    return {
        'difference': difference,
        'better': better,
        'absolute_diff': abs(difference)
    }

# Update the get_multi_level_comparisons function:

def get_multi_level_comparisons(pitch_data, pitcher_throws='Right'):
    """Get percentile-based comparisons while maintaining existing UI structure and adding college averages"""
    try:
        # Group pitches by type (same as before)
        pitch_type_data = {}
        
        for pitch in pitch_data:
            pitch_type = pitch.get('TaggedPitchType', 'Unknown')
            if pitch_type not in pitch_type_data:
                pitch_type_data[pitch_type] = {
                    'pitches': [],
                    'count': 0
                }
            pitch_type_data[pitch_type]['pitches'].append(pitch)
            pitch_type_data[pitch_type]['count'] += 1
        
        # Same sorting logic as before
        multi_level_breakdown = []
        pitcher_name = pitch_data[0].get('Pitcher') if pitch_data else None
        pitcher_comparison_level = get_pitcher_competition_level(pitcher_name) if pitcher_name else 'D1'
        levels = ['D1', 'D2', 'D3']
        
        priority_types = ['Fastball', 'Sinker', 'Cutter', 'Slider', 'Curveball', 'ChangeUp', 'Sweeper', 'Splitter', 'Knuckleball']
        
        sorted_pitch_types = []
        for priority_type in priority_types:
            for actual_type in pitch_type_data.keys():
                if priority_type.lower() in actual_type.lower() and actual_type not in sorted_pitch_types:
                    sorted_pitch_types.append(actual_type)
                    break
        
        remaining_types = [pt for pt in pitch_type_data.keys() if pt not in sorted_pitch_types]
        sorted_pitch_types.extend(sorted(remaining_types))
        
        for i, pitch_type in enumerate(sorted_pitch_types):
            pitches = pitch_type_data[pitch_type]['pitches']
            count = pitch_type_data[pitch_type]['count']
            
            # Same metric extraction as before
            velocities = [p.get('RelSpeed', 0) for p in pitches if p.get('RelSpeed')]
            spin_rates = [p.get('SpinRate', 0) for p in pitches if p.get('SpinRate')]
            ivbs = [p.get('InducedVertBreak', 0) for p in pitches if p.get('InducedVertBreak')]
            hbs = [p.get('HorzBreak', 0) for p in pitches if p.get('HorzBreak')]
            rel_heights = [p.get('RelHeight', 0) for p in pitches if p.get('RelHeight')]
            rel_sides = [p.get('RelSide', 0) for p in pitches if p.get('RelSide')]
            extensions = [p.get('Extension', 0) for p in pitches if p.get('Extension')]
            
            # Same average calculations as before
            pitcher_avg_velocity = sum(velocities)/len(velocities) if velocities else None
            pitcher_max_velocity = max(velocities) if velocities else None
            pitcher_avg_spin = sum(spin_rates)/len(spin_rates) if spin_rates else None
            pitcher_avg_ivb = sum(ivbs)/len(ivbs) if ivbs else None
            pitcher_avg_hb = sum(hbs)/len(hbs) if hbs else None
            pitcher_avg_rel_height = sum(rel_heights)/len(rel_heights) if rel_heights else None
            pitcher_avg_rel_side = sum(rel_sides)/len(rel_sides) if rel_sides else None
            pitcher_avg_extension = sum(extensions)/len(extensions) if extensions else None
            
            level_comparisons = {}
            
            # Get both percentile data AND college averages for each level
            for level in levels:
                # Get percentile data
                college_data = get_college_percentile_data_cached(pitch_type, level, pitcher_throws)
                college_max_velo_data = get_college_max_velocity_percentile_data_cached(pitch_type, level, pitcher_throws)
                
                # Get college averages (your existing function)
                college_averages = get_college_averages_cached(pitch_type, level, pitcher_throws)
                college_max_velo_averages = get_college_max_velocity_averages(pitch_type, level, pitcher_throws)
                
                # Calculate percentiles
                velocity_diff = calculate_difference_from_average_with_percentile(
                    pitcher_avg_velocity, 
                    college_data['velocity'] if college_data else None,
                    metric_name='velocity',
                    pitch_type=pitch_type
                )
                
                max_velocity_diff = calculate_difference_from_average_with_percentile(
                    pitcher_max_velocity, 
                    college_max_velo_data,
                    metric_name='velocity',
                    pitch_type=pitch_type
                )
                
                spin_diff = calculate_difference_from_average_with_percentile(
                    pitcher_avg_spin, 
                    college_data['spin_rate'] if college_data else None,
                    metric_name='spin_rate',
                    pitch_type=pitch_type
                )
                
                ivb_diff = calculate_difference_from_average_with_percentile(
                    pitcher_avg_ivb, 
                    college_data['ivb'] if college_data else None,
                    metric_name='ivb',
                    pitch_type=pitch_type
                )
                
                hb_diff = calculate_difference_from_average_with_percentile(
                    pitcher_avg_hb, 
                    college_data['hb'] if college_data else None,
                    metric_name='hb',
                    pitch_type=pitch_type,
                    pitcher_throws=pitcher_throws
                )
                
                rel_height_diff = calculate_difference_from_average_with_percentile(
                    pitcher_avg_rel_height, 
                    college_data['rel_height'] if college_data else None
                )
                
                rel_side_diff = calculate_difference_from_average_with_percentile(
                    pitcher_avg_rel_side, 
                    college_data['rel_side'] if college_data else None
                )
                
                extension_diff = calculate_difference_from_average_with_percentile(
                    pitcher_avg_extension, 
                    college_data['extension'] if college_data else None
                )
                
                # Provide both college averages AND percentiles
                level_comparisons[level] = {
                    'velocity': {
                        'college_avg': f"{college_averages['avg_velocity']:.1f}" if college_averages and college_averages.get('avg_velocity') else 'N/A',
                        'comparison': velocity_diff,
                        'difference': f"{velocity_diff['difference']:.0f}%" if velocity_diff else 'N/A'
                    },
                    'max_velocity': {
                        'college_avg': f"{college_max_velo_averages['avg_max_velocity']:.1f}" if college_max_velo_averages and college_max_velo_averages.get('avg_max_velocity') else 'N/A',
                        'comparison': max_velocity_diff,
                        'difference': f"{max_velocity_diff['difference']:.0f}%" if max_velocity_diff else 'N/A'
                    },
                    'spin': {
                        'college_avg': f"{college_averages['avg_spin_rate']:.0f}" if college_averages and college_averages.get('avg_spin_rate') else 'N/A',
                        'comparison': spin_diff,
                        'difference': f"{spin_diff['difference']:.0f}%" if spin_diff else 'N/A'
                    },
                    'ivb': {
                        'college_avg': f"{college_averages['avg_ivb']:.1f}" if college_averages and college_averages.get('avg_ivb') else 'N/A',
                        'comparison': ivb_diff,
                        'difference': f"{ivb_diff['difference']:.0f}%" if ivb_diff else 'N/A'
                    },
                    'hb': {
                        'college_avg': f"{college_averages['avg_hb']:.1f}" if college_averages and college_averages.get('avg_hb') else 'N/A',
                        'comparison': hb_diff,
                        'difference': f"{hb_diff['difference']:.0f}%" if hb_diff else 'N/A'
                    },
                    'rel_height': {
                        'college_avg': f"{college_averages['avg_rel_height']:.1f}" if college_averages and college_averages.get('avg_rel_height') else 'N/A',
                        'comparison': rel_height_diff,
                        'difference': f"{rel_height_diff['difference']:.0f}%" if rel_height_diff else 'N/A'
                    },
                    'rel_side': {
                        'college_avg': f"{college_averages['avg_rel_side']:.1f}" if college_averages and college_averages.get('avg_rel_side') else 'N/A',
                        'comparison': rel_side_diff,
                        'difference': f"{rel_side_diff['difference']:.0f}%" if rel_side_diff else 'N/A'
                    },
                    'extension': {
                        'college_avg': f"{college_averages['avg_extension']:.1f}" if college_averages and college_averages.get('avg_extension') else 'N/A',
                        'comparison': extension_diff,
                        'difference': f"{extension_diff['difference']:.0f}%" if extension_diff else 'N/A'
                    }
                }
            
            # Same structure as before, but add is_first flag
            multi_level_breakdown.append({
                'name': pitch_type,
                'count': count,
                'is_first': i == 0,  # Add this flag to identify first pitch type
                'pitcher_velocity': f"{pitcher_avg_velocity:.1f}" if pitcher_avg_velocity else 'N/A',
                'pitcher_max_velocity': f"{pitcher_max_velocity:.1f}" if pitcher_max_velocity else 'N/A',
                'pitcher_spin': f"{pitcher_avg_spin:.0f}" if pitcher_avg_spin else 'N/A',
                'pitcher_ivb': f"{pitcher_avg_ivb:.1f}" if pitcher_avg_ivb else 'N/A',
                'pitcher_hb': f"{pitcher_avg_hb:.1f}" if pitcher_avg_hb else 'N/A',
                'pitcher_rel_height': f"{pitcher_avg_rel_height:.1f}" if pitcher_avg_rel_height else 'N/A',
                'pitcher_rel_side': f"{pitcher_avg_rel_side:.1f}" if pitcher_avg_rel_side else 'N/A',
                'pitcher_extension': f"{pitcher_avg_extension:.1f}" if pitcher_avg_extension else 'N/A',
                'level_comparisons': level_comparisons,
                'comparison_level': pitcher_comparison_level
            })
        
        return multi_level_breakdown

    except Exception as e:
        print(f"Error getting multi-level percentile comparisons: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


# Add this new function to get pitcher's competition level from Info table
def get_pitcher_competition_level(pitcher_name):
    """Get the competition level for a specific pitcher from the Info table"""
    try:
        query = """
        SELECT Comp
        FROM `V1PBRInfo.Info`
        WHERE Prospect = @pitcher_name
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pitcher_name", "STRING", pitcher_name),
            ]
        )
        
        result = client.query(query, job_config=job_config)
        row = list(result)
        
        if row and row[0].Comp:
            return row[0].Comp
        else:
            return 'D1'  # Default to D1 if no competition level found
            
    except Exception as e:
        print(f"Error getting competition level for {pitcher_name}: {str(e)}")
        return 'D1'  # Default to D1 on error

def get_all_pitcher_data_batch(selected_date):
    """Get all pitcher data in one query instead of individual queries"""
    try:
        batch_query = """
        SELECT *
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher IS NOT NULL
        ORDER BY Pitcher, PitchNo
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
            ]
        )
        
        result = client.query(batch_query, job_config=job_config)
        
        # Group by pitcher
        pitcher_data_map = {}
        for row in result:
            pitcher_name = row.Pitcher
            if pitcher_name not in pitcher_data_map:
                pitcher_data_map[pitcher_name] = []
            pitcher_data_map[pitcher_name].append(dict(row))
        
        return pitcher_data_map
        
    except Exception as e:
        print(f"Error in batch pitcher query: {str(e)}")
        return {}


def get_pitcher_stuffplus_data(pitcher_name, selected_date):
    """Get Stuffplus data for a specific pitcher and date from BigQuery"""
    try:
        stuffplus_query = """
        SELECT 
            PitchNo,
            TaggedPitchType,
            RelSpeed,
            SpinRate,
            InducedVertBreak,
            HorzBreak,
            StuffplusRight,
            StuffplusLeft,
            StuffplusAverage
        FROM `V1PBR.Test`
        WHERE CAST(Date AS STRING) = @date
        AND Pitcher = @pitcher
        AND StuffplusAverage IS NOT NULL
        ORDER BY PitchNo
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "STRING", selected_date),
                bigquery.ScalarQueryParameter("pitcher", "STRING", pitcher_name),
            ]
        )
        
        result = client.query(stuffplus_query, job_config=job_config)
        
        # Convert to list of dictionaries
        stuffplus_data = []
        for row in result:
            stuffplus_data.append({
                'pitch_no': row.PitchNo,
                'pitch_type': row.TaggedPitchType,
                'velocity': row.RelSpeed,
                'spin_rate': row.SpinRate,
                'ivb': row.InducedVertBreak,
                'hb': row.HorzBreak,
                'stuffplus_right': row.StuffplusRight,
                'stuffplus_left': row.StuffplusLeft,
                'stuffplus_average': row.StuffplusAverage
            })
        
        return stuffplus_data
        
    except Exception as e:
        print(f"Error getting Stuffplus data for {pitcher_name}: {str(e)}")
        return []

def calculate_stuffplus_summary(stuffplus_data):
    """Calculate summary statistics for Stuffplus scores"""
    if not stuffplus_data:
        return None
    
    try:
        # Group by pitch type
        pitch_type_summary = {}
        
        for pitch in stuffplus_data:
            pitch_type = pitch['pitch_type']
            if pitch_type not in pitch_type_summary:
                pitch_type_summary[pitch_type] = {
                    'count': 0,
                    'stuffplus_right_scores': [],
                    'stuffplus_left_scores': [],
                    'stuffplus_average_scores': []
                }
            
            pitch_type_summary[pitch_type]['count'] += 1
            
            if pitch['stuffplus_right'] is not None:
                pitch_type_summary[pitch_type]['stuffplus_right_scores'].append(pitch['stuffplus_right'])
            if pitch['stuffplus_left'] is not None:
                pitch_type_summary[pitch_type]['stuffplus_left_scores'].append(pitch['stuffplus_left'])
            if pitch['stuffplus_average'] is not None:
                pitch_type_summary[pitch_type]['stuffplus_average_scores'].append(pitch['stuffplus_average'])
        
        # Calculate averages for each pitch type
        summary_by_pitch_type = []
        
        for pitch_type, data in pitch_type_summary.items():
            avg_right = sum(data['stuffplus_right_scores']) / len(data['stuffplus_right_scores']) if data['stuffplus_right_scores'] else None
            avg_left = sum(data['stuffplus_left_scores']) / len(data['stuffplus_left_scores']) if data['stuffplus_left_scores'] else None
            avg_overall = sum(data['stuffplus_average_scores']) / len(data['stuffplus_average_scores']) if data['stuffplus_average_scores'] else None
            
            summary_by_pitch_type.append({
                'pitch_type': pitch_type,
                'count': data['count'],
                'avg_stuffplus_right': f"{avg_right:.0f}" if avg_right else 'N/A',
                'avg_stuffplus_left': f"{avg_left:.0f}" if avg_left else 'N/A',
                'avg_stuffplus_overall': f"{avg_overall:.0f}" if avg_overall else 'N/A'
            })
        
        # Sort by count (most thrown pitches first)
        summary_by_pitch_type.sort(key=lambda x: x['count'], reverse=True)
        
        # Calculate overall averages
        all_right_scores = [pitch['stuffplus_right'] for pitch in stuffplus_data if pitch['stuffplus_right'] is not None]
        all_left_scores = [pitch['stuffplus_left'] for pitch in stuffplus_data if pitch['stuffplus_left'] is not None]
        all_average_scores = [pitch['stuffplus_average'] for pitch in stuffplus_data if pitch['stuffplus_average'] is not None]
        
        overall_avg_right = sum(all_right_scores) / len(all_right_scores) if all_right_scores else None
        overall_avg_left = sum(all_left_scores) / len(all_left_scores) if all_left_scores else None
        overall_avg_overall = sum(all_average_scores) / len(all_average_scores) if all_average_scores else None
        
        return {
            'by_pitch_type': summary_by_pitch_type,
            'overall': {
                'avg_stuffplus_right': f"{overall_avg_right:.0f}" if overall_avg_right else 'N/A',
                'avg_stuffplus_left': f"{overall_avg_left:.0f}" if overall_avg_left else 'N/A',
                'avg_stuffplus_overall': f"{overall_avg_overall:.0f}" if overall_avg_overall else 'N/A',
                'total_pitches': len(stuffplus_data)
            }
        }
        
    except Exception as e:
        print(f"Error calculating Stuffplus summary: {str(e)}")
        return None

def calculate_thunder_data(pitch_data):
    try:
        if not pitch_data:
            return None, None
        
        # Group pitches by type
        pitch_type_data = {}
        
        for pitch in pitch_data:
            pitch_type = pitch.get('TaggedPitchType', 'Unknown')
            if pitch_type not in pitch_type_data:
                pitch_type_data[pitch_type] = []
            
            # FILTER OUT PITCHES WITH MISSING CRITICAL METRICS
            # Check for required metrics that Stuffplus needs
            required_metrics = ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak']
            has_all_metrics = all(
                pitch.get(metric) is not None and 
                str(pitch.get(metric)).upper() != 'N/A' and
                pitch.get(metric) != ''
                for metric in required_metrics
            )
            
            # Only include pitches with Stuffplus scores AND all required metrics
            action_score = pitch.get('StuffplusAverage') or pitch.get('Action', 0)
            if action_score is not None and has_all_metrics:
                pitch_copy = pitch.copy()
                pitch_copy['action_score'] = action_score
                pitch_type_data[pitch_type].append(pitch_copy)
        
        thunder_summary = []
        thunder_movement_data = {'average': {}, 'thunder': {}}
        
        for pitch_type, pitches in pitch_type_data.items():
            if len(pitches) < 3:  # Need at least 3 pitches to get top 25%
                continue
            
            # Sort pitches by action score (highest first)
            sorted_pitches = sorted(pitches, key=lambda x: x['action_score'], reverse=True)
            
            # Get top 25% (thunder) pitches
            thunder_count = max(1, len(sorted_pitches) // 4)
            thunder_pitches = sorted_pitches[:thunder_count]
            
            # Calculate averages for all pitches (now all have complete data)
            def safe_avg(values):
                # Additional filtering to ensure no None/N/A values slip through
                filtered = []
                for v in values:
                    if v is not None and str(v).upper() != 'N/A' and v != '':
                        try:
                            filtered.append(float(v))
                        except (ValueError, TypeError):
                            continue
                return sum(filtered) / len(filtered) if filtered else None
            
            # Average stats (all pitches with complete data)
            avg_velocity = safe_avg([p.get('RelSpeed') for p in pitches])
            avg_spin_rate = safe_avg([p.get('SpinRate') for p in pitches])
            avg_ivb = safe_avg([p.get('InducedVertBreak') for p in pitches])
            avg_hb = safe_avg([p.get('HorzBreak') for p in pitches])
            avg_rel_height = safe_avg([p.get('RelHeight') for p in pitches])
            avg_rel_side = safe_avg([p.get('RelSide') for p in pitches])
            avg_extension = safe_avg([p.get('Extension') for p in pitches])
            
            # Thunder stats (top 25% with complete data)
            thunder_velocity = safe_avg([p.get('RelSpeed') for p in thunder_pitches])
            thunder_spin_rate = safe_avg([p.get('SpinRate') for p in thunder_pitches])
            thunder_ivb = safe_avg([p.get('InducedVertBreak') for p in thunder_pitches])
            thunder_hb = safe_avg([p.get('HorzBreak') for p in thunder_pitches])
            thunder_rel_height = safe_avg([p.get('RelHeight') for p in thunder_pitches])
            thunder_rel_side = safe_avg([p.get('RelSide') for p in thunder_pitches])
            thunder_extension = safe_avg([p.get('Extension') for p in thunder_pitches])
            
            # Only include pitch types where we have valid averages for core metrics
            if all(metric is not None for metric in [avg_velocity, avg_spin_rate, avg_ivb, avg_hb]):
                
                # Store movement data for plotting
                thunder_movement_data['average'][pitch_type] = {
                    'hb': avg_hb,
                    'ivb': avg_ivb,
                    'rel_side': avg_rel_side,
                    'rel_height': avg_rel_height
                }
                
                thunder_movement_data['thunder'][pitch_type] = {
                    'hb': thunder_hb,
                    'ivb': thunder_ivb,
                    'rel_side': thunder_rel_side,
                    'rel_height': thunder_rel_height
                }
                
                thunder_summary.append({
                    'pitch_type': pitch_type,
                    'avg_count': f"{len(pitches)}",
                    'avg_velocity': f"{avg_velocity:.1f}" if avg_velocity else 'N/A',
                    'avg_spin_rate': f"{avg_spin_rate:.0f}" if avg_spin_rate else 'N/A',
                    'avg_ivb': f"{avg_ivb:.1f}" if avg_ivb else 'N/A',
                    'avg_hb': f"{avg_hb:.1f}" if avg_hb else 'N/A',
                    'avg_rel_height': f"{avg_rel_height:.1f}" if avg_rel_height else 'N/A',
                    'avg_rel_side': f"{avg_rel_side:.1f}" if avg_rel_side else 'N/A',
                    'avg_extension': f"{avg_extension:.1f}" if avg_extension else 'N/A',
                    'thunder_count': f"{thunder_count}",
                    'thunder_velocity': f"{thunder_velocity:.1f}" if thunder_velocity else 'N/A',
                    'thunder_spin_rate': f"{thunder_spin_rate:.0f}" if thunder_spin_rate else 'N/A',
                    'thunder_ivb': f"{thunder_ivb:.1f}" if thunder_ivb else 'N/A',
                    'thunder_hb': f"{thunder_hb:.1f}" if thunder_hb else 'N/A',
                    'thunder_rel_height': f"{thunder_rel_height:.1f}" if thunder_rel_height else 'N/A',
                    'thunder_rel_side': f"{thunder_rel_side:.1f}" if thunder_rel_side else 'N/A',
                    'thunder_extension': f"{thunder_extension:.1f}" if thunder_extension else 'N/A'
                })
        
        # Sort by pitch count (most thrown first)
        thunder_summary.sort(key=lambda x: int(x['avg_count']), reverse=True)
        
        return thunder_summary, thunder_movement_data
        
    except Exception as e:
        print(f"Error calculating thunder data: {str(e)}")
        return None, None

def generate_thunder_movement_plot_svg(thunder_movement_data, width=600, height=450):  # INCREASED HEIGHT
    """Generate TALLER SVG showing average vs thunder movement for each pitch type"""
    try:
        if not thunder_movement_data or not thunder_movement_data.get('average'):
            return None
        
        # Colors for pitch types (same as main movement plot)
        colors = {
            'ChangeUp': '#059669', 'Curveball': '#1D4ED8', 'Cutter': '#BE185D',
            'Fastball': '#DC2626', 'Knuckleball': '#9333EA', 'Sinker': '#EA580C',
            'Slider': '#7C3AED', 'Splitter': '#0891B2', 'Sweeper': '#F59E0B',
            'Four-Seam': '#DC2626', '4-Seam': '#DC2626', 'Two-Seam': '#EA580C'
        }
        
        # Set up plot dimensions - TALLER VERSION
        margin_left = 50      
        margin_right = 120    
        margin_top = 50       
        margin_bottom = 50    
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom  # This will be larger now
        
        # Scale functions - EXTENDED TO 30 like main movement plot
        x_min, x_max = -30, 30
        y_min, y_max = -30, 30
        
        def scale_x(x):
            return margin_left + (x - x_min) / (x_max - x_min) * plot_width
        
        def scale_y(y):
            return margin_top + plot_height - (y - y_min) / (y_max - y_min) * plot_height
        
        # Start SVG with improved styling
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '<style>',
            '.axis-line { stroke: #990000; stroke-width: 2; }',
            '.grid-line { stroke: rgba(0,0,0,0.2); stroke-width: 1; }',
            '.axis-text { font-family: Arial, sans-serif; font-size: 9px; fill: black; }',
            '.axis-title { font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; fill: black; }',
            '.legend-text { font-family: Arial, sans-serif; font-size: 8px; fill: black; }',
            '.plot-title { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; fill: #1a1a1a; text-anchor: start; }',
            '.plot-subtitle { font-family: Arial, sans-serif; font-size: 9px; fill: #666666; text-anchor: start; font-style: italic; }',
            '.plot-border { stroke: black; stroke-width: 2; fill: none; }',
            '.avg-point { fill-opacity: 0.6; stroke: white; stroke-width: 1; }',
            '.thunder-point { fill-opacity: 0.9; stroke: #228B22; stroke-width: 2; }',
            '</style>',
            '</defs>',
            
            # Background
            f'<rect width="{width}" height="{height}" fill="white"/>',
            
            # Title and subtitle
            f'<text x="{margin_left}" y="20" class="plot-title">Thunder vs Average Movement</text>',
            f'<text x="{margin_left}" y="32" class="plot-subtitle">Pitcher\'s Perspective</text>',
        ]
        
        # Grid and axes - BACK TO DETAILED GRID LINES EVERY 5 UNITS
        for x in range(-25, 31, 5):  # Every 5 units like main movement plot
            x_pos = scale_x(x)
            line_class = 'axis-line' if x == 0 else 'grid-line'
            svg_parts.append(f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + plot_height}" class="{line_class}"/>')
            if x != 0:
                svg_parts.append(f'<text x="{x_pos}" y="{margin_top + plot_height + 12}" class="axis-text" text-anchor="middle">{x}</text>')
        
        for y in range(-25, 31, 5):  # Every 5 units like main movement plot
            y_pos = scale_y(y)
            line_class = 'axis-line' if y == 0 else 'grid-line'
            svg_parts.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_width}" y2="{y_pos}" class="{line_class}"/>')
            if y != 0:
                svg_parts.append(f'<text x="{margin_left - 8}" y="{y_pos + 3}" class="axis-text" text-anchor="end">{y}</text>')
        
        # Plot border
        svg_parts.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" class="plot-border"/>')
        
        # Plot points
        legend_y = margin_top + 60  # Start legend lower
        
        for pitch_type in thunder_movement_data['average'].keys():
            color = colors.get(pitch_type, '#666666')
            
            # Average point
            avg_data = thunder_movement_data['average'][pitch_type]
            avg_x = scale_x(avg_data['hb'])
            avg_y = scale_y(avg_data['ivb'])
            svg_parts.append(f'<circle cx="{avg_x}" cy="{avg_y}" r="4" fill="{color}" class="avg-point"/>')
            
            # Thunder point (larger, more prominent)
            thunder_data = thunder_movement_data['thunder'][pitch_type]
            thunder_x = scale_x(thunder_data['hb'])
            thunder_y = scale_y(thunder_data['ivb'])
            svg_parts.append(f'<circle cx="{thunder_x}" cy="{thunder_y}" r="6" fill="{color}" class="thunder-point"/>')
            
            # Connection line (dashed like main plot)
            svg_parts.append(f'<line x1="{avg_x}" y1="{avg_y}" x2="{thunder_x}" y2="{thunder_y}" stroke="{color}" stroke-width="2" stroke-dasharray="3,3"/>')
            
            # Legend (improved layout)
            legend_x = margin_left + plot_width + 15
            svg_parts.extend([
                f'<circle cx="{legend_x}" cy="{legend_y}" r="3" fill="{color}" class="avg-point"/>',
                f'<circle cx="{legend_x + 20}" cy="{legend_y}" r="4" fill="{color}" class="thunder-point"/>',
                f'<text x="{legend_x + 28}" y="{legend_y + 3}" class="legend-text">{pitch_type}</text>'
            ])
            legend_y += 16
        
        # Legend header
        legend_header_y = margin_top + 42
        svg_parts.extend([
            f'<text x="{margin_left + plot_width + 15}" y="{legend_header_y}" class="legend-text" style="font-weight: bold;">Avg Thunder</text>'
        ])
        
        # Axis labels
        center_x = margin_left + plot_width/2
        center_y = margin_top + plot_height/2
        svg_parts.extend([
            f'<text x="{center_x}" y="{height - 15}" class="axis-title" text-anchor="middle">Horizontal Break (in)</text>',
            f'<text x="15" y="{center_y}" class="axis-title" text-anchor="middle" transform="rotate(-90, 15, {center_y})">Induced Vertical Break (in)</text>'
        ])
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
        
    except Exception as e:
        print(f"Error generating thunder movement plot: {str(e)}")
        return None

# Update the generate_pitcher_pdf function to include zone rate data
def generate_pitcher_pdf(pitcher_name, pitch_data, date, comparison_level=None):
    """Generate a PDF report for the pitcher using WeasyPrint with college comparisons and movement plot"""
    try:
        # Calculate summary stats
        if not pitch_data:
            print(f"No pitch data for {pitcher_name}")
            return None
            
        # Format pitcher name (convert "Smith, Jack" to "Jack Smith")
        if ', ' in pitcher_name:
            last_name, first_name = pitcher_name.split(', ', 1)
            formatted_name = f"{first_name} {last_name}"
        else:
            formatted_name = pitcher_name
        
        # Get competition level from Info table if not provided
        if comparison_level is None:
            comparison_level = get_pitcher_competition_level(pitcher_name)
            print(f"Retrieved competition level for {formatted_name}: {comparison_level}")
        
        # Get grad year from Info table
        grad_year = None
        try:
            grad_year_query = """
            SELECT GradYear
            FROM `V1PBRInfo.Info`
            WHERE Prospect = @pitcher_name
            LIMIT 1
            """
            
            grad_year_job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("pitcher_name", "STRING", pitcher_name),
                ]
            )
            
            grad_year_result = client.query(grad_year_query, job_config=grad_year_job_config)
            grad_year_row = list(grad_year_result)
            
            if grad_year_row and grad_year_row[0].GradYear:
                grad_year = grad_year_row[0].GradYear
        except Exception as e:
            print(f"Error getting grad year for {pitcher_name}: {str(e)}")
        
        # Determine pitcher handedness from the data
        pitcher_throws = 'Right'  # Default to right-handed
        for pitch in pitch_data:
            if pitch.get('PitcherThrows'):
                pitcher_throws = pitch.get('PitcherThrows')
                break
        
        print(f"Pitcher {formatted_name} throws: {pitcher_throws}")
            
        # Group pitches by type and calculate averages
        pitch_type_data = {}
        
        for pitch in pitch_data:
            pitch_type = pitch.get('TaggedPitchType', 'Unknown')
            if pitch_type not in pitch_type_data:
                pitch_type_data[pitch_type] = {
                    'pitches': [],
                    'count': 0
                }
            pitch_type_data[pitch_type]['pitches'].append(pitch)
            pitch_type_data[pitch_type]['count'] += 1
        
        # Calculate averages for each pitch type WITH college comparisons
        pitch_type_breakdown = []
        
        # Define priority order - Fastball first, then by general usage/importance
        priority_types = ['Fastball', 'Sinker', 'Cutter', 'Slider', 'Curveball', 'ChangeUp', 'Sweeper', 'Splitter', 'Knuckleball']
        
        # Sort pitch types with priority
        sorted_pitch_types = []
        
        # Add priority types first (if they exist)
        for priority_type in priority_types:
            if priority_type in pitch_type_data:
                sorted_pitch_types.append(priority_type)
        
        # Add remaining types alphabetically
        remaining_types = [pt for pt in pitch_type_data.keys() if pt not in sorted_pitch_types]
        sorted_pitch_types.extend(sorted(remaining_types))
        
        for pitch_type in sorted_pitch_types:
            pitches = pitch_type_data[pitch_type]['pitches']
            count = pitch_type_data[pitch_type]['count']
            
            # Calculate averages for this pitch type
            velocities = [p.get('RelSpeed', 0) for p in pitches if p.get('RelSpeed')]
            spin_rates = [p.get('SpinRate', 0) for p in pitches if p.get('SpinRate')]
            ivbs = [p.get('InducedVertBreak', 0) for p in pitches if p.get('InducedVertBreak')]
            hbs = [p.get('HorzBreak', 0) for p in pitches if p.get('HorzBreak')]
            rel_sides = [p.get('RelSide', 0) for p in pitches if p.get('RelSide')]
            rel_heights = [p.get('RelHeight', 0) for p in pitches if p.get('RelHeight')]
            extensions = [p.get('Extension', 0) for p in pitches if p.get('Extension')]
            
            # Calculate pitcher's averages
            pitcher_avg_velocity = sum(velocities)/len(velocities) if velocities else None
            pitcher_avg_spin = sum(spin_rates)/len(spin_rates) if spin_rates else None
            pitcher_avg_ivb = sum(ivbs)/len(ivbs) if ivbs else None
            pitcher_avg_hb = sum(hbs)/len(hbs) if hbs else None
            pitcher_avg_rel_side = sum(rel_sides)/len(rel_sides) if rel_sides else None
            pitcher_avg_rel_height = sum(rel_heights)/len(rel_heights) if rel_heights else None
            pitcher_avg_extension = sum(extensions)/len(extensions) if extensions else None
            
            # Get college averages for comparison (with pitcher handedness)
            college_averages = get_college_averages_cached(pitch_type, comparison_level, pitcher_throws)
            
            # Calculate comparisons
            velocity_comp = calculate_percentile(pitcher_avg_velocity, 
                               college_averages['avg_velocity'] if college_averages else None,
                               metric_name='velocity',
                               pitch_type=pitch_type)
            
            spin_comp = calculate_percentile(pitcher_avg_spin, 
                           college_averages['avg_spin_rate'] if college_averages else None,
                           metric_name='spin_rate',
                           pitch_type=pitch_type)
            
            ivb_comp = calculate_percentile(pitcher_avg_ivb, 
                          college_averages['avg_ivb'] if college_averages else None,
                          metric_name='ivb',
                          pitch_type=pitch_type)
            
            hb_comp = calculate_percentile(pitcher_avg_hb, 
                         college_averages['avg_hb'] if college_averages else None,
                         metric_name='hb',
                         pitch_type=pitch_type,
                         pitcher_throws=pitcher_throws)
            
            rel_side_comp = calculate_percentile(pitcher_avg_rel_side, 
                                               college_averages['avg_rel_side'] if college_averages else None)
            
            rel_height_comp = calculate_percentile(pitcher_avg_rel_height, 
                                                 college_averages['avg_rel_height'] if college_averages else None)
            
            extension_comp = calculate_percentile(pitcher_avg_extension, 
                                                college_averages['avg_extension'] if college_averages else None)
            
            pitch_type_breakdown.append({
                'name': pitch_type,
                'count': count,
                'avg_velocity': f"{pitcher_avg_velocity:.1f}" if pitcher_avg_velocity else 'N/A',
                'avg_spin': f"{pitcher_avg_spin:.0f}" if pitcher_avg_spin else 'N/A',
                'avg_ivb': f"{pitcher_avg_ivb:.1f}" if pitcher_avg_ivb else 'N/A',
                'avg_hb': f"{pitcher_avg_hb:.1f}" if pitcher_avg_hb else 'N/A',
                'avg_rel_side': f"{pitcher_avg_rel_side:.1f}" if pitcher_avg_rel_side else 'N/A',
                'avg_rel_height': f"{pitcher_avg_rel_height:.1f}" if pitcher_avg_rel_height else 'N/A',
                'avg_extension': f"{pitcher_avg_extension:.1f}" if pitcher_avg_extension else 'N/A',
                # College comparison data - always include, even if N/A
                'college_velocity': f"{college_averages['avg_velocity']:.1f}" if college_averages and college_averages['avg_velocity'] else 'N/A',
                'college_spin': f"{college_averages['avg_spin_rate']:.0f}" if college_averages and college_averages['avg_spin_rate'] else 'N/A',
                'college_ivb': f"{college_averages['avg_ivb']:.1f}" if college_averages and college_averages['avg_ivb'] else 'N/A',
                'college_hb': f"{college_averages['avg_hb']:.1f}" if college_averages and college_averages['avg_hb'] else 'N/A',
                'college_rel_side': f"{college_averages['avg_rel_side']:.1f}" if college_averages and college_averages['avg_rel_side'] else 'N/A',
                'college_rel_height': f"{college_averages['avg_rel_height']:.1f}" if college_averages and college_averages['avg_rel_height'] else 'N/A',
                'college_extension': f"{college_averages['avg_extension']:.1f}" if college_averages and college_averages['avg_extension'] else 'N/A',
                # Comparison indicators - always include, even if None
                'velocity_comp': velocity_comp,
                'spin_comp': spin_comp,
                'ivb_comp': ivb_comp,
                'hb_comp': hb_comp,
                'rel_side_comp': rel_side_comp,
                'rel_height_comp': rel_height_comp,
                'extension_comp': extension_comp,
                'has_college_data': college_averages is not None
            })
        
        summary_stats = {
            'pitch_type_breakdown': pitch_type_breakdown,
            'comparison_level': comparison_level,
            'pitcher_throws': pitcher_throws
        }

        multi_level_stats = get_multi_level_comparisons(pitch_data, pitcher_throws)
        
        # Generate SVG plots with debugging
        print(f"Generating plots for {formatted_name}...")
        movement_plot_svg = generate_movement_plot_svg(pitch_data, comparison_level=comparison_level)
        pitch_location_plot_svg = generate_pitch_location_plot_svg(pitch_data)
        
        # Calculate zone rates
        zone_rate_data = calculate_zone_rates(pitch_data, comparison_level, pitcher_throws)
        stuffplus_data = get_pitcher_stuffplus_data(pitcher_name, date)
        stuffplus_summary = calculate_stuffplus_summary(stuffplus_data) if stuffplus_data else None

        thunder_summary, thunder_movement_data = calculate_thunder_data(pitch_data)
        thunder_movement_plot_svg = None

        if thunder_summary and thunder_movement_data:
            thunder_movement_plot_svg = generate_thunder_movement_plot_svg(thunder_movement_data)
        
        # Debug plot generation
        print(f"Movement plot generated: {movement_plot_svg is not None}")
        print(f"Pitch location plot generated: {pitch_location_plot_svg is not None}")
        print(f"Zone rate data calculated: {zone_rate_data is not None}")
        
        if pitch_location_plot_svg:
            print(f"Pitch location SVG length: {len(pitch_location_plot_svg)} characters")
        else:
            print("Pitch location plot is None - checking data structure...")
            if pitch_data:
                sample_pitch = pitch_data[0]
                available_fields = list(sample_pitch.keys())
                print(f"Available fields in pitch data: {available_fields}")
                # Check for common location field variations
                location_fields = [field for field in available_fields if 'loc' in field.lower() or 'plate' in field.lower()]
                print(f"Location-related fields found: {location_fields}")
        
        print(f"Generating PDF for {formatted_name} ({pitcher_throws}) with {len(pitch_data)} pitches and {comparison_level} comparisons")
        
        # Read HTML template
        try:
            with open('pitcher_report.html', 'r', encoding='utf-8') as file:
                html_template = file.read()
        except FileNotFoundError:
            print("Error: pitcher_report.html not found. Make sure it's in the same directory as app.py")
            return None
        
        # Render template with data using Jinja2
        template = Template(html_template)
        rendered_html = template.render(
            pitcher_name=formatted_name,
            date=date,
            grad_year=grad_year,
            summary_stats=summary_stats,
            pitch_data=pitch_data,
            multi_level_stats=multi_level_stats,
            movement_plot_svg=movement_plot_svg,
            pitch_location_plot_svg=pitch_location_plot_svg,
            zone_rate_data=zone_rate_data,  # Add zone rate data to template
            stuffplus_data=stuffplus_data,  # ADD THIS
            stuffplus_summary=stuffplus_summary,  # ADD THIS
            thunder_data=thunder_summary,               # ADD THIS
            thunder_summary=thunder_summary,            # ADD THIS
            thunder_movement_plot_svg=thunder_movement_plot_svg  # ADD THIS
        )
        
        # Generate PDF using WeasyPrint with proper base_url for static files
        try:
            # Get the absolute path to the current directory so WeasyPrint can find static files
            base_url = f"file://{os.path.abspath('.')}/"
            print(f"Using base_url: {base_url}")
            
            # Check if static files exist
            static_dir = os.path.join(os.getcwd(), 'static')
            if not os.path.exists(static_dir):
                print(f"Warning: Static directory not found at {static_dir}")
                os.makedirs(static_dir, exist_ok=True)
                print(f"Created static directory at {static_dir}")
            
            # Check for required images
            required_images = ['pbr.png', 'miss.png']
            for image_name in required_images:
                image_path = os.path.join(static_dir, image_name)
                if os.path.exists(image_path):
                    print(f"Found image at: {image_path}")
                else:
                    print(f"Warning: Image not found at {image_path}")
            
            html_doc = weasyprint.HTML(string=rendered_html, base_url=base_url)
            pdf_bytes = html_doc.write_pdf()
            print(f"PDF generated successfully for {formatted_name}")
            return pdf_bytes
        except Exception as e:
            print(f"WeasyPrint error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"Error generating PDF for {pitcher_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/api/update-email-status', methods=['POST'])
def update_email_status_api():
    """Manually update email status for prospects"""
    try:
        data = request.get_json()
        required_fields = ['date', 'prospect_name', 'status']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields: date, prospect_name, status'}), 400
        
        success = update_email_status(
            data['prospect_name'],
            data['date'],
            data['status'],
            data.get('email')
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f"Status updated for {data['prospect_name']}"
            })
        else:
            return jsonify({'error': 'Failed to update status'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear all caches"""
    global pdf_cache, college_data_cache
    pdf_cache.clear()
    college_data_cache.clear()
    return jsonify({'success': True, 'message': 'Caches cleared'})

if __name__ == '__main__':
    # Test email system on startup
    print("Testing email system...")
    if test_email_connection(email_config):
        print("✓ Email system ready for production!")
        print(f"✓ Estimated time for 300 emails: {estimate_send_time(300, email_config)}")
    else:
        print("✗ Email system needs configuration")
    
    print("Starting Flask server...")
    print("Make sure harvard-baseball-13fab221b2d4.json is in the same directory")
    print("Make sure templates/index.html exists")
    print("Make sure pitcher_report.html exists")
    app.run(debug=True, host='0.0.0.0', port=5000)