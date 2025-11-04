# Secure Login System
from flask import Blueprint, render_template, request, session, flash, jsonify, redirect, url_for, abort
import logging
import secrets
import time
from werkzeug.security import check_password_hash
from functools import wraps
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from app.config.database import get_oracle_credentials
from app.oracle_helpers import OracleError, create_connection_from_credentials
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create Blueprint for login-related routes
login_bp = Blueprint('login', __name__)

# Security Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
SESSION_TIMEOUT = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 900  # 15 minutes
PASSWORD_MIN_LENGTH = 8
ADMIN_USERS = {user.strip() for user in os.getenv('ADMIN_USERS', 'visitor').split(',') if user.strip()}

# In-memory storage for failed attempts (in production, use Redis)
failed_attempts = {}
account_lockouts = {}

# Rate limiting configuration
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

def get_db_connection():
    """Establish Oracle database connection."""
    try:
        connection = create_connection_from_credentials(get_oracle_credentials())
        logging.info("Database connection successful")
        return connection
    except RuntimeError as exc:
        logging.error(f"Oracle driver error: {exc}")
        flash(str(exc), "error")
        return None
    except OracleError as e:
        logging.error(f"Database connection error: {e}")
        flash(f"Database connection error: {e}", "error")
        return None

def is_account_locked(username, ip_address):
    """Check if account is locked due to too many failed attempts"""
    key = f"{username}:{ip_address}"
    current_time = time.time()
    
    if key in account_lockouts:
        if current_time < account_lockouts[key]:
            return True
        else:
            # Lockout expired, remove it
            del account_lockouts[key]
            if key in failed_attempts:
                del failed_attempts[key]
    
    return False

def record_failed_attempt(username, ip_address):
    """Record a failed login attempt"""
    key = f"{username}:{ip_address}"
    current_time = time.time()
    
    if key not in failed_attempts:
        failed_attempts[key] = {'count': 0, 'first_attempt': current_time}
    
    failed_attempts[key]['count'] += 1
    failed_attempts[key]['last_attempt'] = current_time
    
    # If too many attempts, lock the account
    if failed_attempts[key]['count'] >= MAX_LOGIN_ATTEMPTS:
        account_lockouts[key] = current_time + LOCKOUT_DURATION
        logging.warning(f"Account locked for {username} from {ip_address} due to {failed_attempts[key]['count']} failed attempts")
        return True
    
    return False

def clear_failed_attempts(username, ip_address):
    """Clear failed attempts after successful login"""
    key = f"{username}:{ip_address}"
    if key in failed_attempts:
        del failed_attempts[key]
    if key in account_lockouts:
        del account_lockouts[key]

def validate_input(username, password):
    """Validate input for security"""
    if not username or not password:
        return False, "Username and password are required"
    
    if len(username) > 50 or len(password) > 100:
        return False, "Input too long"
    
    # Check for SQL injection patterns
    dangerous_patterns = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_', 'exec', 'execute']
    for pattern in dangerous_patterns:
        if pattern.lower() in username.lower() or pattern.lower() in password.lower():
            return False, "Invalid characters detected"
    
    return True, "Valid"

def generate_csrf_token():
    """Generate CSRF token"""
    return secrets.token_hex(32)

def verify_csrf_token(token):
    """Verify CSRF token"""
    if 'csrf_token' not in session:
        return False
    return secrets.compare_digest(session['csrf_token'], token)

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash("Please log in to access this page", "error")
            return redirect(url_for('login.login'))
        
        # Check session timeout
        if 'last_activity' in session:
            if time.time() - session['last_activity'] > SESSION_TIMEOUT:
                session.clear()
                flash("Session expired. Please log in again", "error")
                return redirect(url_for('login.login'))
        
        # Update last activity
        session['last_activity'] = time.time()
        return f(*args, **kwargs)
    return decorated_function

def sanitize_input(text):
    """Sanitize user input"""
    if not text:
        return ""
    return text.strip()[:100]  # Limit length and strip whitespace

def ensure_access_log_table(connection):
    """Ensure the user_access_log table exists."""
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            BEGIN
                EXECUTE IMMEDIATE '
                    CREATE TABLE user_access_log (
                        id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                        username VARCHAR2(50) NOT NULL,
                        event_type VARCHAR2(20) NOT NULL,
                        event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ip_address VARCHAR2(64),
                        user_agent VARCHAR2(255)
                    )
                ';
            EXCEPTION
                WHEN OTHERS THEN
                    IF SQLCODE != -955 THEN
                        RAISE;
                    END IF;
            END;
            """
        )
        cursor.close()
    except Exception as exc:
        logging.error(f"Erreur lors de la vérification/creation de user_access_log : {exc}")

def log_user_access(connection, username, event_type, ip_address=None, user_agent=None):
    """Enregistrer un événement d'accès utilisateur."""
    if connection is None:
        return
    try:
        ensure_access_log_table(connection)
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO user_access_log (username, event_type, ip_address, user_agent)
            VALUES (:username, :event_type, :ip_address, :user_agent)
            """,
            {
                'username': username,
                'event_type': event_type,
                'ip_address': (ip_address or '')[:64] or None,
                'user_agent': (user_agent or '')[:255] or None
            }
        )
        connection.commit()
        cursor.close()
    except Exception as exc:
        logging.error(f"Erreur lors de l'enregistrement de l'accès utilisateur : {exc}")

def require_admin(f):
    """Decorator to require administrator privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        username = session.get('username')
        if not username or username not in ADMIN_USERS:
            logging.warning(f"Tentative d'accès admin non autorisée par {username}")
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@login_bp.route('/')
def index():
    return redirect(url_for('login.login'))

@login_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")  # Rate limiting
def login():
    if request.method == 'POST':
        # Get client IP for security tracking
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'Invalid request format'}), 400
            
            username = sanitize_input(data.get('username', ''))
            password = sanitize_input(data.get('password', ''))
            csrf_token = data.get('csrf_token', '')
            # Validate input
            is_valid, error_msg = validate_input(username, password)
            if not is_valid:
                logging.warning(f"Invalid input from {client_ip}: {error_msg}")
                return jsonify({'success': False, 'message': error_msg}), 400
            
            # Check for account lockout
            if is_account_locked(username, client_ip):
                remaining_time = int(account_lockouts.get(f"{username}:{client_ip}", 0) - time.time())
                logging.warning(f"Account locked for {username} from {client_ip}")
                return jsonify({
                    'success': False, 
                    'message': f'Account temporarily locked. Try again in {remaining_time} seconds.'
                }), 423
            
            # Verify CSRF token
            if not verify_csrf_token(csrf_token):
                logging.warning(f"Invalid CSRF token from {client_ip}")
                return jsonify({'success': False, 'message': 'Invalid security token'}), 403
            
            # Database authentication
            connection = get_db_connection()
            if connection is None:
                logging.error("Failed to connect to database")
                return jsonify({'success': False, 'message': 'Database connection failed'}), 500
            
            try:
                cursor = connection.cursor()
                # Use parameterized query to prevent SQL injection
                query = "SELECT username, password_hash, is_active, last_login FROM users WHERE username = :1"
                cursor.execute(query, (username,))
                user = cursor.fetchone()

                if user and user[2]:  # Check if user exists and is active
                    stored_hash = user[1]
                    # Use secure password verification
                    if check_password_hash(stored_hash, password):
                        # Successful login
                        session['username'] = username
                        session['last_activity'] = time.time()
                        session['csrf_token'] = generate_csrf_token()
                        session['is_admin'] = username in ADMIN_USERS
                        
                        # Clear failed attempts
                        clear_failed_attempts(username, client_ip)
                        
                        # Update last login
                        cursor.execute(
                            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = :1",
                            (username,)
                        )
                        connection.commit()

                        try:
                            user_agent = request.user_agent.string if request.user_agent else None
                            log_user_access(connection, username, 'login', client_ip, user_agent)
                        except Exception as log_exc:
                            logging.error(f"Impossible d'enregistrer l'accès utilisateur : {log_exc}")

                        return jsonify({
                            'success': True,
                            'message': 'Login successful'
                        })
                    else:
                        # Failed login
                        record_failed_attempt(username, client_ip)
                        logging.warning(f"Failed login attempt for {username} from {client_ip}")
                        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
                else:
                    # User not found or inactive
                    record_failed_attempt(username, client_ip)
                    logging.warning(f"Login attempt for non-existent/inactive user {username} from {client_ip}")
                    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
                    
            except OracleError as e:
                logging.error(f"Database error during login: {e}")
                return jsonify({'success': False, 'message': 'Authentication service temporarily unavailable'}), 500
            finally:
                cursor.close()
                connection.close()
                
        except Exception as e:
            logging.error(f"Unexpected error during login: {e}")
            return jsonify({'success': False, 'message': 'An unexpected error occurred'}), 500
    
    # GET request - show login form with CSRF token
    csrf_token = generate_csrf_token()
    session['csrf_token'] = csrf_token
    return render_template('login.html', csrf_token=csrf_token)

@login_bp.route('/check-auth')
def check_auth():
    """Check if user is authenticated and session is valid"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    # Check session timeout
    if 'last_activity' in session:
        if time.time() - session['last_activity'] > SESSION_TIMEOUT:
            session.clear()
            return jsonify({'success': False, 'message': 'Session expired'})
    
    return jsonify({
        'success': True,
        'username': session['username']
    })

@login_bp.route('/homepage')
@require_auth
def homepage():
    """Protected homepage route"""
    username = session.get('username')
    is_admin = username in ADMIN_USERS if username else False
    session['is_admin'] = is_admin
    return render_template('homepage.html', username=username, is_admin=is_admin)

@login_bp.route('/user_access_dashboard')
@require_auth
@require_admin
def user_access_dashboard():
    """Dashboard to inspect user access activity"""
    return render_template('access_dashboard.html', username=session.get('username', ''))

@login_bp.route('/logout')
def logout():
    """Secure logout with session cleanup"""
    username = session.get('username', 'Unknown')
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)

    try:
        connection = get_db_connection()
        if connection:
            user_agent = request.user_agent.string if request.user_agent else None
            log_user_access(connection, username, 'logout', client_ip, user_agent)
            connection.close()
    except Exception as log_exc:
        logging.error(f"Impossible d'enregistrer la déconnexion : {log_exc}")

    # Clear all session data
    session.clear()

    # Generate new CSRF token for next login
    session['csrf_token'] = generate_csrf_token()
    
    logging.info(f"User {username} logged out from {client_ip}")
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('login.login'))

@login_bp.route('/security-info')
@require_auth
def security_info():
    """Display security information for authenticated users"""
    return jsonify({
        'session_timeout': SESSION_TIMEOUT,
        'max_login_attempts': MAX_LOGIN_ATTEMPTS,
        'lockout_duration': LOCKOUT_DURATION,
        'last_activity': session.get('last_activity', 0)
    })

@login_bp.route('/api/access_logs')
@require_auth
@require_admin
def api_access_logs():
    connection = get_db_connection()
    if connection is None:
        return jsonify({'error': "Impossible d'accéder à la base de données"}), 500

    try:
        ensure_access_log_table(connection)
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, username, event_type, event_time, ip_address, user_agent
            FROM user_access_log
            ORDER BY event_time DESC
            FETCH FIRST 100 ROWS ONLY
            """
        )
        raw_logs = cursor.fetchall()
        columns = [col[0].lower() for col in cursor.description]
        recent_logs = []
        last_ip_map = {}
        for row in raw_logs:
            record = dict(zip(columns, row))
            event_time = record.get('event_time')
            record['event_time'] = event_time.isoformat() if isinstance(event_time, datetime) else str(event_time)
            recent_logs.append(record)
            username = record.get('username')
            if username and username not in last_ip_map:
                last_ip_map[username] = record.get('ip_address')

        cursor.execute(
            """
            SELECT 
                u.username,
                u.email,
                u.last_login,
                SUM(CASE WHEN l.event_type = 'login' THEN 1 ELSE 0 END) AS login_count,
                MAX(l.event_time) AS last_event_time
            FROM users u
            LEFT JOIN user_access_log l ON l.username = u.username
            GROUP BY u.username, u.email, u.last_login
            ORDER BY u.username
            """
        )
        user_rows = cursor.fetchall()
        user_columns = [col[0].lower() for col in cursor.description]
        users_summary = []
        for row in user_rows:
            record = dict(zip(user_columns, row))
            last_login = record.get('last_login')
            last_event_time = record.get('last_event_time')
            username = record.get('username')

            users_summary.append({
                'username': username,
                'email': record.get('email'),
                'last_login': last_login.isoformat() if isinstance(last_login, datetime) else (str(last_login) if last_login else None),
                'login_count': int(record.get('login_count') or 0),
                'last_event_time': last_event_time.isoformat() if isinstance(last_event_time, datetime) else (str(last_event_time) if last_event_time else None),
                'last_ip': last_ip_map.get(username)
            })

        cursor.close()
        return jsonify({
            'users': users_summary,
            'recent_logs': recent_logs
        })
    except Exception as exc:
        logging.error(f"Erreur lors de la récupération des logs d'accès : {exc}")
        return jsonify({'error': 'Erreur lors de la récupération des logs'}), 500
    finally:
        connection.close()
