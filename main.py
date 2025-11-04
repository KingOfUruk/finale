import os
import secrets

from flask import Flask
from app.blueprints.login import login_bp
from app.blueprints.payroll import payroll_bp
from app.blueprints.hr_analytics import hr_analytics_bp
from app.blueprints.employee_performance import employee_performance_bp
from app.blueprints.ml_api import ml_bp


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Register blueprints
app.register_blueprint(login_bp)
app.register_blueprint(payroll_bp)
app.register_blueprint(hr_analytics_bp)
app.register_blueprint(employee_performance_bp)
app.register_blueprint(ml_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
