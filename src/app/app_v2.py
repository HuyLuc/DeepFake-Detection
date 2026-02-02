# src/app/app_v2.py
"""
DeepFake Detection Web App V2.0
Main Flask Application with API routes, database, and modern features
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import database
from src.app.models.database import init_db

# Import blueprints
from src.app.api.routes import api
from src.app.api.export_routes import export_api

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_app(config=None):
    """
    Application factory
    
    Args:
        config: Optional config dict to override defaults
    
    Returns:
        Flask application instance
    """
    logger.info("="*60)
    logger.info("🚀 Creating DeepFake Detection Web App V2.0")
    logger.info("="*60)
    
    # Create Flask app
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static'
    )
    
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'deepfake-detection-v2-secret-key')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'data', 'temp_uploads')
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Override with custom config
    if config:
        app.config.update(config)
    
    # ==========================================================================
    # EXTENSIONS
    # ==========================================================================
    
    # CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
        }
    })
    
    # Database
    init_db(app)
    
    # ==========================================================================
    # REGISTER BLUEPRINTS
    # ==========================================================================
    app.register_blueprint(api)
    app.register_blueprint(export_api)
    
    logger.info("✅ Registered API blueprints")
    
    # ==========================================================================
    # TEMPLATE ROUTES
    # ==========================================================================
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/history')
    def history_page():
        """History page"""
        return render_template('history.html')
    
    @app.route('/about')
    def about():
        """About page"""
        return render_template('about.html')
    
    # ==========================================================================
    # ERROR HANDLERS
    # ==========================================================================
    
    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found'
            }), 404
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Internal server error'
            }), 500
        return render_template('500.html'), 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 100 MB'
        }), 413
    
    # ==========================================================================
    # CONTEXT PROCESSORS
    # ==========================================================================
    
    @app.context_processor
    def inject_globals():
        """Inject global variables into all templates"""
        return {
            'app_name': 'DeepFake Detector',
            'app_version': '2.0.0',
            'current_year': datetime.now().year
        }
    
    logger.info("✅ App created successfully!")
    logger.info(f"   Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info("="*60)
    
    return app


def run_app():
    """Run the application"""
    app = create_app()
    
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"🌐 Starting server on http://0.0.0.0:{port}")
    logger.info(f"   Debug mode: {debug_mode}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )


# Entry point
if __name__ == '__main__':
    run_app()
