"""
Integration Tests for DeepFake Detection Web App V2.0
Tests the full application flow including API routes and database.
"""

import unittest
import os
import json
import tempfile
from io import BytesIO

# Add project root to path for imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app.main_app import create_app, init_db
from src.app.models.database import db, PredictionHistory

class TestWebAppIntegration(unittest.TestCase):
    """
    Integration tests for the Flask Web App.
    Uses a temporary database and test client.
    """
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary file for the test database
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Configure app for testing
        self.app = create_app({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': f'sqlite:///{self.db_path}',
            'WTF_CSRF_ENABLED': False,  # Disable CSRF for easier API testing
            'SECRET_KEY': 'test-secret-key'
        })
        
        # Create test client
        self.client = self.app.test_client()
        
        # Create application context
        self.ctx = self.app.app_context()
        self.ctx.push()
        
        # Create database tables
        db.create_all()

    def tearDown(self):
        """Clean up after each test"""
        db.session.remove()
        db.drop_all()
        self.ctx.pop()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['version'], '2.0.0')

    def test_dashboard_route(self):
        """Test main dashboard route returns 200"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'DeepFake Detector', response.data)

    def test_history_api_empty(self):
        """Test getting history when empty"""
        response = self.client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['total'], 0)
        self.assertEqual(len(data['items']), 0)

    def test_create_and_get_history(self):
        """Test creating a record directly in DB and retrieving it via API"""
        # Create dummy record
        record = PredictionHistory(
            file_name="test_video.mp4",
            file_type="video",
            model_used="standard",
            verdict="FAKE",
            confidence=0.95,
            fake_probability=0.95,
            real_probability=0.05
        )
        db.session.add(record)
        db.session.commit()
        
        # Get history
        response = self.client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['items'][0]['file_name'], "test_video.mp4")
        self.assertEqual(data['items'][0]['verdict'], "FAKE")

    def test_predict_endpoint_no_file(self):
        """Test predict endpoint without file"""
        response = self.client.post('/api/predict')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('No file', data['error'])

    # Note: We skip testing actual prediction (uploading file) here 
    # because it requires loading the heavy ML model and creating fake video files.
    # That belongs in a separate E2E test suite or System test.

if __name__ == '__main__':
    unittest.main()