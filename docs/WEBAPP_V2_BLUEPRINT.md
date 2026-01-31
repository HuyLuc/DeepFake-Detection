# 📐 BLUEPRINT: DeepFake Detection Web App V2.0

**Ngày tạo:** 2026-01-31  
**Mục tiêu:** Nâng cấp web app với 2 models, UI đẹp, đầy đủ tính năng

---

## 🎯 **MỤC TIÊU Dự ÁN**

### **Chức năng chính:**
1. ✅ **Upload ảnh** → Detect trực tiếp bằng 1 trong 2 models
2. ✅ **Upload video** → Extract frames → Detect giống như training pipeline
3. ✅ **Dashboard UI** hiện đại, professional
4. ✅ **Model selection** (Standard / Advanced / Ensemble)
5. ✅ **Visualization**: Confidence charts, timeline analysis
6. ✅ **Export results**: PDF, JSON
7. ✅ **History tracking**: Lưu lịch sử các lần detect

---

## 🏗️ **KIẾN TRÚC HỆ THỐNG**

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT (Browser)                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Modern Dashboard UI (HTML/CSS/JS)                 │ │
│  │  - Upload Widget (Drag & Drop)                     │ │
│  │  - Model Selector                                  │ │
│  │  - Results Visualization (Charts.js)               │ │
│  │  - History Panel                                   │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                            ↕ HTTP/JSON
┌─────────────────────────────────────────────────────────┐
│                  BACKEND (Flask Server)                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │  API Layer (main_app.py)                           │ │
│  │  - /api/upload                                     │ │
│  │  - /api/predict/image                              │ │
│  │  - /api/predict/video                              │ │
│  │  - /api/history                                    │ │
│  │  - /api/export/{format}                            │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Business Logic Layer                              │ │
│  │  ┌──────────────┐  ┌──────────────┐               │ │
│  │  │ModelManager  │  │FileProcessor │               │ │
│  │  │- Standard    │  │- Image       │               │ │
│  │  │- Advanced    │  │- Video       │               │ │
│  │  │- Ensemble    │  │- Face Extract│               │ │
│  │  └──────────────┘  └──────────────┘               │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Data Layer                                        │ │
│  │  - SQLite DB (History)                             │ │
│  │  - File Storage (Temp uploads)                     │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────┐
│                    ML MODELS LAYER                       │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │  Standard Model     │  │  Advanced Model         │  │
│  │  (EfficientNet-B4)  │  │  (EfficientNet + LSTM)  │  │
│  │  - Single frame     │  │  - Sequence of 5 frames │  │
│  │  - Fast (~0.5s)     │  │  - Accurate (~2s)       │  │
│  │  - 98% accuracy     │  │  - 95% accuracy         │  │
│  └─────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 **CẤU TRÚC THƯ MỤC MỚI**

```
DeepFake-Detection/
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main_app.py                    # Flask app entry point
│   │   ├── api/                            # ✨ NEW
│   │   │   ├── __init__.py
│   │   │   ├── routes.py                   # API endpoints
│   │   │   └── validators.py               # Input validation
│   │   ├── services/                       # ✨ NEW
│   │   │   ├── __init__.py
│   │   │   ├── model_manager.py            # Quản lý 2 models
│   │   │   ├── file_processor.py           # Xử lý ảnh/video
│   │   │   ├── prediction_service.py       # Prediction logic
│   │   │   └── export_service.py           # Export PDF/JSON
│   │   ├── models/                         # ✨ NEW
│   │   │   ├── __init__.py
│   │   │   ├── database.py                 # SQLite setup
│   │   │   └── history.py                  # History model
│   │   ├── static/                         # ✨ NEW
│   │   │   ├── css/
│   │   │   │   └── dashboard.css           # Custom styles
│   │   │   ├── js/
│   │   │   │   ├── main.js                 # Main app logic
│   │   │   │   ├── upload.js               # Upload handler
│   │   │   │   ├── charts.js               # Charts.js wrapper
│   │   │   │   └── history.js              # History panel
│   │   │   └── lib/                        # External libs (optional)
│   │   └── templates/
│   │       ├── index.html                  # 🔄 REDESIGN
│   │       ├── dashboard.html              # ✨ NEW
│   │       └── components/                 # ✨ NEW
│   │           ├── upload_widget.html
│   │           ├── model_selector.html
│   │           ├── results_panel.html
│   │           └── history_panel.html
│   ├── architectures/
│   │   ├── standard/
│   │   │   └── model.py                    # Standard model class
│   │   └── advanced/
│   │       └── model.py                    # ✨ NEW - Temporal model class
│   └── utils/
│       └── video_processor.py              # 🔄 UPDATE - Extract frames như training
├── saved_models/
│   ├── standard/
│   │   └── best_model.pth                  # ✅ ĐÃ CÓ (70MB)
│   └── advanced/
│       └── best_temporal_model.pth         # ✅ ĐÃ CÓ (135MB)
├── data/
│   ├── temp_uploads/                       # ✨ NEW - Temp storage
│   └── history.db                          # ✨ NEW - SQLite DB
├── exports/                                # ✨ NEW
│   ├── pdf/
│   └── json/
└── requirements.txt                        # 🔄 UPDATE - Add new packages
```

---

## 🎨 **UI/UX DESIGN - Dashboard Layout**

### **Wireframe:**

```
╔═══════════════════════════════════════════════════════════════╗
║  🎬 DEEPFAKE DETECTION SYSTEM v2.0                   [Profile] ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  📤 UPLOAD MEDIA                                          │ ║
║  │  ┌────────────────────────────────────────────────────┐  │ ║
║  │  │                                                      │  │ ║
║  │  │        Drag & Drop Image/Video Here                 │  │ ║
║  │  │              or Click to Browse                      │  │ ║
║  │  │                                                      │  │ ║
║  │  │  Supported: JPG, PNG, MP4, AVI (Max: 100MB)         │  │ ║
║  │  └────────────────────────────────────────────────────┘  │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                ║
║  ┌────────────────────┐  ┌────────────────────────────────┐  ║
║  │ 🤖 MODEL SELECTION │  │  ⚙️ DETECTION OPTIONS          │  ║
║  ├────────────────────┤  ├────────────────────────────────┤  ║
║  │ ○ Standard         │  │  ☑ Show frame-by-frame         │  ║
║  │   Fast (0.5s)      │  │  ☑ Export results              │  ║
║  │   98% accuracy     │  │  ☑ Save to history             │  ║
║  │                    │  │  Threshold: [██████░░] 0.85    │  ║
║  │ ○ Advanced         │  └────────────────────────────────┘  ║
║  │   Accurate (2s)    │                                      ║
║  │   95% accuracy     │  ┌──────────────────────────────┐   ║
║  │                    │  │  [🚀 START DETECTION]        │   ║
║  │ ○ Ensemble         │  └──────────────────────────────┘   ║
║  │   Best (3s)        │                                      ║
║  │   Compare both     │                                      ║
║  └────────────────────┘                                      ║
║                                                                ║
╠═══════════════════════════════════════════════════════════════╣
║  📊 RESULTS                                                    ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  Final Verdict: [FAKE] 🔴                                 │ ║
║  │  Confidence: 92.5%                                        │ ║
║  │  ┌────────────────────────────────────────────────────┐  │ ║
║  │  │  Confidence Timeline  (Chart.js)                   │  │ ║
║  │  │    %                                                │  │ ║
║  │  │  100┤            ▄▄                                 │  │ ║
║  │  │   90┤        ▄▄▄▀  ▀▄                               │  ║
║  │  │   80┤     ▄▄▀        ▀▄▄                            │  │ ║
║  │  │      └──────────────────────► Frame                 │  │ ║
║  │  └────────────────────────────────────────────────────┘  │ ║
║  │                                                           │ ║
║  │  📋 Details:                                              │ ║
║  │  • Total Frames Analyzed: 127                            │ ║
║  │  • Frames with Face: 98                                  │ ║
║  │  • FAKE Evidence: 85 frames (86.7%)                      │ ║
║  │  • REAL Evidence: 13 frames (13.3%)                      │ ║
║  │  • Processing Time: 2.3s                                 │ ║
║  │                                                           │ ║
║  │  [📥 Download PDF]  [📥 Download JSON]                    │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                ║
╠═══════════════════════════════════════════════════════════════╣
║  📜 HISTORY (Last 10 detections)                               ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Time        │ File         │ Model    │ Result │ Conf.   │ ║
║  ├──────────────────────────────────────────────────────────┤ ║
║  │ 14:05:23    │ video1.mp4   │ Standard │ FAKE   │ 92.5%  │ ║
║  │ 13:42:11    │ image2.jpg   │ Advanced │ REAL   │ 87.3%  │ ║
║  │ ...                                                       │ ║
║  └──────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 🔌 **API ENDPOINTS DESIGN**

### **1. Upload Endpoint**
```python
POST /api/upload
Content-Type: multipart/form-data

Request:
{
    "file": <binary>,
    "model": "standard" | "advanced" | "ensemble",
    "options": {
        "show_timeline": true,
        "export_format": "pdf",
        "save_history": true,
        "threshold": 0.85
    }
}

Response:
{
    "success": true,
    "upload_id": "uuid-1234",
    "file_type": "video" | "image",
    "message": "File uploaded successfully"
}
```

### **2. Predict Image Endpoint**
```python
POST /api/predict/image
Content-Type: application/json

Request:
{
    "upload_id": "uuid-1234",
    "model": "standard" | "advanced"
}

Response:
{
    "success": true,
    "result": {
        "verdict": "FAKE" | "REAL",
        "confidence": 0.925,
        "model_used": "standard",
        "processing_time": 0.52,
        "details": {
            "face_detected": true,
            "face_confidence": 0.98
        }
    }
}
```

### **3. Predict Video Endpoint**
```python
POST /api/predict/video
Content-Type: application/json

Request:
{
    "upload_id": "uuid-1234",
    "model": "standard" | "advanced" | "ensemble"
}

Response:
{
    "success": true,
    "result": {
        "verdict": "FAKE",
        "confidence": 0.925,
        "model_used": "ensemble",
        "processing_time": 3.2,
        "timeline": [
            {"frame": 1, "confidence": 0.92, "verdict": "FAKE"},
            {"frame": 5, "confidence": 0.88, "verdict": "FAKE"},
            ...
        ],
        "stats": {
            "total_frames": 127,
            "frames_analyzed": 98,
            "fake_count": 85,
            "real_count": 13,
            "fake_ratio": 0.867
        },
        "models_comparison": {  // Chỉ khi ensemble
            "standard": {"verdict": "FAKE", "confidence": 0.93},
            "advanced": {"verdict": "FAKE", "confidence": 0.92}
        }
    }
}
```

### **4. History Endpoints**
```python
GET /api/history?limit=10&model=all

Response:
{
    "success": true,
    "count": 10,
    "history": [
        {
            "id": 1,
            "timestamp": "2026-01-31T14:05:23",
            "filename": "video1.mp4",
            "file_type": "video",
            "model": "standard",
            "verdict": "FAKE",
            "confidence": 0.925
        },
        ...
    ]
}
```

### **5. Export Endpoints**
```python
GET /api/export/pdf/{result_id}
GET /api/export/json/{result_id}

Response: File download
```

---

## 🗄️ **DATABASE SCHEMA**

```sql
-- History table
CREATE TABLE detection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    filename TEXT NOT NULL,
    file_type TEXT CHECK(file_type IN ('image', 'video')),
    file_size INTEGER,  -- bytes
    model_used TEXT CHECK(model_used IN ('standard', 'advanced', 'ensemble')),
    verdict TEXT CHECK(verdict IN ('FAKE', 'REAL')),
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    processing_time REAL,  -- seconds
    frames_analyzed INTEGER,  -- NULL for images
    fake_frames INTEGER,      -- NULL for images
    real_frames INTEGER,      -- NULL for images
    result_json TEXT,  -- Full result as JSON
    export_pdf_path TEXT,
    export_json_path TEXT
);

-- Index for faster queries
CREATE INDEX idx_timestamp ON detection_history(timestamp DESC);
CREATE INDEX idx_model ON detection_history(model_used);
```

---

## 📦 **DEPENDENCIES MỚI**

Cập nhật `requirements.txt`:

```txt
# Existing
flask>=2.3.0
flask-wtf>=1.1.1
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
mediapipe>=0.10.0
pillow>=10.0.0

# ✨ NEW - For Dashboard
flask-cors>=4.0.0           # CORS support
flask-sqlalchemy>=3.0.0     # Database ORM (optional)

# ✨ NEW - For Export
reportlab>=4.0.0            # PDF generation
jinja2>=3.1.2              # Template engine (already in Flask)

# ✨ NEW - For Utilities
python-dotenv>=1.0.0        # Environment variables
```

---

## 🧩 **CORE COMPONENTS DESIGN**

### **1. ModelManager Class**

```python
# src/app/services/model_manager.py

class ModelManager:
    """
    Quản lý cả 2 models: Standard và Advanced
    Singleton pattern để load model 1 lần duy nhất
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.standard_model = None
        self.advanced_model = None
        self.load_models()
        self._initialized = True
    
    def load_models(self):
        """Load cả 2 models vào memory"""
        # Load Standard Model (EfficientNet-B4)
        self.standard_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
        self.standard_model.load_state_dict(torch.load('saved_models/standard/best_model.pth', map_location=self.device))
        self.standard_model.to(self.device)
        self.standard_model.eval()
        
        # Load Advanced Model (EfficientNet + LSTM)
        self.advanced_model = TemporalModel(num_classes=2, pretrained=False)
        self.advanced_model.load_state_dict(torch.load('saved_models/advanced/best_temporal_model.pth', map_location=self.device))
        self.advanced_model.to(self.device)
        self.advanced_model.eval()
        
    def predict_image_standard(self, image: PIL.Image) -> dict:
        """Predict với Standard model (1 ảnh)"""
        pass
    
    def predict_image_advanced(self, image: PIL.Image) -> dict:
        """Predict với Advanced model (1 ảnh - treated as single frame in sequence)"""
        pass
    
    def predict_video_standard(self, frames: List[PIL.Image]) -> dict:
        """Predict video với Standard model (frame-by-frame)"""
        pass
    
    def predict_video_advanced(self, frames: List[PIL.Image]) -> dict:
        """Predict video với Advanced model (sequences of 5 frames)"""
        pass
    
    def predict_ensemble(self, input_data, is_video=False) -> dict:
        """Combine predictions từ cả 2 models"""
        pass
```

### **2. FileProcessor Class**

```python
# src/app/services/file_processor.py

class FileProcessor:
    """
    Xử lý ảnh và video
    Extract faces và frames giống như training pipeline
    """
    
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
    
    def process_image(self, image_path: str) -> PIL.Image:
        """
        1. Load image
        2. Detect face
        3. Crop face
        4. Return PIL Image
        """
        pass
    
    def process_video(self, video_path: str, sequence_length: int = 5) -> dict:
        """
        1. Extract frames (skip_frames from config)
        2. Detect faces in each frame
        3. Crop faces
        4. Group into sequences (for Advanced model)
        5. Return dict with frames and metadata
        
        Returns:
        {
            'all_frames': [PIL.Image, ...],  # All frames with faces
            'sequences': [[frame1, frame2, frame3, frame4, frame5], ...],  # For Advanced
            'metadata': {
                'total_frames': 127,
                'frames_with_face': 98,
                'fps': 30
            }
        }
        """
        pass
    
    def extract_face(self, image: PIL.Image) -> Optional[PIL.Image]:
        """Extract face from image như training"""
        pass
```

### **3. PredictionService Class**

```python
# src/app/services/prediction_service.py

class PredictionService:
    """
    Orchestrate prediction workflow
    Kết hợp ModelManager + FileProcessor
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.file_processor = FileProcessor()
    
    def predict(self, file_path: str, file_type: str, model_choice: str, options: dict) -> dict:
        """
        Main prediction logic
        
        Args:
            file_path: Path to uploaded file
            file_type: 'image' or 'video'
            model_choice: 'standard', 'advanced', or 'ensemble'
            options: Dict with detection options
        
        Returns:
            Complete result dict
        """
        if file_type == 'image':
            return self._predict_image(file_path, model_choice, options)
        else:
            return self._predict_video(file_path, model_choice, options)
    
    def _predict_image(self, image_path: str, model_choice: str, options: dict) -> dict:
        """Image prediction workflow"""
        # 1. Process image (extract face)
        # 2. Call appropriate model
        # 3. Format result
        pass
    
    def _predict_video(self, video_path: str, model_choice: str, options: dict) -> dict:
        """Video prediction workflow"""
        # 1. Process video (extract frames + faces)
        # 2. Call appropriate model(s)
        # 3. Aggregate results
        # 4. Build timeline
        # 5. Format result
        pass
```

### **4. ExportService Class**

```python
# src/app/services/export_service.py

class ExportService:
    """Generate PDF and JSON reports"""
    
    def export_pdf(self, result: dict, output_path: str):
        """
        Generate PDF report with:
        - Summary verdict
        - Confidence chart
        - Frame-by-frame analysis (if video)
        - Timestamp, model used
        """
        pass
    
    def export_json(self, result: dict, output_path: str):
        """Export full result as JSON"""
        pass
```

---

## 🎬 **IMPLEMENTATION PLAN - STEP BY STEP**

### **Phase 1: Backend Foundation** (3-4h)

#### **Step 1.1: Setup Project Structure** (30min)
- [ ] Tạo các folder mới: `api/`, `services/`, `models/`, `static/`, `exports/`
- [ ] Tạo `__init__.py` files
- [ ] Update `.gitignore` (temp_uploads, exports, history.db)

#### **Step 1.2: Implement TemporalModel Class** (1h)
- [ ] Tạo `src/architectures/advanced/model.py`
- [ ] Copy TemporalModelV2 từ training notebook
- [ ] Test load model weights

#### **Step 1.3: Implement ModelManager** (1h)
- [ ] Create `model_manager.py`
- [ ] Load cả 2 models
- [ ] Implement predict methods cho từng model
- [ ] Test với sample image/video

#### **Step 1.4: Implement FileProcessor** (1h)
- [ ] Create ` processor.py`
- [ ] Implement image processing (extract face)
- [ ] Implement video processing (extract frames như training)
- [ ] Test với real files

#### **Step 1.5: Implement PredictionService** (45min)
- [ ] Create `prediction_service.py`
- [ ] Orchestrate ModelManager + FileProcessor
- [ ] Implement logic cho từng mode (standard/advanced/ensemble)

---

### **Phase 2: API Layer** (2h)

#### **Step 2.1: Database Setup** (30min)
- [ ] Create `models/database.py` - SQLite init
- [ ] Create `models/history.py` - History model
- [ ] Create tables
- [ ] Test CRUD operations

#### **Step 2.2: API Routes** (1h)
- [ ] Create `api/routes.py`
- [ ] Implement `/api/upload`
- [ ] Implement `/api/predict/image`
- [ ] Implement `/api/predict/video`
- [ ] Implement `/api/history`

#### **Step 2.3: Validators & Error Handling** (30min)
- [ ] Create `api/validators.py`
- [ ] Validate file types, sizes
- [ ] Uniform error responses

---

### **Phase 3: Export Service** (1h)

#### **Step 3.1: PDF Export** (45min)
- [ ] Create `services/export_service.py`
- [ ] Implement PDF generation với ReportLab
- [ ] Include chart images (optional)

#### **Step 3.2: JSON Export** (15min)
- [ ] Implement JSON export
- [ ] Test endpoints

---

### **Phase 4: Frontend - Dashboard UI** (4-5h)

#### **Step 4.1: HTML Structure** (1h)
- [ ] Create `dashboard.html` template
- [ ] Layout sections: Upload, Model Select, Results, History
- [ ] Component templates

#### **Step 4.2: CSS Styling** (1.5h)
- [ ] Create `static/css/dashboard.css`
- [ ] Modern gradient backgrounds
- [ ] Card designs
- [ ] Responsive layout
- [ ] Dark mode toggle (optional)

#### **Step 4.3: JavaScript Logic** (2h)
- [ ] `main.js` - App initialization
- [ ] `upload.js` - Drag & drop upload
- [ ] `charts.js` - Chart.js integration
- [ ] `history.js` - History panel operations

#### **Step 4.4: Charts Integration** (30min)
- [ ] Add Chart.js CDN
- [ ] Implement confidence timeline chart
- [ ] Test với real data

---

### **Phase 5: Integration & Testing** (2h)

#### **Step 5.1: End-to-End Testing** (1h)
- [ ] Test image upload → predict → results
- [ ] Test video upload → predict → timeline → results
- [ ] Test all 3 model modes
- [ ] Test export PDF/JSON

#### **Step 5.2: History Feature** (30min)
- [ ] Test save to history
- [ ] Test load history
- [ ] Test history panel UI

#### **Step 5.3: Error Handling & Edge Cases** (30min)
- [ ] No face detected
- [ ] Invalid file format
- [ ] Large file handling
- [ ] Model loading errors

---

### **Phase 6: Polish & Optimization** (1h)

#### **Step 6.1: Performance** (30min)
- [ ] Test với large videos
- [ ] Optimize frame extraction
- [ ] Add progress indicators

#### **Step 6.2: UX Improvements** (30min)
- [ ] Loading states
- [ ] Success/error animations
- [ ] Tooltips
- [ ] Help text

---

## 📊 **TECHNICAL SPECIFICATIONS**

### **Model Processing:**

| Model | Input | Processing | Output |
|-------|-------|------------|--------|
| **Standard** | Single image/frame | EfficientNet-B4 → Softmax | Class + Confidence |
| **Advanced** | Sequence of 5 frames | EfficientNet → LSTM → FC → Softmax | Class + Confidence |
| **Ensemble** | Same as above | Average predictions | Combined result |

### **Video Processing Pipeline:**

```
Video File
    ↓
Extract frames (skip_frames=5)
    ↓
Detect faces (MediaPipe)
    ↓
Crop & resize faces
    ↓
┌──────────────┬──────────────┐
│  Standard:   │  Advanced:   │
│  Process     │  Group into  │
│  frame-by-   │  sequences   │
│  frame       │  of 5 frames │
└──────────────┴──────────────┘
    ↓               ↓
Aggregate results
    ↓
Final verdict (FAKE if >85% frames are FAKE)
```

---

## 🎯 **SUCCESS CRITERIA**

### **Functional Requirements:**
- ✅ Upload ảnh/video thành công
- ✅ Detect với cả 3 modes (Standard/Advanced/Ensemble)
- ✅ Hiển thị results với confidence chart
- ✅ Export PDF & JSON
- ✅ Lưu và hiển thị history

### **Performance Requirements:**
- ✅ Standard model: < 1s per image, < 30s per video (60s)
- ✅ Advanced model: < 2s per image, < 60s per video (60s)
- ✅ UI responsive: < 200ms interactions

### **UX Requirements:**
- ✅ Modern, professional dashboard
- ✅ Intuitive drag-drop upload
- ✅ Clear model selection
- ✅ Informative results visualization
- ✅ Easy export

---

## 🚦 **ESTIMATED TIMELINE**

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Phase 1: Backend** | 3-4h | ModelManager, FileProcessor, PredictionService |
| **Phase 2: API** | 2h | Routes, Database, Validators |
| **Phase 3: Export** | 1h | PDF/JSON export |
| **Phase 4: Frontend** | 4-5h | Dashboard UI, Charts, JS logic |
| **Phase 5: Testing** | 2h | E2E testing, History |
| **Phase 6: Polish** | 1h | Performance, UX |
| **TOTAL** | **13-15 hours** | Full implementation |

**Chia nhỏ:**
- **Day 1** (6h): Phase 1 + Phase 2
- **Day 2** (5h): Phase 3 + Phase 4
- **Day 3** (3h): Phase 5 + Phase 6

---

## ✅ **DELIVERABLES**

1. ✅ **Fully functional web app** với đầy đủ tính năng
2. ✅ **Modern dashboard UI** với charts
3. ✅ **2 trained models** integrated
4. ✅ **PDF/JSON export** functionality
5. ✅ **History tracking** system
6. ✅ **Complete documentation** (README updates)

---

## 🔧 **NEXT STEPS**

**Sau khi bạn approve blueprint này, chúng ta sẽ:**

1. **Setup project structure** (tạo folders mới)
2. **Start Phase 1** - Implement backend core
3. **Iteratively build** từng component
4. **Test & integrate** từng phase

---

**Bạn có muốn điều chỉnh gì trong blueprint này không?** 🤔

Nếu OK, tôi sẽ bắt đầu **Phase 1: Step 1.1 - Setup Project Structure** ngay! 🚀
