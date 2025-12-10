# Testing Guide for Video Incident Monitoring

## Fixed Issues

✅ **XML parse error**: Fixed namespace issue in analyze-standalone.html
✅ **404 favicon.ico**: Added favicon.ico file to webapp directory
✅ **Backend configuration**: Verified AI CORE configuration in .env file

## Step-by-Step Testing

### Step 1: Start Backend Service
```bash
# Start backend service:
start_backend.bat

# Or manually:
# 1. Activate virtual environment: venv\Scripts\activate
# 2. Install dependencies: pip install -r backend_requirements.txt
# 3. Run: python backend_odata_service.py
```

**Expected result**: Service starts on http://localhost:5000

### Step 2: Test Standalone Page

Choose one of the options:

**Option A - Python server:**
```bash
start_standalone.bat
```

**Option B - Node.js server:**
```bash
start_standalone_node.bat
```

**Option C - Main UI5 server:**
```bash
start_frontend.bat
```

### Step 3: Browser Verification

1. **Open**: http://localhost:8080/analyze-standalone.html
2. **Check**: Page loads without console errors
3. **Check**: Favicon loads (no 404 error)
4. **Check**: All UI elements display correctly

### Step 4: Test AI CORE Functionality

1. **Upload test video** (from Video/ folder if files exist)
2. **Enter instruction**: "Analyze this video for safety violations"
3. **Click "Analyze Media"**
4. **Check**: Request is sent to backend (status "Processing...")
5. **Check**: Response received from AI CORE with analysis results

## Common Issues and Solutions

### Issue: "Python not found"
**Solution**: Use start_standalone_node.bat or install Python

### Issue: "Backend unavailable"
**Solution**: Ensure backend is running on port 5000

### Issue: "AI CORE authentication failed"
**Solution**: Check credentials in .env file

### Issue: "CORS errors"
**Solution**: Backend is already configured with CORS middleware, verify both services are running

## Application Structure

```
📁 AI CORE Video template1/
├── 📄 backend_odata_service.py     # Python FastAPI backend
├── 📄 .env                         # AI CORE credentials
├── 📄 start_backend.bat           # Start backend
├── 📄 start_standalone.bat        # Test server (Python)
├── 📄 start_standalone_node.bat   # Test server (Node.js)
└── 📁 video-incident-monitor/
    ├── 📄 package.json            # UI5 dependencies
    └── 📁 webapp/
        ├── 📄 analyze-standalone.html  # Main page (FIXED)
        ├── 📄 favicon.ico             # Icon (ADDED)
        └── 📁 view/
            └── 📄 AnalyzeMedia.view.xml
```

## Testing Checklist

- [ ] Backend starts without errors
- [ ] Frontend server starts
- [ ] Page http://localhost:8080/analyze-standalone.html opens
- [ ] No errors in Developer Tools (F12)
- [ ] No 404 error for favicon.ico
- [ ] UI elements display correctly
- [ ] File upload works
- [ ] Analysis request submission works
- [ ] Result retrieval from AI CORE works
- [ ] Functionality preserved without degradation
