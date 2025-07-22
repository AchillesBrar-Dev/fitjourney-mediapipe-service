# FitJourney MediaPipe Service

AI-powered body analysis service using MediaPipe for the FitJourney fitness application.

## 🚀 Features

- **Pose Detection**: Extract 33 body landmarks using MediaPipe Pose
- **Body Metrics**: Calculate waist-hip ratio, shoulder-waist ratio
- **Body Shape Classification**: Categorize into rectangle, pear, inverted triangle, hourglass
- **Body Fat Estimation**: Heuristic-based body fat percentage estimation
- **Posture Analysis**: Score posture quality based on landmark alignment
- **Multiple Input Methods**: Support for both URL-based and direct file upload

## 📋 Prerequisites

- Python 3.11+
- Render account (for deployment)
- Git

## 🛠️ Local Development Setup

### 1. Clone and Setup

```bash
# Navigate to the mediapipe-service directory
cd mediapipe-service

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Locally

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The service will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Test the Service

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test analysis with a sample image (replace with actual image URL)
curl -X POST "http://localhost:8000/analyze-body" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/sample-pose-image.jpg",
    "user_height": 175,
    "user_weight": 70
  }'
```

## ☁️ Deployment on Render

### Method 1: Automatic Deployment (Recommended)

1. **Fork/Upload Code**:
   - Create a new repository on GitHub
   - Upload the `mediapipe-service` folder to your repository

2. **Connect to Render**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing the MediaPipe service

3. **Configure Service**:
   - **Name**: `fitjourney-mediapipe-service`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Start with "Starter" (free tier)

4. **Deploy**:
   - Click "Create Web Service"
   - Wait for build and deployment to complete
   - Note the service URL (e.g., `https://fitjourney-mediapipe-service.onrender.com`)

### Method 2: Docker Deployment

```bash
# Build Docker image
docker build -t fitjourney-mediapipe .

# Run locally with Docker
docker run -p 8000:8000 fitjourney-mediapipe

# Deploy to Render with Docker
# (Upload the Dockerfile to your repository and configure Render to use Docker)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `PYTHON_VERSION` | Python version for Render | `3.11.5` |

### MediaPipe Configuration

The service uses these MediaPipe Pose settings:
- **Model Complexity**: 2 (highest accuracy)
- **Detection Confidence**: 0.5 minimum
- **Static Image Mode**: True (optimized for single images)

## 📡 API Endpoints

### GET `/`
Health check endpoint returning service status.

### GET `/health`
Detailed health check with version information.

### POST `/analyze-body`
Analyze body composition from image URL.

**Request Body**:
```json
{
  "image_url": "https://example.com/image.jpg",
  "user_height": 175.0,  // Optional: height in cm
  "user_weight": 70.0    // Optional: weight in kg
}
```

**Response**:
```json
{
  "success": true,
  "message": "Body analysis completed successfully",
  "landmarks_detected": true,
  "landmark_visibility": 0.85,
  "waist_hip_ratio": 0.72,
  "shoulder_waist_ratio": 1.18,
  "body_shape": "inverted_triangle",
  "estimated_body_fat": 12.5,
  "posture_score": 0.78,
  "analysis": {
    "waist_hip_ratio": 0.72,
    "shoulder_waist_ratio": 1.18,
    "body_shape": "inverted_triangle",
    "estimated_body_fat": 12.5,
    "posture_score": 0.78,
    "landmark_count": 33,
    "analysis_confidence": 0.85
  }
}
```

### POST `/analyze-body-upload`
Analyze body composition from uploaded image file.

**Request**: Multipart form with image file
**Response**: Same as `/analyze-body`

## 🔗 Integration with React Frontend

### Environment Variable Setup

In your React app's `.env.local`, add:
```env
VITE_MEDIAPIPE_SERVICE_URL=https://your-service-url.onrender.com
```

### Frontend Integration Example

```typescript
// api/mediapipe.ts
const MEDIAPIPE_URL = import.meta.env.VITE_MEDIAPIPE_SERVICE_URL

export const analyzeBodyPhoto = async (imageUrl: string, height?: number, weight?: number) => {
  try {
    const response = await fetch(`${MEDIAPIPE_URL}/analyze-body`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: imageUrl,
        user_height: height,
        user_weight: weight,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error('MediaPipe analysis failed:', error)
    throw error
  }
}
```

### Update Supabase Database

After getting analysis results, update the `body_analyses` table:

```typescript
const updateBodyAnalysis = async (analysisId: string, results: any) => {
  const { error } = await supabase
    .from('body_analyses')
    .update({
      waist_hip_ratio: results.waist_hip_ratio,
      shoulder_waist_ratio: results.shoulder_waist_ratio,
      body_shape: results.body_shape,
      estimated_body_fat: results.estimated_body_fat,
      landmark_visibility: results.landmark_visibility,
      posture_score: results.posture_score,
      analysis_status: results.success ? 'completed' : 'failed',
      analysis_metadata: results.analysis,
    })
    .eq('id', analysisId)

  if (error) {
    console.error('Failed to update analysis:', error)
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **"No pose landmarks detected"**
   - Ensure the full body is visible in the image
   - Use good lighting conditions
   - Person should be facing the camera

2. **"Poor landmark visibility"**
   - Use higher resolution images
   - Avoid blurry or dark images
   - Ensure contrast between person and background

3. **Build failures on Render**
   - Check that all dependencies are in `requirements.txt`
   - Verify Python version compatibility
   - Check build logs for specific error messages

### Memory Considerations

- MediaPipe models require significant memory
- Consider upgrading to Render's "Standard" plan for production
- Monitor memory usage and adjust instance size if needed

## 🔒 Security Considerations

1. **CORS Configuration**: Update `allow_origins` in production
2. **Rate Limiting**: Consider adding rate limiting for production use
3. **Input Validation**: The service validates image formats and URLs
4. **Error Handling**: Comprehensive error handling prevents crashes

## 📈 Performance Tips

1. **Image Size**: Optimize images before analysis (max 1920x1080)
2. **Caching**: Consider implementing Redis for result caching
3. **Async Processing**: For high volume, consider queue-based processing
4. **Monitoring**: Use Render's monitoring tools to track performance

## 🆘 Support

If you encounter issues:

1. Check the [Render logs](https://dashboard.render.com) for error details
2. Test the service locally first
3. Verify your image URLs are publicly accessible
4. Check network connectivity between frontend and MediaPipe service

## 📚 Next Steps

After successful deployment:

1. ✅ Test the service with sample images
2. ✅ Integrate with your React frontend
3. ✅ Update Supabase database schema if needed
4. ✅ Implement error handling in your frontend
5. ✅ Consider adding result caching for better performance

---

**Service Status**: Ready for deployment 🚀
**Estimated Setup Time**: 15-30 minutes
**Production Ready**: Yes (with Standard plan recommended) 