import os
import io
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

app = FastAPI(
    title="FitJourney MediaPipe Service",
    description="AI-powered body analysis using MediaPipe for fitness tracking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class BodyAnalysisRequest(BaseModel):
    image_url: str
    user_height: Optional[float] = None  # Height in cm
    user_weight: Optional[float] = None  # Weight in kg
    user_age: Optional[int] = None       # Age in years
    user_gender: Optional[str] = None    # 'male', 'female', 'other'

class BodyAnalysisResponse(BaseModel):
    success: bool
    message: str
    analysis: Optional[Dict[str, Any]] = None
    landmarks_detected: bool
    landmark_visibility: float
    waist_hip_ratio: Optional[float] = None
    shoulder_waist_ratio: Optional[float] = None
    body_shape: Optional[str] = None
    estimated_body_fat: Optional[float] = None
    posture_score: Optional[float] = None

def download_image_from_url(url: str) -> np.ndarray:
    """Download image from URL and convert to OpenCV format."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(response.content))
        
        # Convert PIL to OpenCV format (RGB)
        image_rgb = np.array(image.convert('RGB'))
        
        return image_rgb
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def extract_landmarks(image: np.ndarray) -> Optional[List]:
    """Extract pose landmarks from image using MediaPipe."""
    try:
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            return landmarks
        return None
    except Exception as e:
        logger.error(f"Error extracting landmarks: {str(e)}")
        return None

def calculate_landmark_visibility(landmarks: List[Dict]) -> float:
    """Calculate average visibility of key landmarks."""
    if not landmarks:
        return 0.0
    
    # Key landmarks for body analysis
    key_indices = [11, 12, 23, 24]  # Shoulders and hips
    key_landmarks = [landmarks[i] for i in key_indices if i < len(landmarks)]
    
    if not key_landmarks:
        return 0.0
    
    avg_visibility = sum(lm['visibility'] for lm in key_landmarks) / len(key_landmarks)
    return round(avg_visibility, 3)

def calculate_distance(point1: Dict, point2: Dict) -> float:
    """Calculate Euclidean distance between two landmarks."""
    return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

def calculate_waist_hip_ratio(landmarks: List[Dict]) -> Optional[float]:
    """Calculate waist-to-hip ratio from landmarks."""
    try:
        # Hip landmarks (left and right hip)
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Shoulder landmarks for waist approximation
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Calculate hip width
        hip_width = calculate_distance(left_hip, right_hip)
        
        # Approximate waist width (usually narrower than shoulders)
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        waist_width = shoulder_width * 0.75  # Approximation
        
        if hip_width > 0:
            ratio = waist_width / hip_width
            return round(ratio, 3)
        return None
    except (IndexError, ZeroDivisionError):
        return None

def calculate_shoulder_waist_ratio(landmarks: List[Dict]) -> Optional[float]:
    """Calculate shoulder-to-waist ratio from landmarks."""
    try:
        # Shoulder landmarks
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Hip landmarks for waist approximation
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate shoulder width
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        
        # Approximate waist width
        hip_width = calculate_distance(left_hip, right_hip)
        waist_width = hip_width * 0.85  # Approximation
        
        if waist_width > 0:
            ratio = shoulder_width / waist_width
            return round(ratio, 3)
        return None
    except (IndexError, ZeroDivisionError):
        return None

def classify_body_shape(waist_hip_ratio: Optional[float], shoulder_waist_ratio: Optional[float]) -> Optional[str]:
    """Classify body shape based on ratios."""
    if not waist_hip_ratio or not shoulder_waist_ratio:
        return None
    
    try:
        # Simplified body shape classification
        if shoulder_waist_ratio > 1.1:
            if waist_hip_ratio < 0.85:
                return "inverted_triangle"
            else:
                return "rectangle"
        elif waist_hip_ratio < 0.75:
            return "hourglass"
        elif waist_hip_ratio > 0.95:
            return "pear"
        else:
            return "rectangle"
    except:
        return None

def estimate_body_fat_percentage(landmarks: List[Dict], height: Optional[float] = None, weight: Optional[float] = None, age: Optional[int] = None, gender: Optional[str] = None) -> Optional[float]:
    """
    Estimate body fat percentage using scientifically-based methods.
    
    Uses multiple approaches:
    1. Waist-to-Hip Ratio method (most accurate from visual data)
    2. BMI correlation with age/gender adjustments
    3. Body shape classification insights
    
    References:
    - WHO guidelines for waist-hip ratio and health risk
    - Jackson & Pollock body fat equations
    - Research on visual body composition assessment
    """
    try:
        # Calculate body ratios
        waist_hip_ratio = calculate_waist_hip_ratio(landmarks)
        shoulder_waist_ratio = calculate_shoulder_waist_ratio(landmarks)
        
        if not waist_hip_ratio or not height or not weight:
            # Fallback to basic estimation if insufficient data
            return estimate_basic_body_fat(landmarks, height, weight, age, gender)
        
        # Calculate BMI for reference
        bmi = weight / ((height / 100) ** 2)
        
        # Method 1: Waist-Hip Ratio Method (Primary)
        bf_from_whr = estimate_bf_from_waist_hip_ratio(waist_hip_ratio, gender, age)
        
        # Method 2: BMI-based estimation with gender/age adjustments
        bf_from_bmi = estimate_bf_from_bmi(bmi, age, gender)
        
        # Method 3: Visual assessment based on body shape
        bf_from_shape = estimate_bf_from_body_shape(waist_hip_ratio, shoulder_waist_ratio, gender)
        
        # Weighted average of methods (WHR gets highest weight as it's most visual)
        weights = []
        estimates = []
        
        if bf_from_whr is not None:
            estimates.append(bf_from_whr)
            weights.append(0.5)  # 50% weight - most reliable from visual data
        
        if bf_from_bmi is not None:
            estimates.append(bf_from_bmi)
            weights.append(0.3)  # 30% weight - good baseline
            
        if bf_from_shape is not None:
            estimates.append(bf_from_shape)
            weights.append(0.2)  # 20% weight - supplementary
        
        if not estimates:
            return None
            
        # Calculate weighted average
        weighted_sum = sum(est * weight for est, weight in zip(estimates, weights))
        total_weight = sum(weights)
        final_estimate = weighted_sum / total_weight
        
        # Apply confidence adjustment based on measurement quality
        confidence = calculate_measurement_confidence(landmarks, waist_hip_ratio, shoulder_waist_ratio)
        
        # Adjust estimate towards population average if confidence is low
        if confidence < 0.7:
            population_avg = 20.0 if gender == 'male' else 28.0
            final_estimate = final_estimate * confidence + population_avg * (1 - confidence)
        
        # Ensure reasonable physiological range
        final_estimate = max(3.0, min(final_estimate, 50.0))
        
        return round(final_estimate, 1)
        
    except Exception as e:
        logger.error(f"Error in body fat estimation: {e}")
        return estimate_basic_body_fat(landmarks, height, weight, age, gender)

def estimate_bf_from_waist_hip_ratio(whr: float, gender: Optional[str], age: Optional[int]) -> Optional[float]:
    """
    Estimate body fat from waist-hip ratio using established correlations.
    Based on research showing WHR strongly correlates with body fat distribution.
    """
    try:
        if gender == 'male':
            # Men: Lower WHR generally indicates lower body fat
            if whr < 0.85:
                base_bf = 8 + (whr - 0.7) * 20  # 8-11% range for athletic men
            elif whr < 0.90:
                base_bf = 12 + (whr - 0.85) * 40  # 12-14% normal men
            elif whr < 0.95:
                base_bf = 15 + (whr - 0.90) * 30  # 15-16.5% slightly higher
            else:
                base_bf = 18 + (whr - 0.95) * 60  # 18%+ for higher WHR
                
        else:  # female or other
            # Women: Different WHR ranges due to natural body fat distribution
            if whr < 0.75:
                base_bf = 16 + (whr - 0.65) * 20  # 16-18% athletic women
            elif whr < 0.80:
                base_bf = 20 + (whr - 0.75) * 30  # 20-21.5% normal women  
            elif whr < 0.85:
                base_bf = 23 + (whr - 0.80) * 40  # 23-25% slightly higher
            else:
                base_bf = 26 + (whr - 0.85) * 50  # 26%+ for higher WHR
        
        # Age adjustment (body fat tends to increase with age)
        if age:
            age_factor = max(0, (age - 25) * 0.1)  # +0.1% per year after 25
            base_bf += age_factor
            
        return base_bf
    except:
        return None

def estimate_bf_from_bmi(bmi: float, age: Optional[int], gender: Optional[str]) -> Optional[float]:
    """
    Estimate body fat from BMI using Deurenberg et al. equation.
    BF% = (1.2 × BMI) + (0.23 × Age) - (10.8 × Sex) - 5.4
    where Sex = 1 for males, 0 for females
    """
    try:
        sex_factor = 1 if gender == 'male' else 0
        age_years = age or 30  # Default to 30 if age not provided
        
        # Deurenberg equation
        bf_percent = (1.2 * bmi) + (0.23 * age_years) - (10.8 * sex_factor) - 5.4
        
        # Ensure reasonable range
        return max(5.0, min(bf_percent, 45.0))
    except:
        return None

def estimate_bf_from_body_shape(whr: float, swr: Optional[float], gender: Optional[str]) -> Optional[float]:
    """
    Estimate body fat based on overall body shape characteristics.
    """
    try:
        if gender == 'male':
            # Men with lower WHR and higher shoulder-waist ratio tend to be leaner
            if swr and swr > 1.3:  # Broad shoulders relative to waist
                shape_bf = 12 + (whr - 0.8) * 25
            else:
                shape_bf = 15 + (whr - 0.8) * 30
        else:
            # Women's body fat distribution is different
            if whr < 0.8:  # Hourglass/athletic shape
                shape_bf = 18 + (whr - 0.7) * 20
            else:
                shape_bf = 22 + (whr - 0.8) * 35
                
        return max(8.0, min(shape_bf, 40.0))
    except:
        return None

def calculate_measurement_confidence(landmarks: List[Dict], whr: Optional[float], swr: Optional[float]) -> float:
    """
    Calculate confidence in our measurements based on landmark quality and ratio consistency.
    """
    try:
        confidence = 1.0
        
        # Factor 1: Landmark visibility
        visibility = calculate_landmark_visibility(landmarks)
        confidence *= visibility
        
        # Factor 2: Ratio reasonableness
        if whr:
            if 0.6 <= whr <= 1.2:  # Reasonable human range
                confidence *= 1.0
            else:
                confidence *= 0.7  # Penalize unrealistic ratios
                
        if swr:
            if 0.8 <= swr <= 1.8:  # Reasonable human range
                confidence *= 1.0
            else:
                confidence *= 0.7
                
        return min(confidence, 1.0)
    except:
        return 0.5

def estimate_basic_body_fat(landmarks: List[Dict], height: Optional[float], weight: Optional[float], age: Optional[int], gender: Optional[str]) -> Optional[float]:
    """
    Fallback estimation when insufficient data for advanced methods.
    """
    try:
        # Use population averages with basic adjustments
        if gender == 'male':
            base_estimate = 15.0  # Average male
        else:
            base_estimate = 23.0  # Average female
            
        # BMI adjustment if available
        if height and weight:
            bmi = weight / ((height / 100) ** 2)
            if bmi < 18.5:
                base_estimate -= 5
            elif bmi > 25:
                base_estimate += (bmi - 25) * 1.2
                
        # Age adjustment
        if age and age > 30:
            base_estimate += (age - 30) * 0.15
            
        return round(max(5.0, min(base_estimate, 45.0)), 1)
    except:
        return None

def calculate_posture_score(landmarks: List[Dict]) -> Optional[float]:
    """Calculate posture score based on landmark alignment."""
    try:
        # Get key landmarks for posture analysis
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        nose = landmarks[0]
        
        # Calculate shoulder alignment
        shoulder_diff = abs(left_shoulder['y'] - right_shoulder['y'])
        
        # Calculate hip alignment
        hip_diff = abs(left_hip['y'] - right_hip['y'])
        
        # Calculate head position relative to shoulders
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        head_alignment = abs(nose['x'] - shoulder_center_x)
        
        # Score calculation (lower differences = better posture)
        posture_score = 1.0 - (shoulder_diff + hip_diff + head_alignment * 0.5)
        posture_score = max(0.0, min(posture_score, 1.0))
        
        return round(posture_score, 3)
    except (IndexError, ZeroDivisionError):
        return None

@app.get("/")
@app.head("/")  # Allow HEAD requests for health checks
async def root():
    """Health check endpoint."""
    return {
        "service": "FitJourney MediaPipe Service",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
@app.head("/health")  # Allow HEAD requests for health checks
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "mediapipe_version": mp.__version__,
        "opencv_version": cv2.__version__
    }

@app.post("/analyze-body", response_model=BodyAnalysisResponse)
async def analyze_body(request: BodyAnalysisRequest):
    """Analyze body composition from image URL."""
    try:
        logger.info(f"Starting body analysis for image: {request.image_url}")
        
        # Download and process image
        image = download_image_from_url(request.image_url)
        
        # Extract landmarks
        landmarks = extract_landmarks(image)
        
        if not landmarks:
            return BodyAnalysisResponse(
                success=False,
                message="No pose landmarks detected. Please ensure the full body is visible in the image.",
                landmarks_detected=False,
                landmark_visibility=0.0
            )
        
        # Calculate visibility
        visibility = calculate_landmark_visibility(landmarks)
        
        if visibility < 0.3:
            return BodyAnalysisResponse(
                success=False,
                message="Poor landmark visibility. Please use a clearer image with good lighting.",
                landmarks_detected=True,
                landmark_visibility=visibility
            )
        
        # Perform body analysis
        waist_hip_ratio = calculate_waist_hip_ratio(landmarks)
        shoulder_waist_ratio = calculate_shoulder_waist_ratio(landmarks)
        body_shape = classify_body_shape(waist_hip_ratio, shoulder_waist_ratio)
        estimated_body_fat = estimate_body_fat_percentage(
            landmarks, 
            request.user_height, 
            request.user_weight, 
            request.user_age, 
            request.user_gender
        )
        posture_score = calculate_posture_score(landmarks)
        
        analysis_data = {
            "waist_hip_ratio": waist_hip_ratio,
            "shoulder_waist_ratio": shoulder_waist_ratio,
            "body_shape": body_shape,
            "estimated_body_fat": estimated_body_fat,
            "posture_score": posture_score,
            "landmark_count": len(landmarks),
            "analysis_confidence": visibility
        }
        
        logger.info(f"Analysis completed successfully: {analysis_data}")
        
        return BodyAnalysisResponse(
            success=True,
            message="Body analysis completed successfully",
            analysis=analysis_data,
            landmarks_detected=True,
            landmark_visibility=visibility,
            waist_hip_ratio=waist_hip_ratio,
            shoulder_waist_ratio=shoulder_waist_ratio,
            body_shape=body_shape,
            estimated_body_fat=estimated_body_fat,
            posture_score=posture_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        return BodyAnalysisResponse(
            success=False,
            message=f"Analysis failed: {str(e)}",
            landmarks_detected=False,
            landmark_visibility=0.0
        )

@app.post("/analyze-body-upload")
async def analyze_body_upload(file: UploadFile = File(...)):
    """Analyze body composition from uploaded image file."""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Convert to PIL Image and then to OpenCV format
        image = Image.open(io.BytesIO(image_data))
        image_rgb = np.array(image.convert('RGB'))
        
        # Extract landmarks
        landmarks = extract_landmarks(image_rgb)
        
        if not landmarks:
            return BodyAnalysisResponse(
                success=False,
                message="No pose landmarks detected. Please ensure the full body is visible in the image.",
                landmarks_detected=False,
                landmark_visibility=0.0
            )
        
        # Calculate visibility
        visibility = calculate_landmark_visibility(landmarks)
        
        # Perform analysis (similar to URL-based analysis)
        waist_hip_ratio = calculate_waist_hip_ratio(landmarks)
        shoulder_waist_ratio = calculate_shoulder_waist_ratio(landmarks)
        body_shape = classify_body_shape(waist_hip_ratio, shoulder_waist_ratio)
        # Note: File upload doesn't have user metadata, so using basic estimation
        estimated_body_fat = estimate_body_fat_percentage(landmarks, None, None, None, None)
        posture_score = calculate_posture_score(landmarks)
        
        analysis_data = {
            "waist_hip_ratio": waist_hip_ratio,
            "shoulder_waist_ratio": shoulder_waist_ratio,
            "body_shape": body_shape,
            "estimated_body_fat": estimated_body_fat,
            "posture_score": posture_score,
            "landmark_count": len(landmarks),
            "analysis_confidence": visibility
        }
        
        return BodyAnalysisResponse(
            success=True,
            message="Body analysis completed successfully",
            analysis=analysis_data,
            landmarks_detected=True,
            landmark_visibility=visibility,
            waist_hip_ratio=waist_hip_ratio,
            shoulder_waist_ratio=shoulder_waist_ratio,
            body_shape=body_shape,
            estimated_body_fat=estimated_body_fat,
            posture_score=posture_score
        )
        
    except Exception as e:
        logger.error(f"Error in upload analysis: {str(e)}")
        return BodyAnalysisResponse(
            success=False,
            message=f"Analysis failed: {str(e)}",
            landmarks_detected=False,
            landmark_visibility=0.0
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 