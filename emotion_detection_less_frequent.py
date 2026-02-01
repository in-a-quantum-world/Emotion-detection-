"""
Accessibility Emotion Detection Application
For Google Gemini 3 Hackathon (Feb 9, 2026)

FIXED VERSION - Compatible with MediaPipe 0.10.30+

This module provides algorithms for:
1. Facial emotion detection using Gemini 3 API
2. Body language analysis using MediaPipe
3. Accessibility output generation for sensory disabilities

Repository: https://github.com/in-a-quantum-world/Emotion-detection-
"""

import cv2
import numpy as np
import base64
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import warnings
import os
import logging

# Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# New MediaPipe imports for version 0.10.30+
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


# ============================================================================
# PART 1: DATA STRUCTURES
# ============================================================================

class Emotion(Enum):
    """7 Universal emotions based on Ekman's research + neutral"""
    HAPPY = "happiness"
    SAD = "sadness"
    ANGRY = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class PostureState(Enum):
    """Body language states detected from pose"""
    OPEN_CONFIDENT = "open_confident"
    CLOSED_DEFENSIVE = "closed_defensive"
    ENGAGED_FORWARD = "engaged_forward"
    DISENGAGED_BACK = "disengaged_back"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    """Result of emotion analysis"""
    primary_emotion: Emotion
    intensity: float  # 1-7 scale
    confidence: float  # 0-1
    secondary_emotions: List[Tuple[Emotion, float]]  # (emotion, probability)
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    action_units: Dict[str, float]  # AU name -> intensity


@dataclass
class BodyLanguageResult:
    """Result of body language analysis"""
    posture: PostureState
    lean_angle: float  # degrees, positive = forward
    shoulder_openness: float  # 0-1
    arm_position: str
    head_tilt: float  # degrees
    confidence: float


@dataclass
class AccessibilityDescription:
    """Accessibility output for different user needs"""
    for_deaf: str  # Describes auditory emotional cues
    for_blind: str  # Describes visual emotional cues
    for_autism: str  # Explicit emotion labeling with context
    overall_summary: str


# ============================================================================
# PART 2: BODY LANGUAGE ANALYSIS WITH NEW MEDIAPIPE API
# ============================================================================

class BodyLanguageAnalyzer:
    """
    Analyzes body language using MediaPipe Pose landmarks.
    Updated for MediaPipe 0.10.30+ Tasks API.
    """
    
    def __init__(self):
        # Download model if needed - using the lite model for speed
        self.model_path = self._get_model_path()
        
        # Create pose landmarker options
        base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    
    def _get_model_path(self) -> str:
        """Download pose model if not present."""
        import urllib.request
        import os
        
        model_path = "pose_landmarker_lite.task"
        
        if not os.path.exists(model_path):
            print("Downloading pose model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")
        
        return model_path
    
    def analyze(self, frame: np.ndarray) -> Optional[BodyLanguageResult]:
        """Analyze body language from a video frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose
        results = self.pose_landmarker.detect(mp_image)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
        
        landmarks = results.pose_landmarks[0]  # First detected pose
        
        # Extract key points (landmarks are normalized 0-1)
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
        left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
        right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
        left_elbow = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
        right_elbow = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
        left_wrist = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
        right_wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
        nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        
        # Calculate metrics
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        
        # Shoulder openness (normalized 0-1)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        hip_width = np.linalg.norm(right_hip - left_hip)
        openness = min(1.0, shoulder_width / max(hip_width * 1.5, 0.01))
        
        # Lean angle (z-axis indicates forward/backward lean)
        lean_angle = np.arctan2(
            shoulder_mid[2] - hip_mid[2],
            abs(shoulder_mid[1] - hip_mid[1]) + 0.001
        ) * 180 / np.pi
        
        # Head tilt (relative to shoulder line)
        head_offset = nose - shoulder_mid
        head_tilt = np.arctan2(head_offset[0], abs(head_offset[1]) + 0.001) * 180 / np.pi
        
        # Arm position analysis
        arm_position = self._analyze_arm_position(
            left_shoulder, right_shoulder,
            left_elbow, right_elbow,
            left_wrist, right_wrist
        )
        
        # Classify overall posture
        posture = self._classify_posture(openness, lean_angle, arm_position)
        
        # Confidence based on visibility of key landmarks
        confidence = np.mean([
            landmarks[11].visibility,
            landmarks[12].visibility,
            landmarks[23].visibility,
            landmarks[24].visibility
        ])
        
        return BodyLanguageResult(
            posture=posture,
            lean_angle=lean_angle,
            shoulder_openness=openness,
            arm_position=arm_position,
            head_tilt=head_tilt,
            confidence=confidence
        )
    
    def _analyze_arm_position(
        self,
        left_shoulder, right_shoulder,
        left_elbow, right_elbow,
        left_wrist, right_wrist
    ) -> str:
        """Determine arm position state."""
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        
        # Check if arms are crossed (wrists near opposite shoulders)
        left_wrist_to_right = np.linalg.norm(left_wrist[:2] - right_shoulder[:2])
        right_wrist_to_left = np.linalg.norm(right_wrist[:2] - left_shoulder[:2])
        
        if left_wrist_to_right < 0.2 and right_wrist_to_left < 0.2:
            return "crossed"
        
        # Check if arms are open (elbows away from body)
        left_elbow_dist = np.linalg.norm(left_elbow[:2] - shoulder_mid[:2])
        right_elbow_dist = np.linalg.norm(right_elbow[:2] - shoulder_mid[:2])
        
        if left_elbow_dist > 0.3 and right_elbow_dist > 0.3:
            return "open_wide"
        
        # Check if hands are at rest
        if left_wrist[1] > left_elbow[1] and right_wrist[1] > right_elbow[1]:
            return "at_sides"
        
        return "neutral"
    
    def _classify_posture(
        self,
        openness: float,
        lean_angle: float,
        arm_position: str
    ) -> PostureState:
        """Classify body posture state."""
        # Forward lean with open shoulders = engaged
        if lean_angle > 5 and openness > 0.6:
            return PostureState.ENGAGED_FORWARD
        
        # Backward lean = disengaged
        if lean_angle < -5:
            return PostureState.DISENGAGED_BACK
        
        # Crossed arms = defensive
        if arm_position == "crossed":
            return PostureState.CLOSED_DEFENSIVE
        
        # Open shoulders and arms = confident
        if openness > 0.7 and arm_position in ["open_wide", "at_sides"]:
            return PostureState.OPEN_CONFIDENT
        
        return PostureState.NEUTRAL


# ============================================================================
# PART 3: FACE DETECTION WITH NEW MEDIAPIPE API
# ============================================================================

class FaceDetector:
    """Face detection using MediaPipe Tasks API."""
    
    def __init__(self):
        self.model_path = self._get_model_path()
        
        base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5
        )
        self.detector = vision.FaceDetector.create_from_options(options)
    
    def _get_model_path(self) -> str:
        """Download face detection model if not present."""
        import urllib.request
        import os
        
        model_path = "blaze_face_short_range.tflite"
        
        if not os.path.exists(model_path):
            print("Downloading face detection model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")
        
        return model_path
    
    def detect_and_crop(self, frame: np.ndarray, padding: int = 30) -> Optional[np.ndarray]:
        """Detect face and return cropped region."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)
        
        if not results.detections:
            return None
        
        # Get first face
        detection = results.detections[0]
        bbox = detection.bounding_box
        
        h, w = frame.shape[:2]
        
        x_min = max(0, bbox.origin_x - padding)
        y_min = max(0, bbox.origin_y - padding)
        x_max = min(w, bbox.origin_x + bbox.width + padding)
        y_max = min(h, bbox.origin_y + bbox.height + padding)
        
        face = frame[y_min:y_max, x_min:x_max]
        
        if face.size > 0:
            face = cv2.resize(face, (224, 224))
            return face
        
        return None


# ============================================================================
# PART 4: GEMINI 3 API INTEGRATION
# ============================================================================

class GeminiEmotionAnalyzer:
    """
    Emotion analysis using Google Gemini 3 API.
    """
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API key."""
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            # Use Gemini 3 Flash Preview for hackathon
            self.model_flash = "gemini-3-flash-preview"
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")
    
    def analyze_face_image(
        self,
        image: np.ndarray,
        body_language: Optional[BodyLanguageResult] = None
    ) -> Tuple[EmotionResult, AccessibilityDescription]:
        """
        Analyze emotion from a face image using Gemini.
        """
        # Convert to JPEG for API
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Build prompt with body language if available
        body_info = ""
        if body_language:
            body_info = f"""
BODY LANGUAGE CONTEXT:
- Posture: {body_language.posture.value}
- Lean: {body_language.lean_angle:.1f}° (positive = forward lean)
- Shoulder openness: {body_language.shoulder_openness:.2f} (0-1 scale)
- Arms: {body_language.arm_position}
"""
        
        prompt = f"""Analyze this person's facial expression and emotional state.
{body_info}

Return a JSON object with this EXACT structure:
{{
    "primary_emotion": "happiness" or "sadness" or "anger" or "fear" or "surprise" or "disgust" or "neutral",
    "intensity": <number 1-7>,
    "confidence": <number 0-100>,
    "valence": <number -1 to 1>,
    "arousal": <number 0 to 1>,
    "action_units": {{"AU12": 0.8}},
    "accessibility": {{
        "for_deaf": "Description of emotional tone for deaf users",
        "for_blind": "Description of visual cues for blind users", 
        "for_autism": "Clear emotion label with social context",
        "summary": "One sentence summary"
    }}
}}

Key Action Units to detect:
- AU1/AU2: Brow raise (surprise, fear)
- AU4: Brow furrow (anger, sadness)
- AU6: Cheek raise (genuine smile)
- AU12: Lip corner pull (smile)
- AU15: Lip corner depress (sadness)
- AU26: Jaw drop (surprise)

Return ONLY valid JSON, no other text."""

        try:
            response = self.client.models.generate_content(
                model=self.model_flash,
                contents=[
                    {
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                            {"text": prompt}
                        ]
                    }
                ]
            )
            
            return self._parse_response(response.text)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._default_result()
    
    def _parse_response(self, response_text: str) -> Tuple[EmotionResult, AccessibilityDescription]:
        """Parse Gemini response into results."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)
            
            emotion_map = {e.value: e for e in Emotion}
            primary = emotion_map.get(data.get("primary_emotion", "neutral"), Emotion.NEUTRAL)
            
            emotion_result = EmotionResult(
                primary_emotion=primary,
                intensity=float(data.get("intensity", 4)),
                confidence=float(data.get("confidence", 50)) / 100,
                secondary_emotions=[],
                valence=float(data.get("valence", 0)),
                arousal=float(data.get("arousal", 0.5)),
                action_units=data.get("action_units", {})
            )
            
            acc = data.get("accessibility", {})
            accessibility = AccessibilityDescription(
                for_deaf=acc.get("for_deaf", "Unable to analyze"),
                for_blind=acc.get("for_blind", "Unable to analyze"),
                for_autism=acc.get("for_autism", "Unable to analyze"),
                overall_summary=acc.get("summary", "Unable to analyze")
            )
            
            return emotion_result, accessibility
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Parse error: {e}")
            print(f"Response was: {response_text[:500]}")
            return self._default_result()
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        # Remove markdown code blocks if present
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        
        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]
        
        return text.strip()
    
    def _default_result(self) -> Tuple[EmotionResult, AccessibilityDescription]:
        """Return default result when parsing fails."""
        return (
            EmotionResult(
                primary_emotion=Emotion.NEUTRAL,
                intensity=4.0,
                confidence=0.0,
                secondary_emotions=[],
                valence=0.0,
                arousal=0.5,
                action_units={}
            ),
            AccessibilityDescription(
                for_deaf="Could not analyze emotional tone",
                for_blind="Could not analyze visual cues",
                for_autism="Emotion unclear",
                overall_summary="Analysis unavailable"
            )
        )


# ============================================================================
# PART 5: COMPLETE PIPELINE
# ============================================================================

class AccessibilityEmotionPipeline:
    """
    Complete pipeline combining all components for accessibility app.
    """
    
    def __init__(self, gemini_api_key: str):
        print("Initializing pipeline...")
        print("Loading Gemini analyzer...")
        self.gemini = GeminiEmotionAnalyzer(gemini_api_key)
        print("Loading body language analyzer...")
        self.body_analyzer = BodyLanguageAnalyzer()
        print("Loading face detector...")
        self.face_detector = FaceDetector()
        print("Pipeline ready!")
        
        # Frame counter for batch processing
        self.frame_count = 0
        self.last_result = None
    
    def process_frame(
        self,
        frame: np.ndarray,
        analyze_every_n_frames: int = 30  # Analyze every 30 frames (~1 sec at 30fps) to prevent freezing
    ) -> Optional[Dict]:
        """
        Process a video frame through the complete pipeline.
        """
        self.frame_count += 1
        
        # Always update body language (runs locally, fast)
        body_result = self.body_analyzer.analyze(frame)
        
        # Only run Gemini analysis periodically (API rate limits + cost)
        if self.frame_count % analyze_every_n_frames == 0 or self.last_result is None:
            # Get face region
            face_image = self.face_detector.detect_and_crop(frame)
            
            if face_image is not None:
                # Run Gemini analysis
                emotion_result, accessibility = self.gemini.analyze_face_image(
                    face_image, body_result
                )
                
                self.last_result = {
                    "emotion": emotion_result,
                    "body_language": body_result,
                    "accessibility": accessibility,
                    "frame_number": self.frame_count
                }
        elif body_result and self.last_result:
            # Update body language in cached result
            self.last_result["body_language"] = body_result
        
        return self.last_result


# ============================================================================
# PART 6: DEMO / TESTING
# ============================================================================

def demo_webcam(gemini_api_key: str):
    """
    Demo function to run emotion detection on webcam.
    """
    print("\n" + "="*50)
    print("ACCESSIBILITY EMOTION DETECTION DEMO")
    print("="*50 + "\n")
    
    pipeline = AccessibilityEmotionPipeline(gemini_api_key)
    
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not in use by another app.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "-"*50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Force analysis (don't wait for auto-update)")
    print("-"*50)
    print("\nNote: Emotion analysis runs every ~1.5 seconds to keep video smooth.")
    print("Body language updates continuously.\n")
    
    force_analyze = False
    analysis_interval = 45  # Frames between Gemini API calls
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        analyze_interval = 1 if force_analyze else analysis_interval
        result = pipeline.process_frame(frame, analyze_every_n_frames=analyze_interval)
        force_analyze = False
        
        # Draw results on frame
        if result:
            emotion = result["emotion"].primary_emotion.value
            confidence = result["emotion"].confidence
            intensity = result["emotion"].intensity
            
            # Emotion text
            color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
            cv2.putText(
                frame,
                f"Emotion: {emotion.upper()}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%} | Intensity: {intensity:.1f}/7",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Valence/Arousal
            valence = result["emotion"].valence
            arousal = result["emotion"].arousal
            valence_text = "Positive" if valence > 0.2 else "Negative" if valence < -0.2 else "Neutral"
            arousal_text = "High energy" if arousal > 0.6 else "Low energy" if arousal < 0.4 else "Moderate"
            cv2.putText(
                frame,
                f"Mood: {valence_text}, {arousal_text}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Body language
            if result["body_language"]:
                posture = result["body_language"].posture.value.replace("_", " ").title()
                cv2.putText(
                    frame,
                    f"Posture: {posture}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 0),
                    2
                )
            
            # Print accessibility info periodically
            if pipeline.frame_count % 45 == 0:  # Every ~1.5 seconds at 30fps
                print("\n" + "="*50)
                print("ACCESSIBILITY DESCRIPTIONS")
                print("="*50)
                print(f"\n🦻 FOR DEAF/HARD-OF-HEARING USERS:")
                print(f"   {result['accessibility'].for_deaf}")
                print(f"\n👁️ FOR BLIND/LOW-VISION USERS:")
                print(f"   {result['accessibility'].for_blind}")
                print(f"\n🧩 FOR AUTISM SPECTRUM USERS:")
                print(f"   {result['accessibility'].for_autism}")
                print(f"\n📝 SUMMARY: {result['accessibility'].overall_summary}")
                print("="*50)
        
        # Show frame counter
        cv2.putText(
            frame,
            f"Frame: {pipeline.frame_count} | Press 'q' to quit, 's' to analyze now",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1
        )
        
        cv2.imshow("Accessibility Emotion Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            force_analyze = True
            print("\n[Forcing analysis on next frame...]")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


def test_installation():
    """Test that all dependencies are installed correctly."""
    print("Testing installation...")
    
    # Test MediaPipe
    print("\n1. Testing MediaPipe...")
    try:
        import mediapipe as mp
        print(f"   ✓ MediaPipe version: {mp.__version__}")
    except Exception as e:
        print(f"   ✗ MediaPipe error: {e}")
        return False
    
    # Test OpenCV
    print("\n2. Testing OpenCV...")
    try:
        import cv2
        print(f"   ✓ OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"   ✗ OpenCV error: {e}")
        return False
    
    # Test Google GenAI
    print("\n3. Testing Google GenAI...")
    try:
        from google import genai
        print(f"   ✓ Google GenAI installed")
    except Exception as e:
        print(f"   ✗ Google GenAI error: {e}")
        print("   Install with: pip install google-genai")
        return False
    
    # Test webcam
    print("\n4. Testing webcam...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   ✓ Webcam accessible")
        cap.release()
    else:
        print("   ✗ Webcam not accessible")
        return False
    
    print("\n" + "="*50)
    print("All tests passed! Ready to run.")
    print("="*50)
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_installation()
        else:
            api_key = sys.argv[1]
            demo_webcam(api_key)
    else:
        print("="*50)
        print("ACCESSIBILITY EMOTION DETECTION")
        print("="*50)
        print("\nUsage:")
        print("  python emotion_detection_fixed.py YOUR_GEMINI_API_KEY")
        print("  python emotion_detection_fixed.py --test")
        print("\nTo get a Gemini API key: https://ai.google.dev/")
        print("\nRequired packages:")
        print("  pip install google-genai mediapipe opencv-python numpy")
