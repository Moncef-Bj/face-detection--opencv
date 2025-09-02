import os
import cv2
import argparse
from pathlib import Path

def validate_image_file(file_path):
    """Validate image file before processing"""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in valid_extensions:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Check file size (prevent memory exhaustion)
    file_size = os.path.getsize(file_path)
    max_size = 50 * 1024 * 1024  # 50MB limit
    if file_size > max_size:
        raise ValueError(f"File too large: {file_size/1024/1024:.1f}MB (max: 50MB)")
    
    return True

def validate_output_path(output_path):
    """Validate output path and prevent overwriting"""
    if os.path.exists(output_path):
        response = input(f"File {output_path} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    return True
2. Improved Error Handling & Resource Management
pythonimport logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)

@contextmanager
def safe_video_capture(camera_index=0):
    """Context manager for safe camera resource handling"""
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot access camera {camera_index}")
        
        # Set reasonable resolution to prevent memory issues
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        yield cap
    except Exception as e:
        logging.error(f"Camera error: {e}")
        raise
    finally:
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()

def detect_faces_in_image(image_path, output_path=None, **detection_params):
    """Detect faces in image with proper error handling"""
    try:
        # Validate inputs
        validate_image_file(image_path)
        if output_path and not validate_output_path(output_path):
            return None
        
        # Load cascade classifier
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("Haar cascade file not found. Please ensure haarcascade_frontalface_default.xml is in the current directory.")
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=detection_params.get('scale', 1.2),
            minNeighbors=detection_params.get('neighbors', 8),
            minSize=(detection_params.get('min_size', 30), detection_params.get('min_size', 30))
        )
        
        logging.info(f"Detected {len(faces)} face(s) in {image_path}")
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, image)
            logging.info(f"Result saved to {output_path}")
        else:
            cv2.imshow('Face Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return len(faces)
        
    except Exception as e:
        logging.error(f"Face detection failed: {e}")
        return None
3. Safe Webcam Detection
pythondef detect_faces_webcam():
    """Real-time face detection with webcam"""
    try:
        # Inform user about camera access
        print("Starting webcam face detection...")
        print("Press 'q' to quit, 'f' for fullscreen, 's' to save screenshot")
        
        with safe_video_capture() as cap:
            # Load cascade classifier
            cascade_path = 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read frame from camera")
                    break
                
                # Process every 3rd frame for performance
                frame_count += 1
                if frame_count % 3 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.2, 8, minSize=(30, 30))
                    
                    # Draw rectangles
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Add face count to display
                    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Face Detection - Press q to quit', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = cv2.getTickCount()
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logging.info(f"Screenshot saved: {filename}")
                    
    except KeyboardInterrupt:
        logging.info("Detection stopped by user")
    except Exception as e:
        logging.error(f"Webcam detection error: {e}")
4. Enhanced CLI Interface
pythondef main():
    """Main function with improved argument parsing"""
    parser = argparse.ArgumentParser(
        description='Face Detection using OpenCV Haar Cascades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python face_detection.py --webcam
  python face_detection.py --image photo.jpg --output result.jpg
  python face_detection.py --image photo.jpg --scale 1.3 --neighbors 5
        '''
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--webcam', action='store_true', 
                           help='Use webcam for real-time detection')
    mode_group.add_argument('--image', type=str, 
                           help='Path to input image file')
    
    # Parameters
    parser.add_argument('--output', type=str, 
                       help='Path to save output image (only for image mode)')
    parser.add_argument('--scale', type=float, default=1.2, 
                       help='Scale factor for detection (default: 1.2)')
    parser.add_argument('--neighbors', type=int, default=8, 
                       help='Minimum neighbors for detection (default: 8)')
    parser.add_argument('--min-size', type=int, default=30, 
                       help='Minimum face size in pixels (default: 30)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.webcam:
            detect_faces_webcam()
        else:
            detection_params = {
                'scale': args.scale,
                'neighbors': args.neighbors,
                'min_size': args.min_size
            }
            result = detect_faces_in_image(args.image, args.output, **detection_params)
            if result is not None:
                print(f"Detection completed. Found {result} face(s).")
            else:
                print("Detection failed. Check logs for details.")
                
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
