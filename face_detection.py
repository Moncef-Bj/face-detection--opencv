import cv2
import sys
import argparse
import os

def parse_arguments():
    """
    Command line arguments:
    --image <path>: Path to an image file to detect faces on
    --cascade <path>: Path to the Haar cascade file (default: haarcascade_frontalface_default.xml)
    """
    parser = argparse.ArgumentParser(description="Simple face detection using OpenCV.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file. If not provided, webcam will be used.")
    parser.add_argument("--cascade", type=str, default="haarcascade_frontalface_default.xml",
                        help="Path to Haar cascade file.")
    return parser.parse_args()

def detect_faces_in_image(image_path, cascade_path):
    """
    Detect faces in a static image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Number of faces detected: {len(faces)}")

    # Draw rectangles around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_from_webcam(cascade_path):
    """
    Detect faces in real-time using the default webcam.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access the webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Webcam Face Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()

    # Check if cascade file exists
    if not os.path.exists(args.cascade):
        print(f"Haar cascade file not found: {args.cascade}")
        print("Download it from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
        sys.exit(1)

    if args.image:
        detect_faces_in_image(args.image, args.cascade)
    else:
        detect_faces_from_webcam(args.cascade)

if __name__ == "__main__":
    main()
