import cv2
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple face detection using OpenCV.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file. If not provided, webcam will be used.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save annotated image (headless mode if set).")
    parser.add_argument("--cascade", type=str, default="haarcascade_frontalface_default.xml",
                        help="Path to Haar cascade file.")
    parser.add_argument("--scale", type=float, default=1.2,
                        help="Scale factor for detectMultiScale (e.g., 1.1–1.3).")
    parser.add_argument("--neighbors", type=int, default=8,
                        help="minNeighbors for detectMultiScale (higher = stricter).")
    parser.add_argument("--min-size", type=int, default=30,
                        help="Minimum face size in pixels.")
    return parser.parse_args()

def resolve_cascade(cascade_arg: str) -> str:
    if os.path.exists(cascade_arg):
        return cascade_arg
    return os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

def detect_faces_in_image(image_path, cascade_path, output_path=None,
                          scale=1.2, neighbors=8, min_size=30):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optionnel : améliore le contraste
    # gray = cv2.equalizeHist(gray)

    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=(min_size, min_size),
    )
    print(f"Number of faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")
    else:
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_faces_from_webcam(cascade_path, output_path=None,
                             scale=1.2, neighbors=8, min_size=30):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access the webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Headless: capture 1 frame, annote, save, exit
    if output_path:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=scale, minNeighbors=neighbors,
                minSize=(min_size, min_size)
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
        cap.release()
        return

    # GUI branch
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors,
            minSize=(min_size, min_size)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Webcam Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    cascade_path = resolve_cascade(args.cascade)

    if args.image:
        detect_faces_in_image(
            args.image, cascade_path, args.output,
            scale=args.scale, neighbors=args.neighbors, min_size=args.min_size
        )
    else:
        detect_faces_from_webcam(
            cascade_path, args.output,
            scale=args.scale, neighbors=args.neighbors, min_size=args.min_size
        )

if __name__ == "__main__":
    main()
