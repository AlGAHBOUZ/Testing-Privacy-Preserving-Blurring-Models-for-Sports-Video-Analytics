
import os
import cv2
import numpy as np


class RegularApproach:
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load multiple Haar cascades for better detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
    
    def detect_faces_multi_scale(self, gray):
        """
        Detect faces using multiple cascades and scale factors for robustness.
        Returns a list of unique face rectangles.
        """
        all_faces = []
        
        # Primary detector with multiple scale factors
        for scale in [1.05, 1.1, 1.2, 1.3]:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=3,  # Lower threshold to catch more faces
                minSize=(20, 20),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend(faces)
        
        # Alternative frontal face detector
        faces_alt = self.face_cascade_alt.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces_alt)
        
        # Profile face detector (left and right profiles)
        # Detect left profiles
        faces_profile = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        all_faces.extend(faces_profile)
        
        # Detect right profiles (flip image)
        gray_flipped = cv2.flip(gray, 1)
        faces_profile_flipped = self.profile_cascade.detectMultiScale(
            gray_flipped,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        # Flip coordinates back
        for (x, y, w, h) in faces_profile_flipped:
            flipped_x = gray.shape[1] - x - w
            all_faces.append((flipped_x, y, w, h))
        
        # Remove duplicate/overlapping detections using Non-Maximum Suppression
        if len(all_faces) == 0:
            return []
        
        all_faces = np.array(all_faces)
        return self.non_max_suppression(all_faces, overlap_thresh=0.3)
    
    def non_max_suppression(self, boxes, overlap_thresh=0.3):
        """
        Remove overlapping bounding boxes using Non-Maximum Suppression.
        """
        if len(boxes) == 0:
            return []
        
        boxes = boxes.astype(float)
        pick = []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        area = boxes[:, 2] * boxes[:, 3]
        
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
        return boxes[pick].astype(int)
    
    def apply_gaussian_blur(self, img, x, y, w, h, blur_strength=51):
        """
        Apply Gaussian blur to face region with padding for better effect.
        """
        # Add padding around face (10% on each side)
        padding = int(0.1 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # Extract ROI
        roi = img[y1:y2, x1:x2].copy()
        
        # Ensure blur_strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Apply strong Gaussian blur
        roi_blur = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        
        # Optional: Apply additional pixelation for stronger anonymization
        # Uncomment the following lines for pixelation effect
        # small = cv2.resize(roi_blur, (blur_strength, blur_strength), interpolation=cv2.INTER_LINEAR)
        # roi_blur = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Place blurred ROI back
        img[y1:y2, x1:x2] = roi_blur
        
        return img
    
    def process_img(self, img, show_count=False):
        """
        Process image and blur all detected faces.
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection in varied lighting
        gray = cv2.equalizeHist(gray)
        
        # Detect faces using multi-scale approach
        faces = self.detect_faces_multi_scale(gray)
        
        # Blur each detected face
        for (x, y, w, h) in faces:
            img = self.apply_gaussian_blur(img, x, y, w, h, blur_strength=71)
        
        # Optionally show face count
        if show_count and len(faces) > 0:
            cv2.putText(
                img,
                f"Faces blurred: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        return img, len(faces)
    
    def process_image(self, file_path):
        """Process a single image file."""
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Could not read image from {file_path}")
            return None
        
        img, face_count = self.process_img(img, show_count=False)
        
        output_path = os.path.join(self.output_dir, "output_blurred.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Processed image: {face_count} face(s) blurred")
        print(f"✓ Saved to {output_path}")
        return img
    
    def process_video(self, file_path):
        """Process a video file."""
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read video.")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = os.path.join(self.output_dir, "output_blurred.mp4")
        output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame.shape[1], frame.shape[0]),
        )

        frame_count = 0
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while ret:
            frame, faces = self.process_img(frame, show_count=False)
            output_video.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Progress: {frame_count}/{total_frames} frames ({faces} faces in current frame)")
            
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        print(f"✓ Video processing complete!")
        print(f"✓ Saved to {output_path}")
    
    def process_webcam(self):
        """Process live webcam feed."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam active. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, face_count = self.process_img(frame, show_count=True)
            cv2.imshow("Face Blur (Press 'q' to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def run(self, mode="image", file_path="data/erling.jpg"):
        """
        Run face blurring in specified mode.
        
        Args:
            mode: 'image', 'video', or 'webcam'
            file_path: path to input file (for image/video modes)
        """
        if mode == "image":
            self.process_image(file_path)
        elif mode == "video":
            self.process_video(file_path)
        elif mode == "webcam":
            self.process_webcam()
        else:
            print(f"Error: Unknown mode '{mode}'. Choose from: image, video, webcam")