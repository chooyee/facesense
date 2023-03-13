import cv2
import base64
import numpy as np
import uuid
import threading
from sanic.log import logger
from mtcnn import MTCNN

path = "upload"

def detectMTCNN(image_data):
    """Read the image into OpenCV then detect human face. Return as base64"""
    logger.info('detectMTCNN')
    try:
        # Convert the image data to a numpy array using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = MTCNN()
        faces = detector.detect_faces(img)
        cropped_faces = []
        icnt = 1
        filename = uuid.uuid4().hex

        logger.info(f'Face found: {len(faces)}')
        for v in faces:          
            x, y, w, h = v['box']           
            cropped_face = img[y:y+h, x:x+w]
            success, encoded_face = cv2.imencode('.jpg', cropped_face)
            if success:
                base64_face = base64.b64encode(encoded_face).decode('utf-8')
                faceObj = {}
                faceObj["width"] = str(w)
                faceObj["height"] = str(h)
                faceObj["base64"] = base64_face
                cropped_faces.append(faceObj)

                # cropped_path = f'{path}/{filename}-cropped-{str(icnt)}.jpg'
                # cv2.imwrite(cropped_path, cropped_face)
                icnt+=1   
        return cropped_faces
    except Exception as e:
        logger.error(f'detectMTCNN error: {e}')
        raise

def detect(image_data):
    """Read the image into OpenCV then detect human face. Return as base64"""
    logger.info('Detect')
    try:
        # Convert the image data to a numpy array using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img = cv2.imread(faceImage)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Crop the faces and encode them in base64 format
        cropped_faces = []
        i = 1
        filename = uuid.uuid4().hex

        logger.info(f'Face found: {len(faces)}')
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            
            success, encoded_face = cv2.imencode('.jpg', face)
            if success:
                base64_face = base64.b64encode(encoded_face).decode('utf-8')
                faceObj = {}
                faceObj["width"] = str(w)
                faceObj["height"] = str(h)
                faceObj["base64"] = base64_face
                cropped_faces.append(faceObj)
                
                #cropped_path = f'{path}/{filename}-cropped-{str(i)}.jpg'
                #cv2.imwrite(cropped_path, face)
                #save_image_async(face, cropped_path)
                i += 1
        return cropped_faces
    except Exception as e:
        logger.error(f'Detect error: {e}')
        raise

def detectDnn(image_data):
    logger.info(f'Detect Dnn')
    try:
        model = cv2.dnn.readNetFromCaffe("model/deploy.prototxt.txt", "model/res10_300x300_ssd_iter_140000.caffemodel")

        # Load the input image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Prepare the input image for face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Use the deep learning face detection model to detect faces in the input image
        model.setInput(blob)
        detections = model.forward()

        cropped_faces = []
        icnt = 1
        filename = uuid.uuid4().hex
        # Extract the bounding box coordinates of each detected face
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Crop the detected faces from the input image
                face = img[startY:endY, startX:endX]
                success, encoded_face = cv2.imencode('.jpg', face)
                if success:
                    base64_face = base64.b64encode(encoded_face).decode('utf-8')
                    faceObj = {}
                    faceObj["width"] = str(endX - startX)
                    faceObj["height"] = str(endY - startY)
                    faceObj["base64"] = base64_face
                    cropped_faces.append(faceObj)

                    # cropped_path = f'{path}/{filename}-cropped-{str(icnt)}.jpg'
                    # cv2.imwrite(cropped_path, face)
                    #write_to_file(cropped_path, face)
                    icnt+=1   
        logger.info(f'Face found: {icnt-1}')        
        return cropped_faces
    except Exception as e:
        logger.error(f'Detect Dnn error: {e}')
        raise

def save_image_async(img, filename):
    t = threading.Thread(target=cv2.imwrite, args=(filename, img))
    t.start()

async def write_to_file(img, filename):   
    import aiofiles   
    async with aiofiles.open(filename, 'wb') as f:
        await f.write(img)
    f.close()
