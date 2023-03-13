from sanic import Sanic
from sanic.response import HTTPResponse, text, json
from sanic.request import Request
from sanic_ext import Extend
from factory.face import detect, detectDnn, detectMTCNN
from sanic.log import logger
import os, time, datetime, logging, base64

from sanic.worker.manager import WorkerManager

WELCOME_MESSAGE = r"""

███████╗░█████╗░░█████╗░███████╗░██████╗███████╗███╗░░██╗░██████╗███████╗
██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝████╗░██║██╔════╝██╔════╝
█████╗░░███████║██║░░╚═╝█████╗░░╚█████╗░█████╗░░██╔██╗██║╚█████╗░█████╗░░
██╔══╝░░██╔══██║██║░░██╗██╔══╝░░░╚═══██╗██╔══╝░░██║╚████║░╚═══██╗██╔══╝░░
██║░░░░░██║░░██║╚█████╔╝███████╗██████╔╝███████╗██║░╚███║██████╔╝███████╗
╚═╝░░░░░╚═╝░░╚═╝░╚════╝░╚══════╝╚═════╝░╚══════╝╚═╝░░╚══╝╚═════╝░╚══════╝
"""

print(WELCOME_MESSAGE)

LOGGING_CONFIG= logging.basicConfig(
            filename=datetime.datetime.now().strftime('log_%Y-%m-%d.log'),
            filemode="a",
            level=logging.DEBUG,
            format="%(asctime)s:%(levelname)s:%(message)s",
            datefmt="%Y-%m-%d %I:%M:%S%p",
        )
        
WorkerManager.THRESHOLD = 100  # 这个值对应着 0.1s

app = Sanic(__name__, log_config=LOGGING_CONFIG)
app.config.CORS_ORIGINS = "*"
Extend(app)
app.config["UPLOAD_FOLDER"] = "upload"

app.static('/public', './public')

@app.get("/")
@app.ext.template("main.html")
async def hello_world(request):
    return {"seq": ["one", "two"]}

@app.get("/typed")
async def typed_handler(request: Request) -> HTTPResponse:
    return text("Done.")

@app.route("/face/v1/detect", methods=['POST'])
async def facedetect(request):
    # uploadFolder = app.config['UPLOAD_FOLDER']
   
    # fullPath = await write_to_file(uploadFolder, request)
    # if os.path.exists(fullPath):
    #     print("File Exists:" + fullPath)
    # result = detect(fullPath)
    id = request.form.get('id')
    result = {}
    logger.info("id: " + id)
    start_time = time.time()
    faceDetectResult = detectDnn(request.files["file"][0].body)

    if (len(faceDetectResult) < 2):
        faceDetectResult2 = detectMTCNN(request.files["file"][0].body)
        if (len(faceDetectResult2) > len(faceDetectResult)):
            faceDetectResult = faceDetectResult2

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time: " + str(elapsed_time))
    result["face"] = faceDetectResult
    result["id"] = id
    result["elapsed_time"] = elapsed_time
    return json(result)

@app.route("/face/v1/detectjson", methods=['POST'])
async def facedetectjson(request):
    """
    openapi:
    ---
    operationId: facedetectjson
    requestBody:
        description: "Mykad face detection"
        content:
          application/json:
            schema:
              type: object
              properties:
                id:
                  type: string
                image:
                  type: string
                  format: base64
              required:
                - id
                - image
            example:
              id: "Unique ID"
              image: "base64 of image"
   
    responses:
      '200':
        description: json result
    
    """
    result = {}
    reqJson = request.json
    
    logger.info("id: " + reqJson["id"])
    start_time = time.time()
    bytes_data = base64.b64decode(reqJson["image"])
    faceDetectResult = detectDnn(bytes_data)

    if (len(faceDetectResult) < 2):
        faceDetectResult2 = detectMTCNN(bytes_data)
        if (len(faceDetectResult2) > len(faceDetectResult)):
            faceDetectResult = faceDetectResult2

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time: " + str(elapsed_time))
    result["face"] = faceDetectResult
    result["id"] = reqJson["id"]
    result["elapsed_time"] = elapsed_time
    return json(result)

async def write_to_file(uploadFolder, request):   
    import aiofiles
    if not os.path.exists(uploadFolder):
        os.makedirs(uploadFolder)
    async with aiofiles.open(uploadFolder +"/"+request.files["file"][0].name, 'wb') as f:
        await f.write(request.files["file"][0].body)
    f.close()
    filename = request.files["file"][0].name
    fullPath = uploadFolder +"/"+ filename
    return fullPath

if __name__ == "__main__":
  
    app.run(host='0.0.0.0', port=8000)