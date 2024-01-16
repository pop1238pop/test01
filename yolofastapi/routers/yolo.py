# For API operations and standards
from fastapi import APIRouter, UploadFile, Response, status, HTTPException
# Our detector objects
from yolofastapi.detectors import yolov8
# For encoding images
import cv2
# For response schemas
from yolofastapi.schemas.yolo import ImageAnalysisResponse
from linebot import LineBotApi
from linebot.models import TextSendMessage

CHANNEL_ACCESS_TOKEN = "NXPZK6DUO2XBHZ50aea7aOuXyZeGSxMjo2/OoDQ8unQL3jpsEGOjkq12PFurLHC3U7O1q/vqtxzETOIo0ptBZPp3m7qjtNVkYoZ+3+kZfFfV91HazW/m1mYEV/pgehSJouDtQZc4cWqNrT9mUTyYeQdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "224ef83e27180960722019da70d1faf9"

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)

router = APIRouter(tags=["Image Upload and analysis"], prefix="/yolo")
images = []

async def send_message_to_user(user_id, message):
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=message))
    except Exception as e:
        print(f"Error sending message: {str(e)}")

@router.post("/",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Successfully Analyzed Image."}
    },
    response_model=ImageAnalysisResponse,
)
async def yolo_image_upload(file: UploadFile) -> ImageAnalysisResponse:
    contents = await file.read()
    dt = yolov8.YoloV8ImageObjectDetection(chunked=contents)
    frame, labels = await dt()
    success, encoded_image = cv2.imencode(".png", frame)
    images.append(encoded_image)
    
    # เพิ่มการส่งข้อความกลับไปยังผู้ใช้
    user_id = "USER_ID_TO_SEND_MESSAGE_TO"
    message = "Image analysis completed. Labels: " + ", ".join(labels)
    await send_message_to_user(user_id, message)
    
    return ImageAnalysisResponse(id=len(images), labels=labels)


@router.get(
    "/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Image ID Not Found."}
    },
    response_class=Response,
)
async def yolo_image_download(image_id: int) -> Response:

    try:
        return Response(content=images[image_id - 1].tobytes(), media_type="image/png")
    except IndexError:
        raise HTTPException(status_code=404, detail="Image not found") 