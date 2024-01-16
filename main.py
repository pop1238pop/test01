from uvicorn import Server, Config
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os

from yolofastapi.routers import yolo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebhookPayload(BaseModel):
    event: str
    data: dict

@app.post("/webhook")
async def handle_webhook(payload: WebhookPayload):
    # ประมวลผล Webhook payload
    event = payload.event
    data = payload.data

    # เพิ่มโลจิกที่กำหนดเองที่นี่ตามข้อมูล Webhook ที่ได้รับ
    # ...

    return {"message": f"ได้รับเหตุการณ์ Webhook: {event}"}
app.include_router(yolo.router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    server = Server(Config(app, host="0.0.0.0", port=port, lifespan="on"))
    server.run()
