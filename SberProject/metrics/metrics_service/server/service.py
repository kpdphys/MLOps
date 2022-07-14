from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from metrics_service.server.routes.search_result import router as search_result_router
from metrics_service.server.routes.clicks_metrics import router as metrics_router
from metrics_service.server.database import add_index, user, password, mongo_uris


app = FastAPI()

# Enabling CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_result_router, tags=["SearchResult"], prefix="/searches")
app.include_router(metrics_router, tags=["Metrics"], prefix="/metrics")


@app.on_event("startup")
async def create_index():
    # add index to link_pos element
    return await add_index()


@app.get("/", tags=["Root"])
async def read_healthcheck():
    return {"status": "Green", "version": "1.0"}
