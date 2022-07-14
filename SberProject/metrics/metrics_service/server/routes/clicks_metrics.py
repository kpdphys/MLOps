from fastapi import APIRouter
from fastapi.responses import JSONResponse

from metrics_service.server.database import retrieve_clicks_statistics
from metrics_service.server.models.search_result import ResponseModel

router = APIRouter()


@router.get("/", response_description="Search metrics retrieved")
async def get_search_metrics():
    clicks_statistics = await retrieve_clicks_statistics()
    sum_all = 0
    sum_clicked = 0
    avg_clicked = 0
    ctr = []

    for k in clicks_statistics.keys():
        sum_all = sum_all + clicks_statistics[k]
        avg_clicked = avg_clicked + k * clicks_statistics[k]
        if k != 0:
            sum_clicked = sum_clicked + clicks_statistics[k]
            ctr.append(sum_clicked)
    avg_clicked = round(avg_clicked / (sum_clicked if sum_clicked > 0 else 1), 4)
    ctr = [round(i / (sum_all if sum_all > 0 else 1), 4) for i in ctr]
    metrics_result = {"searches_count": sum_all, "CTR_i": ctr, "AHC": avg_clicked}
    return JSONResponse(content=ResponseModel(metrics_result, 200, "Metrics were computed successfully"),
                        status_code=200)
