from fastapi import APIRouter, Body, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from metrics_service.server.database import (
    add_search_result,
    delete_search_result,
    retrieve_search_results,
    retrieve_search_result,
    update_search_result,
)

from metrics_service.server.models.search_result import (
    ErrorResponseModel,
    ResponseModel,
    SearchResultSchema,
    UpdateSearchResultSchemaModel,
)

router = APIRouter()


@router.post("/", response_description="Search result data added into the database.")
async def add_search_result_data(request: Request, search_result: SearchResultSchema = Body(...)):
    search_result = jsonable_encoder(search_result)
    print(search_result)
    new_search_result = await add_search_result(search_result)
    new_search_result_location = request.url.path
    return JSONResponse(content=ResponseModel(new_search_result, 201, "Search result added successfully."),
                        status_code=201,
                        headers={"Location": new_search_result_location + new_search_result["id"]}
                        )


@router.get("/", response_description="Search results retrieved")
async def get_search_results():
    search_results = await retrieve_search_results()
    if search_results:
        return ResponseModel(search_results, 200, "Search results data retrieved successfully")
    return ResponseModel(search_results, 200, "Empty list returned")


@router.get("/{id}", response_description="Search result data retrieved")
async def get_search_result_data(id):
    search_result = await retrieve_search_result(id)
    if search_result:
        return ResponseModel(search_result, 200, "Search result data retrieved successfully")
    return ErrorResponseModel("An error occurred.", 404, "Search result doesn't exist.")


@router.put("/{id}")
async def update_search_result_data(id: str, req: UpdateSearchResultSchemaModel = Body(...)):
    req = {k: v for k, v in req.dict().items() if v is not None}
    updated_search_result = await update_search_result(id, req)
    if updated_search_result:
        return ResponseModel(
            "Search result with ID: {} name update is successful".format(id),
            200,
            "Search result name updated successfully",
        )
    return ErrorResponseModel(
        "An error occurred",
        404,
        "There was an error updating the search result data.",
    )


@router.delete("/{id}", response_description="Search result data deleted from the database")
async def delete_search_result_data(id: str):
    deleted_search_result = await delete_search_result(id)
    if deleted_search_result:
        return ResponseModel(
            "Search result with ID: {} removed".format(id), 204, "Search result deleted successfully"
        )
    return ErrorResponseModel(
        "An error occurred", 404, "Search result with id {0} doesn't exist".format(id)
    )
