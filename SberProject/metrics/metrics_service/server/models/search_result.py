from typing import Optional
from pydantic import BaseModel, AnyHttpUrl, Field


class ResultModel(BaseModel):
    title: str = Field(...)
    uri: AnyHttpUrl = Field(...)


class SearchResultSchema(BaseModel):
    query: str = Field(...)
    result: list[ResultModel]
    link_pos: int = Field(..., ge=0, le=5)

    class Config:
        schema_extra = {
            "example": {
                "query": "подарок на день рождения",
                "result": [
                    {
                        "title": "СберМаркет",
                        "uri": "https://spasibosberbank.ru/partners/sbermarket",
                    },
                    {
                        "title": "СберМаркет2",
                        "uri": "https://spasibosberbank.ru/partners/sbermarket",
                    },
                ],
                "link_pos": 1,
            }
        }


class UpdateSearchResultSchemaModel(BaseModel):
    query: Optional[str]
    result: Optional[list[ResultModel]]
    link_pos: Optional[int]

    class Config:
        schema_extra = {
            "example": {
                "query": "подарок на день рождения",
                "result": [
                    {
                        "title": "СберМаркет",
                        "uri": "https://spasibosberbank.ru/partners/sbermarket",
                    },
                    {
                        "title": "СберМаркет2",
                        "uri": "https://spasibosberbank.ru/partners/sbermarket",
                    },
                ],
                "link_pos": 2,
            }
        }


def ResponseModel(data, code, message):
    return {
        "data": data,
        "code": code,
        "message": message,
    }


def ErrorResponseModel(error, code, message):
    return {"error": error, "code": code, "message": message}
