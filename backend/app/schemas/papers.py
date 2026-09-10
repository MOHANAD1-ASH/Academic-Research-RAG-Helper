from pydantic import BaseModel, Field


class PaperResponse(BaseModel):
    paper_id: str = Field(alias="paperId")
    title: str
    year: int | None = None
    authors: list[dict[str, str]] = Field(default_factory=list)
    url: str = ""
    pdf_url: str = ""
    doi: str | None = None
    abstract: str = ""

    model_config = {"populate_by_name": True}


class IngestResponse(BaseModel):
    ok: bool
    message: str
    paper_id: str | None = None
    title: str | None = None
    n_chunks: int | None = None
    already_indexed: bool = False
    repaired: bool = False
