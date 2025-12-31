"""FastAPI main application."""

from fastapi import FastAPI

from bayescal.api import endpoints

app = FastAPI(
    title="BayesCal API",
    description="API for Bayesian Neural Network Calibration Research",
    version="0.1.0",
)

app.include_router(endpoints.router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "BayesCal API"}

