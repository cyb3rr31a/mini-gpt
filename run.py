import uvicorn # type:ignore
import numpy as np # type:ignore

if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)