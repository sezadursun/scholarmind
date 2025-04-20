# app/model_selector.py

def get_model(task_type: str) -> str:
    """
    Verilen görev türüne göre uygun model adını döndürür.
    """
    task_type = task_type.lower()

    if task_type in ["detailed_summary", "long_analysis"]:
        return "gpt-4o"
    
    elif task_type in ["short_summary", "qa", "title"]:
        return "gpt-3.5-turbo"
    
    elif task_type == "embedding":
        return "text-embedding-3-small"

    else:
        # Varsayılan model (isteğe göre değiştirilebilir)
        return "gpt-3.5-turbo"
