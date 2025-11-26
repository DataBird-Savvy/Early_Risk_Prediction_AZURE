def explainability_css():
    return """
    <style>
    .explain-box {
        background-color: #141a25;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4ea8de;
    }

    .explain-item {
        font-size: 16px;
        padding: 6px 0;
        color: #e2e8f0;
        display: flex;
        justify-content: space-between;
    }

    .explain-feature {
        font-weight: 600;
        color: #93c5fd;
    }

    .explain-importance {
        color: #f9d976;
        font-weight: 600;
    }
    </style>
    """
