# TU Study Assistant

Start the application:

```bash
uv run streamlit run app.py
```

`app.py` configures the Streamlit server, including the initial page title and
browser icons. `study_manager.py` contains the application UI. Use `app.py` when
launching or deploying so Safari receives the correct icon on the first load.

Run tests:

```bash
uv run pytest
```
